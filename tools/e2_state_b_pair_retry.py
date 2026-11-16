"""复用已验证的f68梯度，只重做同一提案与评测；保留原实验全部文件。"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch

from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha


PROTOCOL_SHA = 'acdbf13ae5d7f0c893c627957fc8e50d4431004266a7bc845514d039f6146fc9'
SOURCES = ('e2_b_pair_finetune_losses.py', 'e2_state_b_pair_finetune.py',
           'e2_state_b_pair_evaluation.py', 'e2_state_b_pair_summary.py', 'e2_state_b_pair_retry.py')


def copy_verified(source, destination, expected):
    assert sha(source) == expected, str(source)
    assert not destination.exists(), str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(destination) == expected, str(destination)


def prepare(args):
    prior, out = args.prior.resolve(), args.out.resolve()
    assert prior != out and not (out / 'protocol.json').exists()
    assert not (out / 'training').exists() and not (out / 'evaluation').exists()
    spec = read(prior / 'protocol.json')
    manifest = read(prior / 'manifest.json')
    audit = read(prior / 'projection_audit.json')
    old_proposals = read(prior / 'training/proposals.json')
    assert sha(prior / 'protocol.json') == PROTOCOL_SHA
    assert spec['checkpoint_sha256'] == sha(args.checkpoint) == CHECKPOINT_SHA
    assert spec['training']['attempts'] == 1 and spec['training']['projection_sweeps'] == 4096
    assert all(v['parameter_updates'] == 0 for v in old_proposals['arms'].values())
    assert audit['inputs_unchanged'] and audit['new_candidates'] == 0
    # 原审计记录了梯度PT及原候选的SHA；不能靠文件名判断缓存归属。
    original_hashes = dict(audit['input_sha256'])
    assert all(sha(Path(path)) == digest for path, digest in original_hashes.items())
    selected = {'protocol.json': PROTOCOL_SHA}
    selected.update(spec['assets'])
    for i in range(4):
        pt, meta = 'gradients/shard%d.pt' % i, 'gradients/shard%d.json' % i
        selected[pt] = original_hashes[str(prior / pt)]
        selected[meta] = manifest[meta]
        cached = read(prior / meta)
        assert cached['protocol_sha256'] == PROTOCOL_SHA and cached['checkpoint_sha256'] == CHECKPOINT_SHA
        assert cached['frozen_unchanged'] and cached['training_updates'] == 0
    for split, count in (('replay', 48), ('holdout', 44)):
        name = 'evaluation/f68/%s/result.json' % split
        selected[name] = manifest[name]
        report = read(prior / name)
        assert report['arm'] == 'f68' and report['split'] == split
        assert report['checkpoint_sha256'] == CHECKPOINT_SHA and report['protocol_sha256'] == PROTOCOL_SHA
        assert report['frozen_unchanged'] and len(report['rows']) == count
        assert report['evaluation_optimizer_updates'] == 0
        for row in report['rows']:
            assert row['png_sha256'] == manifest[row['image']]
            selected[row['image']] = row['png_sha256']
    for name, digest in sorted(selected.items()):
        (prior / name).resolve().relative_to(prior)
        (out / name).resolve().relative_to(out)
        copy_verified(prior / name, out / name, digest)
        original_hashes[str(prior / name)] = digest
    original_hashes[str(prior / 'projection_audit.json')] = sha(prior / 'projection_audit.json')
    original_hashes[str(prior / 'result.json')] = sha(prior / 'result.json')
    provenance = dict(prior_run=str(prior), original_protocol_sha256=PROTOCOL_SHA,
        original_protocol_commit=spec['source_commit'],
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: sha(Path(__file__).with_name(name)) for name in SOURCES},
        original_input_sha256=original_hashes, copied_sha256=selected,
        cached_gradients_reused=True, new_backward_passes=0,
        cached_full_task_reverse_passes=336, cached_late_y_reverse_passes=12,
        cached_late_absolute_reverse_passes=4, reused_f68_images=92,
        planned_new_candidate_images=184, optimizer_proposals_per_arm=1,
        protocol_unchanged=True, starting_checkpoint_sha256=CHECKPOINT_SHA,
        change='only FP64 projection arithmetic and provenance diagnostics; original losses/margins/tolerances/4096/step rules unchanged',
        policy='restart from f68 with the same cached gradients; never continue old C0/C1; no new gradient generation or adaptive retries')
    write(out / 'run_provenance.json', provenance)
    print('PAIR_RETRY_PREPARED', len(selected), flush=True)


def verify(args):
    info = read(args.out / 'run_provenance.json')
    assert sha(args.out / 'protocol.json') == info['original_protocol_sha256'] == PROTOCOL_SHA
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    assert all(sha(Path(path)) == digest for path, digest in info['original_input_sha256'].items())
    assert all(sha(args.out / name) == digest for name, digest in info['copied_sha256'].items())
    assert all(sha(Path(__file__).with_name(name)) == digest for name, digest in info['source_sha256'].items())
    print('PAIR_RETRY_INPUTS_UNCHANGED', flush=True)


def rounded(args):
    """只读检查最终float32参数位移；沿用原资格规则，不新增采用门槛。"""
    from models.e2_state_texture_b import StateTextureB
    from tools.e2_b_endpoint_losses import TASKS, flatten
    from tools.e2_oracle_texture import weights_digest
    verify(args)
    spec = read(args.out / 'protocol.json')
    cfg = spec['training']
    proposals = read(args.out / 'training/proposals.json')
    old = read(args.prior / 'training/proposals.json')
    shards = [torch.load(args.out / 'gradients' / ('shard%d.pt' % i), map_location='cpu',
                         weights_only=False) for i in range(4)]
    gradients = torch.cat([s['gradients'] for s in shards]).flatten(0, 1)
    items = [item for s in shards for item in s['items']]
    rows = (gradients / gradients.norm(dim=1)[:, None]).double()
    margins = torch.full((336,), cfg['default_margin'])
    descriptions = []
    for i, item in enumerate(items):
        if [item['case'], item['condition']] == cfg['protected_condition']:
            margins[i * 7 + TASKS.index('amplitude')] = cfg['protected_amplitude_margin']
        descriptions.extend(dict(case=item['case'], condition=item['condition'], arm=item['arm'], task=t) for t in TASKS)
    b = StateTextureB(color_channels=2, reference_channels=192)
    b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start = flatten(list(b.parameters()))
    assert all(s['b_hash'] == weights_digest(b) and s['checkpoint_sha256'] == CHECKPOINT_SHA for s in shards)
    reports = {}
    for arm in ('C0', 'C1'):
        report = proposals['arms'][arm]
        for key in ('raw_norm', 'gradient_norm'):
            assert abs(report[key] - old['arms'][arm][key]) <= 1e-10, (arm, key)
        path = args.out / 'training' / (arm + '.pt')
        assert sha(path) == report['checkpoint_sha256']
        b.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['b'])
        actual = start - flatten(list(b.parameters()))
        assert weights_digest(b) == report['final_b_hash']
        assert float(actual.norm()) == report['actual_norm']
        if not report['parameter_updates']:
            reports[arm] = dict(nonzero=False, minimum_cosine=None, scaled_slack_min=None)
            continue
        scale = report['norm_scale'] * report['fixed_scale'] * report['cap_scale']
        dots = rows @ actual.double() / report['raw_norm']
        slack = dots - margins.double() * scale
        violations = [dict(descriptions[i], slack=float(slack[i]))
                      for i in range(336) if slack[i] < -1e-6]
        reports[arm] = dict(nonzero=True, actual_norm=report['actual_norm'], rule_scale=scale,
            minimum_cosine=float((rows @ actual.double() / actual.double().norm()).min()),
            scaled_slack_min=float(slack.min()), scaled_margin_violations=violations,
            tolerance=1e-6, scope='same rows and raw norm; margins scaled by unchanged norm/0.5/cap rules; diagnostic only')
    write(args.out / 'rounded_projection.json', dict(arms=reports, new_backward_passes=0,
        new_training_updates=0, new_candidates=0, policy='read-only audit; original summary acceptance rules unchanged'))
    print('PAIR_ROUNDED_PROJECTION', json.dumps(reports), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'rounded'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('/share/home/u2515283058/Mymodel/output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {'prepare': prepare, 'verify': verify, 'rounded': rounded}[args.action](args)


if __name__ == '__main__':
    main()
