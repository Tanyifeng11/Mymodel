"""只读重放既有投影：区分迭代上限、Gram近似及重建/点积舍入；不产生新候选。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_b_balanced_losses as balanced
from tools.e2_b_endpoint_losses import TASKS, flatten
from tools.e2_oracle_texture import weights_digest
from tools.e2_state_b_pair_finetune import LEGACY_FOCUS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def same_number(actual, expected, label):
    assert abs(actual - expected) <= 1e-10, (label, actual, expected)


def replay_projection(raw, gradients, margins, descriptions, sweeps, tolerance=1e-6):
    """保持原float32运算和4096上限；双精度仅验算同一个既有dual/delta。"""
    norms = gradients.norm(dim=1)
    active = norms > 1e-12
    assert bool(active.any()) and float(raw.norm()) > 0
    rows = gradients[active] / norms[active, None]
    length = raw.norm()
    required = margins[active]
    gram = (rows @ rows.T).double().cpu().numpy()
    offset = (rows @ (raw / length)).double().cpu().numpy() - required.double().cpu().numpy()
    dual = np.zeros(len(rows))
    internal_break = False
    for sweep in range(sweeps):
        for i in range(len(rows)):
            dual[i] = max(0., dual[i] - (offset[i] + gram[i] @ dual) / max(gram[i, i], 1e-12))
        if (offset + gram @ dual).min() >= -tolerance / 4:
            internal_break = True
            break
    gram_slack = torch.from_numpy(offset + gram @ dual)
    delta = raw + length * (rows.T @ torch.as_tensor(dual, dtype=rows.dtype, device=rows.device))
    dots = (rows @ delta) / length
    slack32 = dots - required
    feasible = bool((slack32 >= -tolerance).all())

    # 检查同一float32 delta，既不更换dual，也不重跑双精度优化器。
    rows64, required64 = rows.double(), required.double()
    slack64 = (rows64 @ delta.double()) / length.double() - required64
    # 同一dual的实数公式验算：用于区分原Gram/offset误差与delta重建舍入。
    gram64 = rows64 @ rows64.T
    offset64 = rows64 @ (raw.double() / length.double()) - required64
    ideal64 = offset64 + gram64 @ torch.from_numpy(dual)
    violations = []
    active_indices = active.nonzero().flatten().tolist()
    for i, original_index in enumerate(active_indices):
        if min(float(gram_slack[i]), float(slack32[i]), float(slack64[i]), float(ideal64[i])) >= -tolerance:
            continue
        violations.append(dict(descriptions[original_index], constraint_index=original_index,
            required_margin=float(required[i]), gradient_norm=float(norms[original_index]),
            dual=float(dual[i]), gram_slack=float(gram_slack[i]),
            float32_rebuilt_slack=float(slack32[i]), same_delta_float64_slack=float(slack64[i]),
            same_dual_ideal_float64_slack=float(ideal64[i]),
            violates_float32=bool(slack32[i] < -tolerance),
            violates_same_delta_float64=bool(slack64[i] < -tolerance)))
    minimum = lambda x: float(x.min())
    return dict(iterations=sweep + 1, maximum_iterations=sweeps, internal_break=internal_break,
        hit_iteration_cap=not internal_break, tolerance=tolerance, internal_tolerance=tolerance / 4,
        constraints=len(gradients), active_constraints=len(rows),
        reproduced_feasible=feasible, minimum_dot=float(dots.min()),
        relative_correction=float((delta - raw).norm() / length),
        gram_slack_min=minimum(gram_slack), float32_rebuilt_slack_min=minimum(slack32),
        same_delta_float64_slack_min=minimum(slack64), same_dual_ideal_float64_slack_min=minimum(ideal64),
        gram_violations=int((gram_slack < -tolerance).sum()),
        float32_violations=int((slack32 < -tolerance).sum()),
        same_delta_float64_violations=int((slack64 < -tolerance).sum()),
        same_dual_ideal_float64_violations=int((ideal64 < -tolerance).sum()),
        maximum_gram_and_offset_error=float((gram_slack - ideal64).abs().max()),
        maximum_delta_reconstruction_error=float((slack64 - ideal64).abs().max()),
        maximum_dot_accumulation_error=float((slack32.double() - slack64).abs().max()),
        maximum_gram_entry_error=float((torch.from_numpy(gram) - gram64).abs().max()),
        dual_max=float(dual.max()), dual_l1=float(np.abs(dual).sum()),
        violations=violations,
        interpretation='feasible=false仅表示本次有限迭代/有限精度验算未通过，不是数学不可行证明；本审计不产生可采用候选')


def audit(args):
    spec = read(args.out / 'protocol.json')
    proposals = read(args.out / 'training/proposals.json')
    cfg = spec['training']
    assert cfg['attempts'] == 1 and cfg['projection_sweeps'] == 4096
    inputs = [args.out / 'protocol.json', args.out / 'training/proposals.json', args.checkpoint]
    inputs += [args.out / 'gradients' / ('shard' + str(i) + '.pt') for i in range(4)]
    inputs += [args.out / 'training' / (arm + '.pt') for arm in ('C0', 'C1')]
    hashes = {str(path): sha(path) for path in inputs}
    assert hashes[str(args.checkpoint)] == spec['checkpoint_sha256']
    shards = [torch.load(path, map_location='cpu', weights_only=False) for path in inputs[3:7]]
    assert all(s['protocol_sha256'] == hashes[str(args.out / 'protocol.json')] for s in shards)
    assert len({s['b_hash'] for s in shards}) == len({s['net_hash'] for s in shards}) == 1
    gradients = torch.cat([s['gradients'] for s in shards])
    items = [r for s in shards for r in s['items']]
    assert gradients.shape[:2] == (48, 7) and gradients.dtype == torch.float32
    weights = torch.tensor(spec['replay_weights'])
    merged = (gradients.mean(0) * weights[:, None]).sum(0)
    old_abs = torch.stack([s['absolute_gradient'] for s in shards if s['case'] in LEGACY_FOCUS]).mean(0)
    new_abs = torch.stack([s['absolute_gradient'] for s in shards if s['case'] not in LEGACY_FOCUS]).mean(0)
    pair = torch.stack([s['pair_gradient'] for s in shards]).mean(0)
    normalized = lambda x: x * (merged.norm() / x.norm().clamp_min(1e-12))
    directions = dict(C0=merged + normalized(old_abs),
        C1=merged + normalized(old_abs) + cfg['pair_gradient_norm_fraction'] * normalized(pair)
        + cfg['extra_absolute_gradient_norm_fraction'] * normalized(new_abs))
    constraints = gradients.flatten(0, 1)
    margins = torch.full((len(constraints),), cfg['default_margin'])
    descriptions = []
    for i, row in enumerate(items):
        if [row['case'], row['condition']] == cfg['protected_condition']:
            margins[i * 7 + TASKS.index('amplitude')] = cfg['protected_amplitude_margin']
        for task in TASKS:
            descriptions.append(dict(case=row['case'], condition=row['condition'], arm=row['arm'], task=task))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b']
    reports = {}
    for arm, direction in directions.items():
        prior = proposals['arms'][arm]
        assert prior['parameter_updates'] == 0 and not prior['projection']['feasible']
        assert hashes[str(args.out / 'training' / (arm + '.pt'))] == prior['checkpoint_sha256']
        b = StateTextureB(color_channels=2, reference_channels=192)
        b.load_state_dict(saved)
        initial_hash = weights_digest(b)
        assert initial_hash == shards[0]['b_hash'] == prior['initial_b_hash']
        parameters = list(b.parameters())
        start = flatten(parameters)
        optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        balanced.assign(parameters, direction)
        torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
        # 只复现已经发生的原始proposal；模型仅在内存中，不落盘、不采样。
        optimizer.step()
        raw = start - flatten(parameters)
        same_number(float(raw.norm()), prior['raw_norm'], arm + ':raw_norm')
        same_number(float(direction.norm()), prior['gradient_norm'], arm + ':gradient_norm')
        result = replay_projection(raw, constraints, margins, descriptions, cfg['projection_sweeps'])
        assert result['reproduced_feasible'] == prior['projection']['feasible']
        for key in ('minimum_dot', 'relative_correction'):
            same_number(result[key], prior['projection'][key], arm + ':' + key)
        result.update(initial_b_hash=initial_hash, raw_norm=float(raw.norm()),
                      gradient_norm=float(direction.norm()), original_proposal_reproduced=True)
        reports[arm] = result
        print('PROJECTION_AUDIT_ARM', arm, json.dumps({k: v for k, v in result.items() if k not in ('violations', 'interpretation')}), flush=True)
        del optimizer, b, parameters, start, raw
    assert all(sha(Path(path)) == value for path, value in hashes.items())
    files = [Path(__file__), Path(__file__).with_name('e2_b_pair_finetune_losses.py'),
             Path(__file__).with_name('e2_state_b_pair_finetune.py')]
    result = dict(arms=reports, input_sha256=hashes, inputs_unchanged=True,
        source_sha256={str(path): sha(path) for path in files},
        audit_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        original_protocol_commit=spec['source_commit'], torch_version=torch.__version__,
        numpy_version=np.__version__, torch_threads=torch.get_num_threads(),
        new_training_updates=0, new_candidates=0, weights_written=False,
        policy='仅诊断既有4096-sweep拒绝，保持原loss/margin/tolerance/步长；不续训、不改候选、不改采用结论')
    destination = args.out / 'projection_audit.json'
    assert not destination.exists(), '不覆盖既有审计'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('PROJECTION_AUDIT_COMPLETE', sha(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('/share/home/u2515283058/Mymodel/output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args)


if __name__ == '__main__':
    main()
