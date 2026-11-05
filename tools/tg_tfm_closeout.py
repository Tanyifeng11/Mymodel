"""A未晋级后的诊断收尾；沿用1000步权重，不能称为28365步正式比较。"""
import argparse
import subprocess
import sys
from collections import Counter

import numpy as np
from PIL import Image

from tools.tg_tfm_experiment import ROOT, read, write, sha
from tools.bf_t3_three_arm_report import gray, paired, table

OUT = ROOT / 'closeout'
ARMS = ('T0', 'T3')


def prepare():
    if (OUT / 'protocol.json').exists():
        return
    assert read(ROOT / 'screen_complete.json')['complete']
    assert read(ROOT / 'audit/four_arm_integrity.json')['passed']
    # 先依据32件开发集及全部文本配对图锁定停止决定，再查看原128/换参考结果。
    write(ROOT / 'audit/stage_A_decision.json', dict(advance_to_B=False,
        reason='T3 has no credible additional material/decoration response over T0 in dev32',
        evidence_sha256=sha(ROOT / 'screen_metrics/paired_statistics.json'),
        ai_visual_review_identities=32, human_reviewers=0,
        observations=['T0 already responds to leather, fur, lace and denim text',
                      'T3 largely reproduces the same caption/neutral response',
                      'caption versus neutral also changes color and background; this is not proof of FiLM benefit'],
        not_run=['B', 'C', 'T6', '28365-update formal comparison'],
        limitation='single-seed 1000-update screening; absence of evidence is not proof of universal ineffectiveness'))
    rows = read(ROOT / 'evaluation.json')
    assert Counter((r['split'], r['condition']) for r in rows) == {
        ('heldout128', 'original'): 128,
        ('neutral32', 'neutral_original'): 32, ('neutral32', 'neutral_swapped'): 32}
    assert len({r['id'] for r in rows}) == 128
    write(OUT / 'evaluation.json', rows)
    dev = {r['id']: r for r in read(ROOT / 'dev_evaluation.json') if r['condition'] == 'original'}
    conflicts = []
    # 参考主色已在开发集原图中由AI核对；这里只测冲突边界，不作为文本收益指标。
    cases = [('200002806', 'blue', 'red', 'leather biker jacket'),
             ('212628739', 'yellow', 'blue', 'sleeveless dress'),
             ('130537310', 'beige', 'red', 'V-neck knitted sweater'),
             ('130108430', 'black', 'red', 'sports bra with mesh panels')]
    for identity, reference_color, conflicting_color, noun in cases:
        for condition, color in [('color_match', reference_color), ('color_conflict', conflicting_color)]:
            conflicts.append(dict(dev[identity], condition=condition, split='color_conflict4',
                caption='a photograph of a ' + color + ' ' + noun + '.',
                reference_color=reference_color, requested_color=color,
                intended_priority='sketch geometry and reference appearance take priority over conflicting text'))
    write(OUT / 'color_conflicts.json', conflicts)
    write(OUT / 'generation_manifest.json', rows + conflicts)
    write(OUT / 'protocol.json', dict(stage='diagnostic closeout after negative A screen',
        arms=list(ARMS), training_updates=1000, initialization='same complete BF-SD-T3 checkpoint',
        formal_comparison=False, original128=128, reference_swap_identities=32,
        conflict_identities=4, images_per_arm=200, inputs_sha256=sha(OUT / 'generation_manifest.json'),
        decision_sha256=sha(ROOT / 'audit/stage_A_decision.json'),
        inference=read(ROOT / 'protocol.json')['inference']))
    print('CLOSEOUT_PREPARED', flush=True)


def generate(arm):
    assert sha(OUT / 'generation_manifest.json') == read(OUT / 'protocol.json')['inputs_sha256']
    subprocess.run([sys.executable, '-m', 'tools.train_tg_tfm', 'generate', '--variant', arm,
        '--output-dir', str(OUT), '--initialization-root', str(ROOT),
        '--checkpoint', str(ROOT / 'screen' / arm / 'step01000.pt'),
        '--eval-manifest', str(OUT / 'generation_manifest.json')], check=True)


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_color_fidelity, compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import bf_t3_frozen_core as core, t3_response_run as response
    from tools.t3_response_common import patches
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    response.ROOT = OUT
    dists = response.load_dists('cuda')
    lpips, assets = frozen_lpips()
    rows = read(OUT / 'evaluation.json')
    original = {r['id']: r for r in rows if r['condition'] == 'neutral_original'}
    cache, noise, records = {}, {}, []

    def features(parts):
        values = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().cuda() / 255
        return dists.forward_once(values)

    def reference(path, is_gray):
        key = (path, is_gray)
        if key not in cache:
            image = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))
            parts = [image[y:y+64, x:x+64] for y in np.linspace(0, 448, 4).round().astype(int)
                     for x in np.linspace(0, 320, 4).round().astype(int)]
            cache[key] = features([gray(p) for p in parts] if is_gray else parts)
        return cache[key]

    def distance(a, b):
        matrix = response.distances(dists, a, b)
        return float(.5 * (matrix.min(0).values.mean() + matrix.min(1).values.mean()))

    with torch.inference_mode():
        for row in rows + read(OUT / 'color_conflicts.json'):
            masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
            coords, _ = patches(masks['garment'])
            for arm in ARMS:
                path = OUT / arm / 'outputs' / row['condition'] / (row['id'] + '.png')
                meta = read(path.with_suffix('.json'))
                assert sha(path) == meta['output_sha256'] and not meta['GT_used_in_inference']
                assert meta['checkpoint_sha256'] == read(ROOT / 'screen' / arm / 'complete.json')['checkpoint_sha256']
                assert meta['cfg_calls'] == [50, 50] and meta['caption'] == row['caption']
                assert meta['seed'] == row['seed'] and meta['cfg'] == 7. and meta['steps'] == 50
                for key in ('sketch', 'reference'):
                    assert sha(row[key]) == meta['input_' + key + '_sha256'] == row[key + '_sha256']
                if row['id'] in noise:
                    assert noise[row['id']] == meta['initial_latent_sha256']
                noise[row['id']] = meta['initial_latent_sha256']
                kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=masks)
                structure = compute_structure_preservation(**kwargs)
                record = dict(arm=arm, id=row['id'], split=row['split'], condition=row['condition'],
                    edge_f1=structure.get('edge_f1'), iou=structure.get('sketch_iou'), edge_l1=structure.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'),
                    lpips=None, tcf_lab=None, dists_margin=None, gray_margin=None, gray_dists=None, patch_n=len(coords))
                if row['condition'] == 'original':
                    record['lpips'] = float(lpips(core.tensor(path).cuda(), core.tensor(row['gt']).cuda()).mean())
                if row['condition'] != 'neutral_swapped':
                    record['tcf_lab'] = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta')
                if row['condition'] == 'neutral_swapped' and len(coords) >= 4:
                    assert row['caption'] == original[row['id']]['caption']
                    image = np.asarray(Image.open(path).convert('RGB'))
                    parts = [image[y:y+64, x:x+64] for x, y in coords]
                    for is_gray in (False, True):
                        generated = features([gray(p) for p in parts] if is_gray else parts)
                        old = distance(generated, reference(original[row['id']]['reference'], is_gray))
                        new = distance(generated, reference(row['reference'], is_gray))
                        record['gray_margin' if is_gray else 'dists_margin'] = old - new
                        if is_gray:
                            record['gray_dists'] = new
                records.append(record)
            print('CLOSEOUT_EVALUATED', row['condition'], row['id'], flush=True)
    safe = [{key: (None if isinstance(value, float) and not np.isfinite(value) else value)
             for key, value in row.items()} for row in records]
    write(OUT / 'metrics/per_image.json', safe)
    table(OUT / 'metrics/per_image.csv', safe)
    metrics = ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab', 'dists_margin', 'gray_margin', 'gray_dists']
    summaries, comparisons = [], []
    conditions = [('heldout128', 'original'), ('neutral32', 'neutral_original'), ('neutral32', 'neutral_swapped'),
                  ('color_conflict4', 'color_match'), ('color_conflict4', 'color_conflict')]
    for split, condition in conditions:
        for metric in metrics:
            grouped = {}
            for arm in ARMS:
                selected = [r for r in safe if (r['arm'], r['split'], r['condition']) == (arm, split, condition)]
                values = {r['id']: r[metric] for r in selected if r[metric] is not None}
                grouped[arm] = values
                summaries.append(dict(arm=arm, split=split, condition=condition, metric=metric,
                    n=len(values), planned_n=len(selected), mean=float(np.mean(list(values.values()))) if values else None))
            delta = [grouped['T3'][key] - grouped['T0'][key] for key in grouped['T0'].keys() & grouped['T3'].keys()]
            comparisons.append(dict(comparison='T3 minus T0', split=split, condition=condition, metric=metric, **paired(delta)))
    write(OUT / 'metrics/summary.json', summaries)
    table(OUT / 'metrics/summary.csv', summaries)
    write(OUT / 'metrics/paired_statistics.json', comparisons)
    write(OUT / 'evaluation_complete.json', dict(complete=True, outputs=len(safe),
        original128_per_arm=128, reference_swap32_per_arm=64, conflict_per_arm=8,
        training_updates=1000, formal_comparison=False, inference_hashes_cfg_no_gt_verified=True,
        lpips_assets=assets, note='diagnostic results do not reopen the development-stage gate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate', 'evaluate'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.stage == 'generate':
        generate(args.arm)
    else:
        globals()[args.stage]()
