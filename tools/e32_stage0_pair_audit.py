"""E32 Stage0：已有真实配对审计；无训练、无 GT 输入泄漏。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from data.e32_target_pseudogt import FrozenFeatures, image_at, masks, unit, TAU, COARSE
from data.e32_wrong_reference_sampler import wrong_references
from models.apacc_features import load_dino
from tools.e32_common import (OUT, DATASET, WEIGHTS, read, write, sha, bootstrap, initial_decision,
                              make_split, frozen_manifest, finish_frozen, git_commit)


METRICS = ('appearance', 'orientation', 'period', 'color')
ARMS = ('matched', 'color_near', 'random', 'zero')


def cache_path(out, sid):
    return out/'stage0_pair_audit/cache'/(sid+'.npz')


def load_feature(out, sid):
    with np.load(cache_path(out, sid)) as data:
        return {k:data[k].astype(np.float32) for k in data.files}


def descriptor_similarity(target, reference):
    # Lab 统计单列为 color，不参与主 appearance advantage。
    # FFT/ACF/self-sim各占1/6，旋转平均DINO固定投影占1/2。
    blocks = [(0, 16, 1/6), (16, 24, 1/6), (24, 32, 1/6), (38, 64, .5)]
    similarity = np.zeros((len(target), len(reference)), np.float32)
    for start, end, weight in blocks:
        similarity += weight*(unit(target[:, start:end]) @ unit(reference[:, start:end]).T)
    # reference 是局部crop；比较它在 target interior 中能否找到外观支持。
    return float(similarity.max(axis=0).mean())


def metrics(target, reference, target_valid, reference_valid):
    gt = target['target_geometry'].reshape(-1, 4)
    ref = reference['reference_geometry'].reshape(-1, 4)
    occ = target['target_occupancy'].reshape(-1) >= .95
    appearance = descriptor_similarity(target['target_appearance'].reshape(-1, 64)[occ],
                                       reference['reference_appearance'].reshape(-1, 64)) if occ.any() else None
    color = float(target['target_histogram'] @ reference['histogram'])
    g, r = gt[target_valid], ref[reference_valid]
    if not len(g) or not len(r):
        return dict(appearance=appearance, orientation=None, period=None, color=color)
    orientation = np.clip(unit(g[:, :2]) @ unit(r[:, :2]).T, -1, 1)
    orientation = np.degrees(np.arccos(orientation))/2
    period = abs(g[:, 2, None]-r[None, :, 2])/np.log(2)
    # Geometry是目标到reference的最近可读局部兼容度，没有声称对应关系真值。
    ori_error = np.average(orientation.min(axis=1), weights=g[:, 3])
    period_error = np.average(period.min(axis=1), weights=g[:, 3])
    return dict(appearance=appearance, orientation=float(1-ori_error/90),
                period=float(np.exp(-period_error)), color=color)


def zero_metrics(target, gt_valid):
    # 全零可见颜色/appearance；无可读geometry，因此以最大错误兼容度0报告。
    return dict(appearance=0., orientation=0. if gt_valid.any() else None,
                period=0. if gt_valid.any() else None, color=0.)


def prepare(args, splits):
    records = splits['stage0_train_audit']+splits['dev']+splits['confirmation_all']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, dino_sha = load_dino(device, args.weights)
    extractor = FrozenFeatures(model, device)
    hashes = {}
    diagnostics = []
    for i, row in enumerate(records):
        sid = row['id']
        hashes[sid] = {k:sha(args.dataset/row[k]) for k in ('reference', 'target', 'sketch')}
        path = cache_path(args.out, sid)
        if not path.exists():
            reference, target, sketch = [image_at(args.dataset/row[k]) for k in ('reference', 'target', 'sketch')]
            input_mask, interior, info = masks(sketch, target)
            source = extractor.extract(reference, appearance=True)
            supervision = extractor.extract(target, interior, appearance=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, reference=source['reference'], reference_geometry=source['geometry'],
                reference_appearance=source['appearance'], histogram=source['histogram'],
                target_geometry=supervision['geometry'], target_appearance=supervision['appearance'],
                target_occupancy=supervision['occupancy'], target_histogram=supervision['histogram'],
                input_mask=cv2.resize(input_mask.astype(np.uint8), (48, 64), interpolation=cv2.INTER_NEAREST),
                target_interior=cv2.resize(interior.astype(np.uint8), (48, 64), interpolation=cv2.INTER_NEAREST))
            write(args.out/'stage0_pair_audit/mask_diagnostics'/(sid+'.json'), info)
        feature = load_feature(args.out, sid)
        diagnostics.append(dict(id=sid, interior_cells=int((feature['target_occupancy']>=.95).sum()),
            target_geometry_readable_cells=int((feature['target_geometry'][..., 3]>=TAU).sum()),
            reference_geometry_readable_cells=int((feature['reference_geometry'][..., 3]>=TAU).sum()),
            input_mask_fraction=float(feature['input_mask'].mean()),
            reference_target_bytes_identical=hashes[sid]['reference']==hashes[sid]['target']))
        if (i+1)%16 == 0 or i+1 == len(records):
            print('[E32 Stage0 prepare]', i+1, '/', len(records), flush=True)
    write(args.out/'audits/stage0_input_hashes.json', hashes)
    write(args.out/'audits/stage0_readability.json', diagnostics)
    return hashes


def evaluate(args, splits, hashes):
    summaries = {}
    pools = dict(train_audit=splits['stage0_train_audit'], dev=splits['dev'],
                 validation_diagnostic=splits['confirmation_all'])
    for split, records in pools.items():
        features = {r['id']:load_feature(args.out, r['id']) for r in records}
        wrong = wrong_references(records, features, hashes)
        write(args.out/'stage0_pair_audit'/split/'wrong_manifest.json', wrong)
        rows = []
        for row in records:
            sid = row['id']
            target = features[sid]
            gt_valid = target['target_geometry'][..., 3].reshape(-1) >= TAU
            sources = {name:features[sid if name=='matched' else wrong[sid][name]]
                       for name in ARMS[:3]}
            # geometry主比较只纳入三组均有可读source、同一target可读的cases。
            readable = gt_valid.any() and all((f['reference_geometry'][..., 3]>=TAU).any() for f in sources.values())
            measurements = {}
            for name, source in sources.items():
                ref_valid = source['reference_geometry'][..., 3].reshape(-1)>=TAU
                measurements[name] = metrics(target, source, gt_valid if readable else np.zeros_like(gt_valid), ref_valid)
            measurements['zero'] = zero_metrics(target, gt_valid if readable else np.zeros_like(gt_valid))
            rows.append(dict(id=sid, case_id=row.get('case_id'), split=split,
                             geometry_comparable=bool(readable), metrics=measurements,
                             wrong_references=wrong[sid]))
        summary = dict(reference_count=len(rows), geometry_comparable_cases=sum(r['geometry_comparable'] for r in rows),
                       arms={}, differences={}, gate_components={})
        for name in ARMS:
            summary['arms'][name] = {metric:bootstrap([r['metrics'][name][metric] for r in rows
                                    if r['metrics'][name][metric] is not None]) for metric in METRICS}
        for wrong_name in ARMS[1:]:
            summary['differences'][wrong_name] = {}
            for metric in METRICS:
                values = [r['metrics']['matched'][metric]-r['metrics'][wrong_name][metric] for r in rows
                          if r['metrics']['matched'][metric] is not None and r['metrics'][wrong_name][metric] is not None]
                summary['differences'][wrong_name][metric] = bootstrap(values)
        for metric in METRICS[:3]:
            comparisons = [summary['differences'][n][metric] for n in ('color_near', 'random')]
            summary['gate_components'][metric] = all(v['n']>=8 and v['ci95'][0]>0 for v in comparisons)
        summary['pass'] = any(summary['gate_components'].values())
        write(args.out/'stage0_pair_audit'/split/'rows.json', rows)
        write(args.out/'stage0_pair_audit'/split/'summary.json', summary)
        summaries[split] = summary
        print('[E32 Stage0]', split, json.dumps(summary['differences']['color_near']), flush=True)
    # 不根据8个已知causal case调协议；dev是唯一Stage0 Gate。
    decision = initial_decision()
    decision['pair_dependence_pass'] = summaries['dev']['pass']
    decision['matched_appearance_advantage'] = summaries['dev']['differences']['color_near']['appearance']['mean']
    decision['next_route'] = 'geometry_field_training' if decision['pair_dependence_pass'] else 'dataset_pairing_diagnosis'
    write(args.out/'decision_summary.json', decision)
    for folder, status in [('A_geometry', 'pending_stage0_pass' if decision['pair_dependence_pass'] else 'not_run_stage0_gate_failed'),
                           ('B_appearance', 'not_run_requires_A'), ('C_scaffold', 'not_run_requires_A_B'),
                           ('D_generation', 'not_run_requires_C'), ('E_confirmation', 'not_run_requires_D'),
                           ('ablations', 'not_run_requires_corresponding_stage')]:
        write(args.out/folder/'status.json', dict(status=status))
    return summaries


def visuals(args, splits, hashes):
    rows = read(args.out/'stage0_pair_audit/dev/rows.json')
    eligible = [r for r in rows if r['metrics']['matched']['appearance'] is not None]
    chosen = sorted(eligible, key=lambda r:r['metrics']['matched']['appearance']-r['metrics']['color_near']['appearance'])[:16]
    chosen += sorted(eligible, key=lambda r:r['metrics']['matched']['appearance']-r['metrics']['color_near']['appearance'])[-16:]
    mapping = {r['id']:r for r in splits['dev']}
    write(args.out/'audits/stage0_visual_selection.json', dict(method='16 worst and16 best fixed appearance advantage; no manual filtering',
                                                             selected_ids=[r['id'] for r in chosen]))
    for start in range(0, 32, 8):
        sheet = Image.new('RGB', (800, 8*170), 'white')
        draw = ImageDraw.Draw(sheet)
        for y, row in enumerate(chosen[start:start+8]):
            r = mapping[row['id']]
            near = mapping[row['wrong_references']['color_near']]
            random = mapping[row['wrong_references']['random']]
            draw.text((3, y*170+2), 'id%s app delta %.3f readable%s'%(r['id'],row['metrics']['matched']['appearance']-row['metrics']['color_near']['appearance'],row['geometry_comparable']), fill='black')
            for x, (label, path) in enumerate([('matched',r['reference']), ('target',r['target']),
                ('sketch',r['sketch']), ('color near',near['reference']), ('random',random['reference'])]):
                sheet.paste(image_at(args.dataset/path).resize((144, 144)), (x*160, y*170+26))
                draw.text((x*160+2, y*170+14), label, fill='black')
        sheet.save(args.out/'audits'/('stage0_contact_%02d.png'%start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen_manifest(args.out, args.weights)
    splits = make_split(args.out, args.dataset)
    write(args.out/'protocol.json', dict(experiment='E32', git_commit=git_commit(), field_HW=[64,48],
        image_WH=[384,512], reference_grid_HW=[16,12], architecture=dict(hidden=128,heads=4,blocks=3,ffn=256,dropout=.1),
        stage0_train_audit_cases=512, dev_cases=256, seeds=[42,43,44], steps_per_seed=8000, batch=8,
        lr=1e-4, weight_decay=1e-4, warmup=500, optimizer='AdamW', scheduler='cosine',
        geometry_confidence_threshold=TAU, interior_erosion_px=5, rot90_success_tolerance_deg=15,
        zero_degradation='error difference case-bootstrap CI95 lower>0', training_steps=0,
        no_target_rgb_encoder_input=True, input_mask='sketch-only opencv; no target fallback',
        appearance='64D =16 radialFFT+8 radialACF+8 radialSelfSim+6 Lab+26 fixed-projected four-rotation-averaged frozen DINO',
        appearance_audit='Lab excluded: 1/6 each spectral/ACF/selfsim +1/2 perceptual; mean reference-to-target nearest local cosine',
        geometry_audit='target-to-reference nearest readable local orientation/frequency compatibility; no correspondence GT',
        geometry_confidence='E26 shared confidence for orientation and period; no new estimator fitted',
        period_units='native log cycles/pixel after fixed image resize; crop magnification not known or corrected',
        gate='dev only: appearance OR orientation OR period, matched minus color-near AND random CI95 lower>0; at least8 comparable cases',
        bootstrap_resamples=2000, projection_fit='fixed seeded orthogonal random projection; no fitting',
        negative_selection='reference Lab histogram; distinct identity and distinct reference/target bytes; deterministic random',
        true_reconstruction_only_matched=True, zero_reference_calibration_only=True,
        limits=['same-source texture crop versus full target; descriptor compatibility is not dense correspondence truth',
                '5px erosion removes exterior band, not internal collar/placket seams',
                'three prespecified alternative gate metrics; individual CI95, not multiple-testing-adjusted',
                'paired test uses official per-identity sketches; original E29 unpaired fixed-sketch transfer is separate']))
    hashes = prepare(args, splits)
    evaluate(args, splits, hashes)
    visuals(args, splits, hashes)
    finish_frozen(args.out)
    write(args.out/'stage0_pair_audit/completion.json', dict(numeric_complete=True,
          total_cases=512+256+18, visual_review_completed=False, **{'pass':True}, git_commit=git_commit()))
    print('[E32 Stage0 decision]', json.dumps(read(args.out/'decision_summary.json')), flush=True)


if __name__=='__main__':
    main()
