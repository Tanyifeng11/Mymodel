"""仅收束G0–G2：CPU复核、正控制、固定视觉页和审计归档；不调用训练。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
from data.e32_target_pseudogt import FrozenFeatures, image_at, upsample_geometry
from data.e33gc_renderer import support
from data.e33rf_real_rotation_dataset import rotate
from tools.e33gc_protocol import *
from tools.e33tmif_metrics import pair, summarize

ARMS = ('R0', 'R90', 'R180')


def orientation_map(geometry):
    angle = np.arctan2(geometry[1], geometry[0]) / 2 % np.pi
    hsv = np.stack([angle / np.pi * 179, np.clip(geometry[3], 0, 1) * 255,
                    np.full_like(angle, 255)], -1).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)).resize((384, 512))


def run():
    prepare()
    ids = read(OUT / 'splits/diagnostic64.json')
    gate = read(OUT / 'G0_reproduction/gate.json')
    smoke = read(OUT / 'G2_autograd_smoke/smoke_audit.json')
    inputs = read(OUT / 'G0_reproduction/input_audit.json')
    cases = [read(OUT / 'G0_reproduction' / sid / 'case.json') for sid in ids]
    assert len(cases) == 64 and len(set(ids)) == 64
    png_checks = 0
    for case in cases:
        dest = OUT / 'G0_reproduction' / case['id']
        assert all(sha(dest / name) == digest for name, digest in case['files'].items())
        for arm in ARMS:
            assert np.array_equal(np.asarray(Image.open(dest / (arm + '_B0.png'))),
                                  np.asarray(Image.open(dest / (arm + '_B1.png'))))
            png_checks += 1
        assert len({r['noise_sha256'] for r in case['noop']}) == 1
        assert len({tuple(r['timesteps']) for r in case['noop']}) == 1
        assert all(len(r['timesteps']) == 50 for r in case['noop'])
    print('[GC prevalidation] all64 artifacts and 192 no-op PNGs verified', flush=True)

    # 解析目标用原E26读出作额外交叉检查；独立DFT审核仍是G1主自动证据。
    extractor = FrozenFeatures(None, 'cpu')
    controlled = []
    for row in read(OUT / 'G1_counterfactual_targets/train_audit16_rows.json'):
        dest = OUT / 'G1_counterfactual_targets/train_audit16' / row['id']
        audit = read(dest / 'audit.json')
        assert all(sha(dest / name) == digest for name, digest in audit['output_sha256'].items())
        _, inner, _ = support(image_at(dest / 'sketch.png'))
        valid = cv2.resize(inner.astype(np.float32), (48, 64), interpolation=cv2.INTER_AREA) >= .95
        geometry = {arm: upsample_geometry(extractor.extract(image_at(dest / (arm + '_target.png')),
                       perceptual=False)['geometry']).transpose(2, 0, 1) for arm in ARMS}
        controlled.append(dict(id=row['id'], **pair(geometry, valid, np.ones_like(valid, np.float32)),
            DFT_max_error=max(audit['FFT_analytic_errors']),
            coverage=audit['target_pattern_coverage']))
    write(OUT / 'G1_counterfactual_targets/positive_control_crosscheck.json', dict(
        records=controlled, r90_success=sum(r['r90_success'] for r in controlled),
        r180_success=sum(r['r180_success'] for r in controlled), n=16,
        method='original E26 local-pattern readout on independent rasterizer RGB; automatic only',
        DFT_primary=True, independent_human_review=None))
    historical = IF / 'carrier_purity/summary.json'
    if historical.exists():
        write(OUT / 'G1_counterfactual_targets/historical_P3_control.json', dict(
            source=str(historical), sha256=sha(historical),
            P3=read(historical)['groups']['P3'],
            note='historical analytic carrier, different sampling start; no module gain attribution'))

    lookup = {c['id']: c['B0'] for c in cases}
    auto_ids = inputs['automatic_eligible_ids']
    write(OUT / 'G0_reproduction/input_group_summaries.json', dict(
        all_fixed64=summarize(list(lookup.values())),
        automatic_input_eligible=summarize([lookup[sid] for sid in auto_ids]),
        groups={k: summarize([lookup[sid] for sid in members]) for k, members in inputs['groups'].items()},
        note='input-only automatic subgroups frozen before generated outputs; human eligibility unverified; groups may overlap'))

    # 固定hash16及输入预选组的所有身份均做视觉页，不按输出选择病例。
    cohort = {r['id']: r for r in read(TM / 'manifests/cohorts.json')['primary']}
    visual_ids = list(dict.fromkeys(inputs['fixed16'] + [sid for group in inputs['groups'].values() for sid in group]))
    folder = OUT / 'visual_audit/G0_baseline'
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for sid in visual_ids:
        row = cohort[sid]; dest = OUT / 'G0_reproduction' / sid
        ref = image_at(DATASET / row['reference']); sketch = image_at(DATASET / row['sketch'])
        with np.load(IF / 'reproduction/seed42/fields' / (sid + '.npz')) as z:
            # 统一为E26显示布局cos2θ/sin2θ/logfreq/conf；RF2没有频率通道。
            rf = np.concatenate([z['orientation'][0], np.zeros_like(z['confidence'][0]), z['confidence'][0]], 0)
        top = [('Sketch', sketch), ('Ref R0', ref), ('Ref R90', rotate(ref, 90)),
               ('Ref R180', rotate(ref, 180)), ('GT real R0', image_at(DATASET / row['target']))]
        top += [(arm + ' E5', image_at(dest / (arm + '_B0.png'))) for arm in ARMS]
        top += [('RF2 R0 hue/conf', orientation_map(rf))]
        bottom = []
        for arm in ARMS:
            with np.load(dest / (arm + '_orientation.npz')) as z:
                bottom.append((arm + ' E26 hue/conf', orientation_map(z['geometry'])))
        for arm in ARMS:
            with np.load(dest / (arm + '_orientation.npz')) as z:
                bottom.append((arm + ' valid support', Image.fromarray(z['support'].astype(np.uint8) * 255).convert('RGB')))
        for name in ('wrong', 'near'):
            path = dest / ('R0_B2_' + name + '.png')
            if path.exists(): bottom.append(('B2 ' + name + ' R0', image_at(path)))
        panel = Image.new('RGB', (1260, 450), 'white'); draw = ImageDraw.Draw(panel)
        draw.text((4, 2), sid + ' | ' + row['caption'][:180], fill='black')
        for line, images in enumerate((top, bottom)):
            for column, (label, image) in enumerate(images):
                draw.text((140 * column + 3, 25 + 194 * line), label, fill='black')
                image = image.copy(); image.thumbnail((134, 168))
                panel.paste(image, (140 * column + 3, 42 + 194 * line))
        b = lookup[sid]
        draw.text((4, 416), 'R90={} R180={} | Texture={:.5f} Contour={:.5f} Sketch={:.5f} Text={:.5f}'.format(
            b['r90_success'], b['r180_success'], b['texture_score'], b['contour_f1'], b['sketch_similarity'], b['text_score']), fill='black')
        draw.text((4, 432), 'B1 exact B0 | A1 sham/A2 trained candidate not run | support: original GT evaluation support + generated confidence', fill='black')
        path = folder / (sid + '.png'); panel.save(path); index[sid] = str(path.relative_to(OUT))
    write(OUT / 'visual_audit/G0_baseline/index.json', dict(fixed16=inputs['fixed16'], groups=inputs['groups'], files=index))

    frozen_check()
    decision(execution_scope='G0-G2 only; user stopped before G3/G4',
        formal_training_started=False, G3_G4_authorized=False,
        next_route='prevalidation_only_closed_no_G3_G4',
        unresolved=['independent human target/input review', 'controlled dev128 overlaps real heldout126/128'],
        feasibility_pass=None)
    write(OUT / 'completion_check.json', dict(
        execution_complete=True, execution_scope='G0-G2 computational prevalidation only',
        scientific_all_gates_pass=None, G0_reproduction_pass=gate['reproduction_pass'],
        G0_noop_png_checks=png_checks, G0_case_count=64,
        G1_automatic_pass=read(OUT / 'G1_counterfactual_targets/automatic_audit.json')['automatic_pass'],
        G1_independent_human_gate=None, G2_pass=smoke['G2_pass'],
        real_automatic_eligible_n=len(auto_ids), real_fully_audited_eligible_n=None,
        formal_training_updates=0, smoke_optimizer_steps=2,
        G3_G4='not run per user request', confirmation_generation_not_run=True,
        frozen_check=read(OUT / 'frozen_check.json')['pass'], code_commit=commit()))
    write(OUT / 'artifact_manifest.json', dict(files={str(p.relative_to(OUT)): sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ('.json', '.png', '.npz', '.pt')
        and p.name != 'artifact_manifest.json'}))
    bundle('prevalidation_G0_G2')
    print('[GC prevalidation] closed; formal training updates=0', flush=True)


if __name__ == '__main__':
    run()
