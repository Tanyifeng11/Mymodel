"""E45 固定配方及判定；协议在新终图和指标生成前冻结。"""
import random
from datetime import datetime, timezone
from pathlib import Path
from tools.e44_protocol import OUT as OLD, E5, E5_SHA, SIZE, LAYER, read, write, sha, commit
from models.e45_pnp_anchor import FEATURE, ATTENTION

OUT = Path('output_eval/e45_b1_pnp_anchor_20261013')
CKPT = OLD / 'p1/B1/checkpoints/step0800.pt'
CKPT_SHA = 'd107dc331c5d8b15079040dc351ec737b0644f9af745834379876e3e524ab84e'
BAD = ['172600892', '193488637', '180633937', '204446309']
INVALID = ['53887902', '53328647', '170162876', '137601298']
ARMS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
KEYS = ['lpips_gt', 'iou', 'contour_f1', 'leak', 'lab', 'hp_gray', 'clip_text', 'clip_reference', 'clip_real']
CONFIG = dict(experiment='E45_B1_PnP_inspired_anchor', version=1, training_updates=0,
    source_plan_sha256='1a33aa80c328927a857fa2f9dd2b973b41b10ce7af5968e087d8bf24390c6c80',
    e5_sha256=E5_SHA, b1_sha256=CKPT_SHA, new_trainable_parameters=0,
    writer='frozen architecture-matched SD1.5; not official MimicBrush weights',
    arms=dict(C0='E5', C1='B1 global', C2='B1 interior', C3='per-step equal-L2-energy global',
        C4='conditional epsilon teacher mix', C5='downstream Feature: PRIMARY',
        C6='downstream self Q/K, student V', C7='Feature + self Q/K'),
    feature_layer=FEATURE, attention_layer=ATTENTION, b1_layer=LAYER,
    amplitude_max=.25, schedule='i0..14:0.25;i15..24:0.125*(1+cos(pi*(i-14)/10));i25..49:0',
    teacher='B1 OFF E5 at same CURRENT student latent/t/conditions; direct forward, no scheduler step',
    conditional_only=True, ddim_steps=50, cfg=7., seed=42, width=384, height=512,
    scales=dict(sketch=.6, texture=1., ipa=1.), smoke_seed=45042,
    invalid='retain all32; B1/anchors OFF; regional NA, validN28', bad_ids=BAD,
    bootstrap=dict(unit='identity', replicates=10000, seed=32042),
    gates=dict(iou_ci_lower=-.01, contour_ci_lower=-.02, catastrophic_f1_drop=-.15,
        lpips_retention_min=.5, lpips_delta_ci_upper=0., leak_ci_upper=.005, clip_text_delta_mean_min=-.005),
    equal_energy_max_relative_error=.003,
    mechanism='C5 must Pareto-dominate BOTH C3 and C4: upper Bonferroni-adjusted LPIPS bound<0 '
        'and lower IoU/F1 bounds>=-0.005/-0.01; OR lower F1 bound>0 with upper LPIPS bound<=0.002 '
        'and lower IoU bound>=-0.005. Bootstrap 10000; six comparisons; Bonferroni 99.167% two-sided '
        'bounds operationalize conservative family-wise0.05.',
    threshold_scan=dict(color_distance=[12,18,24], brightness=[225,235,245], morphology='unchanged E44'),
    selection='C5 fixed primary; other arms exploratory; no dev-tuned layer/strength/window',
    confirmation='only after full primary gates + real visual review; audit actual independent identities first; seeds42/43/44',
    human_review='anonymous pack only; no invented human ratings', historical_e44_decision='unchanged')
def protocol():
    rows = read(OLD / 'splits/dev32.json')
    ids = [r['id'] for r in rows]
    others = [x for x in ids if x not in BAD + INVALID]
    smoke = BAD + random.Random(45042).sample(others, 4)
    value = dict(config=CONFIG, dev_ids=ids, smoke_ids=smoke,
        split_sha256=sha(OLD / 'splits/dev32.json'))
    path = OUT / 'protocol.json'
    if path.exists():
        assert read(path) == value, '冻结协议不得根据结果修改'
    else:
        frozen = read(OLD / 'manifests/input_hashes.json')
        inputs = {r[k]:frozen[r[k]] for r in rows for k in ['gt','sketch','reference']}
        inputs.update({str(E5):E5_SHA, str(CKPT):CKPT_SHA})
        assert all(sha(p) == h for p,h in inputs.items()), '输入/权重与 E44 不符'
        models = read(OLD / 'manifests/model_file_hashes.json')
        assert all(sha(p) == h for p,h in models.items())
        write(OUT / 'manifests/input_hashes.json', inputs)
        write(OUT / 'manifests/model_file_hashes.json', models)
        write(OUT / 'splits/dev32.json', rows)
        write(OUT / 'splits/smoke8.json', [next(r for r in rows if r['id']==x) for x in smoke])
        write(path, value)
        write(OUT / 'manifests/pre_registration.json', dict(commit=commit(),
            protocol_sha256=sha(path), timestamp_utc=datetime.now(timezone.utc).isoformat(),
            stage='before E45 RGB/metrics; original E44 already observed'))
    return rows, value


def image_path(arm, identity, stage='s2'):
    return OUT / stage / 'images' / arm / (identity + '.png')


def historical(arm, identity):
    if arm in ['C0','C1']:
        return OLD / 'p1/images' / ('B0' if arm=='C0' else 'B1') / (identity+'.png')
    return OLD / 'b1_mask_intervention_20261012/cases' / identity / 'interior/image.png'
