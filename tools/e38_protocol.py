"""E38 只读表示评估合同；门槛在读取新指标前冻结。"""
import json
import hashlib
import subprocess
from pathlib import Path

OUT = Path('output_eval/e38_structure_prior_20261011')
SOURCE = Path('output_eval/e37_scrao_feasibility_20261010')
SIZE = (384, 512)
CONFIG = dict(
    experiment='E38_Prior', version=1, seed=38011, bootstrap_seed=38042,
    bootstrap_reps=10000, cohort='original E36/E37 dev32, already used in previous experiments',
    candidates=['A_SDF', 'B_FOURIER', 'C_GRAPH'], baseline='one_minus_IoU',
    weights_modified=False, caption_modified=False, dataset_modified=False,
    training_updates=0, new_generation=False, confirm96_used=False, validation500_used=False,
    rgb_extractor='original estimate_cloth_foreground_mask; RGB only, no GT/sketch guidance',
    oracle='fixed GT-derived foreground; appearance shares this mask; geometry warps mask with RGB',
    sdf='EDT(outside)-EDT(inside), image diagonal normalized; mean absolute difference',
    fourier='largest external contour, 256 arc-length samples; complex FFT coefficients -16..16; cyclic-start minimization; image coordinates/diagonal, no scale/rotation/translation removal',
    graph='experimental upper-garment contour landmarks collar/body/hem/left-right sleeve; explicit missing nodes and validated incidence; category from unchanged caption; not a learned semantic parser',
    appearance=[dict(kind='hue', level=x) for x in [0.15, 0.30]] +
               [dict(kind='texture', level=x) for x in [0.10, 0.25]] +
               [dict(kind='brightness', level=x) for x in [0.75, 1.25]],
    structure=[dict(kind=k, level=x) for k in ['sleeve', 'hem', 'contour'] for x in [0.08, 0.16]],
    perturbation_note='sleeve only for caption-declared sleeved upper garments; hem vertical lower-region warp; contour asymmetric horizontal warp; known geometry changes are synthetic, not real editing supervision',
    statistics='identity unit; AUROC gives equal identity weight; paired superiority on same valid identities; 10000 bootstrap; no optimizing sign using observed results',
    gates=dict(coverage_min=.80, structure_over_appearance_min=.80, auroc_min=.90,
               correct_direction_rho_min=.40, synthetic_iou_superiority_ci_lower=0.),
    invariance_gate='coverage + max appearance distance < min structural distance on >=80% valid identities, in oracle and RGB pipelines; each eligible structural family checked separately',
    prediction_gate='rho(delta D,-delta IoU)>.4 AND rho(delta D,-delta EdgeF1)>.4; LPIPS auxiliary only, not a structural truth label',
    superiority_gate='synthetic AUROC minus 1-IoU, paired identity bootstrap lower>0; real blinded structural ratings must corroborate superiority for a Go',
    blind_review=dict(seed=38099, scale='0 no defect,1 slight,2 clear,3 severe',
                      dimensions=['outer silhouette', 'parts/length proportions', 'internal construction lines'],
                      source='single AI visual assessor; shuffled arm labels; not a human study or independent heldout evidence',
                      policy='create review sheets/key separately; save ratings before decoding key or reading candidate statistics; if unavailable, Go withheld'),
    graph_validation='image overlays must confirm semantic landmark correctness; coverage below80% rejects C; cannot call unlabeled skeleton a garment semantic graph',
    stop='no candidate satisfying all gates => No-Go; do not add samples, seeds, thresholds or train a new method',
    budget=dict(gpus=0, cpus=2, walltime_minutes=30),
    differentiability='hard RGB threshold/contour/EDT graph pipeline not differentiable; no claim of usable end-to-end structure loss',
    novelty_claim=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
