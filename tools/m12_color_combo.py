"""固定CFG5输出上的Lab组合对照；不训练、不重新扩散、不修改父实验。"""
import argparse
import csv
import platform
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image
from tools import m12_color_experiment as parent
from tools.m12_color_metrics import calibrate, bootstrap

RUN = parent.PROJECT/'output_eval/m12_color_combo_dev32_20261021'
PARENT = parent.PROJECT/'output_eval/m12_color_dev32_20261021'
PARENT_COMMIT = '6f4085653c6e41bcf42426456fd8bb2b80b04f3a'
ARMS = ['CFG5', 'CFG5_LAB']
SOURCES = ['tools/m12_color_combo.py', 'submit/m12_color_combo.sh']
read, write, sha = parent.read, parent.write, parent.sha
SETTINGS = dict(identities=32, seed=42, arms=ARMS, new_diffusion_outputs=0,
    training_updates=0, reused_outputs=32, new_postprocessed_outputs=32,
    calibration=parent.SETTINGS['LAB_AB'], evaluation=parent.SETTINGS['evaluation'],
    comparisons='CFG5_LAB minus CFG5, paired identity bootstrap10000 seed32042',
    sample='all old dev32 in original order; exploratory development only',
    parameters_changed=False, mask_changed=False, reference_changed=False)


def image(arm, identity):
    return RUN/'outputs'/arm/(identity+'.png')


def full_statistics(records):
    # 首条中性参考可能为NA；保留所有预定字段，避免父实验首行推断遗漏。
    keys = list(dict.fromkeys(k for r in records for k, v in r.items()
        if isinstance(v, (float, int)) and k not in ['seed', 'mask_valid', 'inference_mask_valid']))
    base = {r['id']: r for r in records if r['arm'] == ARMS[0]}
    means = {a: {k: bootstrap([r.get(k) for r in records if r['arm'] == a]) for k in keys} for a in ARMS}
    paired = {ARMS[1]: {k: bootstrap([r[k]-base[r['id']][k] for r in records
        if r['arm'] == ARMS[1] and r.get(k) is not None and base[r['id']].get(k) is not None]) for k in keys}}
    return dict(means=means, paired=paired, baseline=ARMS[0],
        unit='identity; seed42 only, development intervals without multiplicity correction')


def guardrails(records, stats):
    delta = stats['paired'][ARMS[1]]
    base = {r['id']: r for r in records if r['arm'] == ARMS[0]}
    extreme = [r['id'] for r in records if r['arm'] == ARMS[1] and r['contour_f1']-base[r['id']]['contour_f1'] < -.15]
    checks = dict(f1=delta['contour_f1']['ci95'][0] >= -.02,
        iou=delta['iou']['ci95'][0] >= -.01, leak=delta['leak']['ci95'][1] <= .005, extreme=not extreme)
    return dict(checks=checks, all_passed=all(checks.values()), extreme_ids=extreme,
        paired_color_95_signal={k: delta[k]['n'] == 32 and delta[k]['ci95'][1] < 0
            for k in ['ref_rgbuv_h', 'ref_ab_w1']}, status='DEVELOPMENT_DIAGNOSTIC_ONLY')


def prepare():
    frozen_path = RUN/'protocol/frozen.json'
    if frozen_path.exists():
        require()
        return
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    old = read(PARENT/'protocol/frozen.json')
    assert old['commit'] == PARENT_COMMIT and old['settings'] == parent.SETTINGS
    assert read(PARENT/'metrics/complete.json')['images'] == 128
    assert all(sha(p) == v for p, v in old['code_sha256'].items())
    identities = read(PARENT/'protocol/identities.json')
    assert len(identities) == 32 and sha(PARENT/'protocol/identities.json') == old['identities_sha256']
    sources = dict(old['code_sha256'])
    sources.update({p: sha(p) for p in SOURCES})
    inputs = read(PARENT/'protocol/input_manifest.json')
    rgb_audit = read(PARENT/'audit/input_rgb_decoder_check.json')
    assert rgb_audit['all_actual_rgb_equal_old_canonical'] and len(rgb_audit['checks']) == 96
    decoded = {(r['id'],r['kind']):r for r in rgb_audit['checks']}
    for rec in inputs:
        source = PARENT/'inputs'/rec['id']/(rec['kind']+'.jpg')
        assert sha(source) == rec['sha256']
        proof = decoded[rec['id'],rec['kind']]
        assert __import__('hashlib').sha256(Image.open(source).convert('RGB').tobytes()).hexdigest() == proof['decoded_rgb_sha256']
        dest = RUN/source.relative_to(PARENT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write(RUN/'protocol/identities.json', identities)
    write(RUN/'protocol/input_manifest.json', inputs)
    write(RUN/'audit/input_decoded_rgb.json',rgb_audit)
    baseline = []
    for row in identities:
        source = PARENT/'outputs/CFG5'/(row['id']+'.png')
        meta = read(source.with_suffix('.json'))
        assert sha(source) == meta['png_sha256'] and meta['seed'] == 42
        assert meta['initial_sha256'] == meta['paired_noise_sha256'] and not meta['gt_used_in_inference']
        target = image('CFG5', row['id'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        shutil.copy2(source.with_suffix('.json'), target.with_suffix('.json'))
        baseline.append(dict(id=row['id'],png_sha256=sha(source),metadata_sha256=sha(source.with_suffix('.json'))))
    write(frozen_path, dict(commit=parent.commit(),parent_commit=PARENT_COMMIT,settings=SETTINGS,
        identities_sha256=sha(RUN/'protocol/identities.json'),input_manifest_sha256=sha(RUN/'protocol/input_manifest.json'),
        baseline=baseline,code_sha256=sources,parent_metrics_sha256=sha(PARENT/'metrics/per_image.json'),
        frozen_before_combination_outputs=True))
    print('COMBO_FROZEN', parent.commit(), flush=True)


def require():
    frozen = read(RUN/'protocol/frozen.json')
    assert frozen['settings'] == SETTINGS and frozen['commit'] == parent.commit()
    assert all(sha(p) == v for p, v in frozen['code_sha256'].items())
    assert sha(RUN/'protocol/identities.json') == frozen['identities_sha256']
    assert sha(RUN/'protocol/input_manifest.json') == frozen['input_manifest_sha256']
    assert sha(PARENT/'metrics/per_image.json') == frozen['parent_metrics_sha256']
    for rec in read(RUN/'protocol/input_manifest.json'):
        assert sha(RUN/'inputs'/rec['id']/(rec['kind']+'.jpg')) == rec['sha256']
    for rec in frozen['baseline']:
        path = image('CFG5', rec['id'])
        assert sha(path) == rec['png_sha256'] and sha(path.with_suffix('.json')) == rec['metadata_sha256']


def generate():
    from tools.e44_runtime import sketch_masks
    require()
    for row in read(RUN/'protocol/identities.json'):
        dest = image('CFG5_LAB', row['id'])
        if dest.exists():
            assert sha(dest) == read(dest.with_suffix('.json'))['png_sha256']
            continue
        started = time.perf_counter()
        sketch = Image.open(RUN/'inputs'/row['id']/'sketch.jpg').convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        _, mask, info = sketch_masks(sketch,'cpu')
        assert info == read(image('CFG5', row['id']).with_suffix('.json'))['mask']
        soft = mask[0,0].numpy()
        original = Image.open(image('CFG5',row['id'])).convert('RGB')
        ref = Image.open(RUN/'inputs'/row['id']/'reference.jpg').convert('RGB')
        output = calibrate(original,ref,soft,info['valid'])
        assert np.array_equal(output[soft == 0], np.asarray(original)[soft == 0])
        if not info['valid']:
            assert np.array_equal(output,np.asarray(original))
        dest.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(output).save(dest)
        write(dest.with_suffix('.json'),dict(id=row['id'],arm='CFG5_LAB',seed=42,mask=info,
            png_sha256=sha(dest),source_cfg5_sha256=sha(image('CFG5',row['id'])),
            softmask_rgb_float32_sha256=__import__('hashlib').sha256(soft.tobytes()).hexdigest(),
            outside_exact=True,invalid_exact_off=not info['valid'],seconds=time.perf_counter()-started,
            gt_used_in_inference=False,postprocess=True,training_updates=0,commit=parent.commit()))
    write(RUN/'generation/complete.json',dict(identities=32,arms=ARMS,new_postprocessed_outputs=32,
        reused_outputs=32,new_diffusion_outputs=0,training_updates=0))


def evaluate():
    import torch, cv2
    from eval.eval_utils import prepare_evaluation_masks
    from tools.m54_eval import regional_metrics,crop,ncc
    from tools.e33tmoc_appearance_eval import frozen_lpips
    require()
    assert read(RUN/'generation/complete.json')['new_postprocessed_outputs'] == 32
    # 保持父评价函数及其读取的mask元数据不变，仅替换输出来源。
    parent.RUN = PARENT
    records = parent.color_records(ARMS,image)
    index = {(r['arm'],r['id']): r for r in records}
    metric,info = frozen_lpips()
    metric = metric.cuda()
    tensor = lambda im: torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    def hp(im):
        a = cv2.cvtColor(np.asarray(crop(im)),cv2.COLOR_RGB2GRAY).astype(np.float32)
        return a-cv2.GaussianBlur(a,(0,0),2.)
    rows = read(RUN/'protocol/identities.json')
    wrong = read(PARENT/'metrics/complete.json')['wrong_reference']
    refs = {r['id']: hp(Image.open(RUN/'inputs'/r['id']/'reference.jpg').convert('RGB')) for r in rows}
    with torch.inference_mode():
        for row in rows:
            gtpath = RUN/'inputs'/row['id']/'gt.jpg'
            gt = Image.open(gtpath).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
            bundle = prepare_evaluation_masks((384,512),sketch_path=str(RUN/'inputs'/row['id']/'sketch.jpg'),mask_policy='sketch_only')
            for arm in ARMS:
                path = image(arm,row['id']); rgb = Image.open(path).convert('RGB'); rec = index[arm,row['id']]
                rec.update(lpips_gt=float(metric(tensor(rgb),tensor(gt)).mean()),
                    **regional_metrics(rgb,path,str(gtpath),bundle,rec['mask_valid']))
                feature = hp(crop(rgb,bundle['garment']))
                rec['hp_margin'] = ncc(feature,refs[row['id']])-ncc(feature,refs[wrong[row['id']]])
            print('COMBO_EVALUATED',row['id'],flush=True)
    stats = full_statistics(records)
    old = {r['id']: r for r in read(PARENT/'metrics/per_image.json') if r['arm'] == 'CFG5'}
    replay = {k: max(abs(r[k]-old[r['id']][k]) for r in records
        if r['arm'] == 'CFG5' and r.get(k) is not None and old[r['id']].get(k) is not None)
        for k in stats['means']['CFG5']}
    write(RUN/'metrics/per_image.json',records)
    write(RUN/'metrics/statistics.json',stats)
    write(RUN/'metrics/guardrails.json',guardrails(records,stats))
    write(RUN/'metrics/baseline_replay.json',dict(max_error=replay,
        strict_passed=all(v <= (1e-6 if k == 'lpips_gt' else 1e-12) for k,v in replay.items()),
        original_parent_values_preserved=True))
    write(RUN/'metrics/complete.json',dict(images=64,new_postprocessed_outputs=32,lpips=info,
        environment=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),pillow=__import__('PIL').__version__),
        wrong_reference=wrong,no_blind_review=True,no_pattern_ground_truth=True))


def package():
    from tools.m54_finalize import package as pack
    require(); assert read(RUN/'metrics/complete.json')['images'] == 64
    files = {str(p.relative_to(RUN)): p for p in RUN.rglob('*')
        if p.is_file() and p.suffix in ['.json','.png','.jpg'] and 'resources' not in p.relative_to(RUN).parts}
    for name in read(RUN/'protocol/frozen.json')['code_sha256']:
        files['code/'+name] = Path(name)
    pack(RUN/'m12_color_combo_review.tar.gz',files)
    print('COMBO_READY',sha(RUN/'m12_color_combo_review.tar.gz'),flush=True)


def accounting():
    from tools.m54_finalize import package as pack
    jobs = read(RUN/'resources/jobs.json')
    raw = subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    ledger = list(csv.DictReader(raw.splitlines(),delimiter='|'))
    assert len(ledger) == len(jobs) and all(j['State'] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] for j in ledger)
    write(RUN/'resources/accounting.json',ledger)
    pack(RUN/'m12_color_combo_ledger.tar.gz',{str(p.relative_to(RUN)):p for p in (RUN/'resources').glob('*') if p.is_file()})
    print('COMBO_ACCOUNTED',[(j['JobIDRaw'],j['State']) for j in ledger],flush=True)


def submit():
    path = RUN/'resources/jobs.json'; assert not path.exists()
    command = ['sbatch','--parsable','--job-name=color_combo','submit/m12_color_combo.sh']
    job = int(subprocess.check_output(command,text=True).strip().split(';')[0])
    write(path,[dict(id=job,action='run',command=command)])
    print('COMBO_SUBMITTED',job,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=['run','submit','accounting'])
    args = parser.parse_args(); (RUN/'resources').mkdir(parents=True,exist_ok=True)
    if args.action == 'run':
        prepare(); generate(); evaluate(); package()
    else:
        globals()[args.action]()
