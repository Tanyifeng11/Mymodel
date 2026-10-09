"""补齐统一外观及Sketch recall；只读已完成图像，不改模型、样本或Gate。"""
import tarfile
import cv2
import numpy as np
import torch
from PIL import Image
from data.e32_target_pseudogt import image_at, masks
from data.e33rf_real_rotation_dataset import rotate
from garment_mask_utils import estimate_cloth_foreground_mask
from tools.e33tmoc_appearance_eval import patches, measure, frozen_lpips, model_hash, APPEARANCE_PROTOCOL
from tools.e33tmif_metrics import bootstrap
from tools.e33tmms_protocol import OUT, IF, TM, DATASET, read, write, sha, commit, frozen_check

KEYS = ['Lab_color_distance', 'mean_RGB_distance', 'color_histogram_similarity', 'patch_lpips', 'texture_score']
DEST = OUT/'supplement'

def aggregate(records):
    return {key: bootstrap([r[key] for r in records]) for key in records[0] if key != 'id'}

def average(sid, arms, keys=KEYS):
    return dict(id=sid, **{key: float(np.mean(values)) if values else None for key in keys
        for values in [[a[key] for a in arms if a[key] is not None]]})

def run():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    protected = {str(p): sha(p) for p in [OUT/'decision_summary.json', OUT/'completion_check.json',
        OUT/'artifact_manifest.json', OUT/'M3/diagnostic64/gate.json', OUT/'M3/diagnostic64/summary.json']}
    manifest = read(OUT/'artifact_manifest.json')['files']
    old_manifest = read(IF/'artifact_manifest.json')['files']
    def verify(path):
        digest = sha(path)
        if IF in path.parents:
            assert digest == old_manifest[str(path.relative_to(IF))]
        elif OUT in path.parents:
            assert digest == manifest[str(path.relative_to(OUT))]
        return digest
    metric, init = frozen_lpips(); metric_sha = model_hash(metric)
    assert metric_sha == read(OUT/'protocol/appearance_metrics.json')['lpips_state_sha256']
    ids = read(OUT/'splits/diagnostic64.json')
    cohort = [r for r in read(TM/'manifests/cohorts.json')['primary'] if r['id'] in ids]
    assert len(cohort) == 64
    collected = {name: [] for name in ['M3', 'oldFull', 'matched_E5']}
    inputs = {}
    for row in cohort:
        sid = row['id']; ref_path = DATASET/row['reference']; sketch_path = DATASET/row['sketch']
        inputs.update({str(p):sha(p) for p in [ref_path,sketch_path]})
        ref = image_at(ref_path); inner = cv2.erode(masks(image_at(sketch_path))[0].astype(np.uint8), np.ones((17,17),np.uint8)) > 0
        tb = patches(ref,inner,sid,'shared','target')
        refs = [ref, rotate(ref,90), rotate(ref,180)]
        arms = {name:[] for name in collected}
        for arm, reference in zip(['R0','R90','R180'],refs):
            sm = np.asarray(estimate_cloth_foreground_mask(reference,*reference.size)[0]) > 127
            sb = patches(reference,sm,sid,arm,'source')
            paths = dict(M3=OUT/'M3/diagnostic64/seed42'/sid/(arm+'.png'),
                oldFull=IF/'stage_survival/seed42'/sid/'S9'/(arm+'.png'),
                matched_E5=TM/'baseline_e5'/sid/'d42'/(arm+'.png'))
            for name,path in paths.items():
                digest = verify(path)
                if name == 'matched_E5': assert digest == read(path.with_suffix('.json'))['output_sha256']
                inputs[str(path)] = digest
                value = measure(Image.open(path).convert('RGB'),reference,inner,sm,tb,sb,metric)
                value.update(id=sid,arm=arm,image_path=str(path),image_sha256=digest,source_boxes=sb,target_boxes=tb)
                write(DEST/name/sid/(arm+'.json'),value); arms[name].append(value)
            
        for name in collected: collected[name].append(average(sid,arms[name]))
        print('[MS supplement]',sid,flush=True)
    summary = {name:dict(case_count=64,statistics=aggregate(records)) for name,records in collected.items()}
    for name,records in collected.items():write(DEST/name/'cases.json',records)
    # M2逐臂指标已经由训练结束时的冻结LPIPS算出；不重跑不同队列。
    m2 = []
    for case in sorted((OUT/'M2/controlled/dev128').glob('*/case.json')):
        sid = case.parent.name; records=[]
        for arm in ['R0','R90','R180']:
            path=case.parent/(arm+'_appearance.json'); verify(path); records.append(read(path))
        m2.append(average(sid,records))
    assert len(m2)==128
    write(DEST/'M2_controlled/cases.json',m2)
    summary['M2_controlled']=dict(case_count=128,statistics=aggregate(m2),
        support='original controlled M_cf support, different cohort from real diagnostic64')
    for name,filename in [('M3','rows.json'),('oldFull','oldFull_rows.json'),('matched_E5','matched_control_rows.json')]:
        rows=read(OUT/'M3/diagnostic64'/filename)
        sketch=[average(r['id'],list(r['arms'].values()),['sketch_similarity']) for r in rows]
        write(DEST/name/'sketch_cases.json',sketch)
        summary[name]['statistics']['sketch_similarity']=aggregate(sketch)['sketch_similarity']
    summary['paired_vs_oldFull'] = {k:bootstrap([r[k]-b[k] for r,b in zip(collected['M3'],collected['oldFull'])
        if r[k] is not None and b[k] is not None]) for k in KEYS}
    summary['paired_vs_matched_E5'] = {k:bootstrap([r[k]-b[k] for r,b in zip(collected['M3'],collected['matched_E5'])
        if r[k] is not None and b[k] is not None]) for k in KEYS}
    assert model_hash(metric)==metric_sha and all(sha(p)==h for p,h in inputs.items())
    assert all(sha(p)==h for p,h in protected.items()); frozen_check()
    write(DEST/'summary.json',summary)
    write(DEST/'protocol.json',dict(appearance=APPEARANCE_PROTOCOL,**init,lpips_state_sha256=metric_sha,
        git_commit=commit(),base_protected_sha256=protected,input_sha256=inputs,
        statistical_unit='identity; mean of valid arms then2000 bootstrap RNG32042',
        purpose='complete auxiliary metrics only; original Gate and decisions unchanged',complete=True))
    files={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*.json')) if p.name!='artifact_manifest.json'}
    write(DEST/'artifact_manifest.json',dict(files=files,file_count=len(files)))
    path=OUT/'supplement_review.tar.gz'
    with tarfile.open(path,'w:gz') as tar:tar.add(DEST,arcname='supplement')
    print('[MS supplement complete]',path,'SHA256',sha(path),flush=True)

if __name__=='__main__':run()
