"""A的独立结构评测；旧指标与v3纹理协议两臂同算，内部线只作为代理。"""
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from tools.e2_fft_experiment import read, write, sha, setup


def boundary_f1(predicted, target, radius):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    a = predicted & ~binary_erosion(predicted,structure=np.ones((3,3)),border_value=0)
    b = target & ~binary_erosion(target,structure=np.ones((3,3)),border_value=0)
    if not a.any() or not b.any():
        return 0.
    precision=float((distance_transform_edt(~b)[a]<=radius).mean())
    recall=float((distance_transform_edt(~a)[b]<=radius).mean())
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def local_response(image):
    from scipy.ndimage import gaussian_filter, maximum_filter
    x=np.asarray(image.convert('RGB'),dtype=np.float64)/255.
    gray=x@np.array([.2989,.5870,.1140])
    gy,gx=np.gradient(gaussian_filter(gray,1.))
    magnitude=np.hypot(gx,gy)
    return maximum_filter(magnitude/(magnitude+.02),size=5)


def training_audit(out):
    reuse=read(out/'audit/baseline_reuse.json')
    assert read(out/'audit/baseline_replay.json')['passed']
    assert read(out/'smoke/result.json')['passed']
    assert read(out/'audit/training_inputs_final.json')['passed']
    assert read(out/'audit/mask_inputs_final.json')['passed']
    checkpoints={'A0':Path(reuse['checkpoint']),'A5':out/'training/A5/latest.pt'}
    baseline=None;result={}
    for arm in ['A0','A5']:
        complete=read(out/'training'/arm/'complete.json')
        assert complete['status']=='completed' and complete['updates']==2000
        assert sha(checkpoints[arm])==complete['checkpoint_sha256']
        rows=[__import__('json').loads(t) for t in (out/'training'/arm/'training.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,2001))
        assert all(np.isfinite(r['grad_norm']) and all(np.isfinite(v) for v in r['losses'].values()) for r in rows)
        tokens=[r['tokens'] for r in rows]
        if baseline is None:baseline=tokens
        else:assert baseline==tokens
        result[arm]=dict(updates=2000,checkpoint_sha256=complete['checkpoint_sha256'],
            training_seconds=sum(r['seconds'] for r in rows),reused=arm=='A0',
            peak_gib=max(r['peak_gib'] for r in rows))
    write(out/'audit/training_completion.json',dict(passed=True,same_legacy_stream=True,arms=result))


def generate(args):
    from tools import e2_b_screen_evaluation as legacy
    from tools.e2_a_minimal import load_a,a_environment
    # 复用已验收生成函数，替换的只是加载器；不改旧B源码及生成定义。
    with patch.object(legacy,'load',load_a):
        legacy.generate(args)
    path=Path(args.out)/'generation/A5.json';value=read(path)
    value['environment']=a_environment();write(path,value)


def evaluate(args):
    from tools import e2_b_screen_evaluation as legacy
    from tools.bf_t3_three_arm_report import paired
    from tools.bf_t3_ablation_report import holm
    from tools.e2_a_minimal import a_environment
    from models.e2_structure import structure_maps
    from eval.eval_utils import prepare_evaluation_masks, estimate_foreground_mask
    import torch
    core,out=setup(args)
    with patch.object(legacy,'ARMS',['A0','A5']),patch.object(legacy,'training_audit',training_audit):
        legacy.evaluate(args)
    texture=read(out/'metrics/paired_statistics.json')
    for r in texture:r['family']='texture_gray_strict_guard' if r['family']=='primary' else 'texture_other_guard'
    write(out/'metrics/texture_paired_statistics.json',texture)
    legacy_rows=read(out/'metrics/per_image.json')
    old_lookup={(r['arm'],r['case_id'],r['condition']):r for r in legacy_rows}
    rows=[]
    for pair in read(out/'strict_pairs.json')['pairs']:
        sketch=Path(args.data)/pair['sketch']
        mask=prepare_evaluation_masks((384,512),sketch_path=str(sketch),mask_policy='sketch_only')['garment']
        assert mask.any()
        maps=structure_maps(core.tensor(sketch))
        candidate=maps[0,4].cpu().numpy()>.5
        gt=Image.open(Path(args.data)/pair['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        target_response=local_response(gt)
        eligible=candidate & (target_response>=.05)
        for arm in ['A0','A5']:
            for condition in ['old','new']:
                path=out/'outputs'/arm/pair['case_id']/(condition+'.png')
                image=Image.open(path).convert('RGB')
                predicted=estimate_foreground_mask(image,(384,512))
                response=local_response(image)
                inner=float(np.minimum(response[eligible]/target_response[eligible],1.).mean()) if eligible.any() else None
                row=dict(arm=arm,condition=condition,case_id=pair['case_id'],index=pair['index'],identity_group=pair['identity_group'],
                    predicted_mask_area=float(predicted.mean()),inner_candidate_pixels=int(candidate.sum()),
                    inner_eligible_pixels=int(eligible.sum()),inner_support_proxy=inner)
                for radius in [1,2,4]:row['boundary_f1_r%d'%radius]=boundary_f1(predicted,mask,radius)
                rows.append(row)
    lookup={(r['arm'],r['case_id'],r['condition']):r for r in rows}
    comparisons=[]
    for metric,source,key in [('old_boundary_f1_r2',lookup,'boundary_f1_r2'),
                              ('old_edge_f1',old_lookup,'edge_f1'),('old_iou',old_lookup,'iou'),
                              ('old_inner_support_proxy',lookup,'inner_support_proxy')]:
        delta=[]
        for pair in read(out/'strict_pairs.json')['pairs']:
            a,b=[source[arm,pair['case_id'],'old'][key] for arm in ['A0','A5']]
            if a is not None and b is not None:delta.append(b-a)
        comparisons.append(dict(comparison='A5_minus_A0',metric=metric,
            family='primary' if metric=='old_boundary_f1_r2' else 'secondary',**paired(delta)))
    for family in ['primary','secondary']:holm([r for r in comparisons if r['family']==family])
    write(out/'metrics/structure_per_image.json',rows)
    write(out/'metrics/paired_statistics.json',comparisons)
    version=read(out/'metrics/version.json')
    version.update(protocol=read(out/'protocol.json'),primary_comparisons=1,secondary_comparisons=3,
        texture_comparisons=4,environment=a_environment(),
        boundary='mask estimated independently from generated RGB by existing v1 foreground estimator; inner contour3x3; Euclidean tolerance1/2/4pixels; primary r2 old condition',
        inner='candidate skeleton from input sketch,exclude outer9x9; GT response>=.05 eligibility fixed for both; Gaussian sigma1+gradient/(gradient+.02),maxpool5; clipped generated/GT response ratio; proxy not semantic F1')
    write(out/'metrics/version.json',version)
    print('A_EVALUATION_COMPLETED',flush=True)


def bundle(args):
    out=Path(args.out)
    prefixes=['inputs','audit','smoke','comparison','training','generation','outputs','metrics','logs','tests','regression']
    files=sorted({p for name in prefixes for p in (out/name).rglob('*') if p.is_file()
                  and p.suffix in ['.json','.jsonl','.png','.jpg','.jpeg','.txt','.out','.err','.csv']})
    files += [out/n for n in ['strict_pairs.json','split.json','protocol.json']]
    write(out/'bundle_manifest.json',{p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with zipfile.ZipFile(out/'review_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'bundle_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    write(out/'bundle_info.json',dict(bytes=(out/'review_results.zip').stat().st_size,sha256=sha(out/'review_results.zip'),files=len(files)))
    print('A_BUNDLE',sha(out/'review_results.zip'),flush=True)
