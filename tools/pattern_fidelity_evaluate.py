"""参考输入校准/冻结，随后运行同一套RGB与灰度纹样评价。"""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from tools.t3_p0p1 import read,write,sha,csv_write
from tools.pattern_fidelity_core import PROTOCOL,descriptors,compare,stable_period,gray_rgb,js,spectrum,direction,period
from tools.pattern_fidelity_run import NAME,SERVER,SOURCE,reference_images,frozen,resolve,asset


def labels():
    with (SOURCE/'tools/pattern_reference_labels.csv').open(encoding='utf-8-sig',newline='') as stream:
        return {r['reference_sha256']:r for r in csv.DictReader(stream)}


def calibration(root):
    x,y=np.meshgrid(np.arange(64),np.arange(64))
    def stripe(period=8,angle=0,phase=0):
        a=.5+.4*np.sin(2*np.pi*(x*np.cos(angle)+y*np.sin(angle))/period+phase)
        return np.repeat(np.round(a[...,None]*255).astype('uint8'),3,axis=2)
    cases=[]
    for p,a in [(8,0),(16,0),(8,np.pi/4),(8,np.pi/2)]:
        image=stripe(p,a);orientation=direction(image);per=period(image)
        true_angle=(a*180/np.pi+90)%180
        error=min(abs(orientation['angle']-true_angle),180-abs(orientation['angle']-true_angle))
        assert error<2 and per['value'] is not None and abs(per['value']-p/64)<.04
        cases.append(dict(period_pixels=p,frequency_normal_angle=a*180/np.pi,estimated_line_angle=orientation['angle'],orientation_error=error,estimated_period=per['value']))
    a=stripe();self_distance=js(spectrum(a),spectrum(a));rotation=js(spectrum(a),spectrum(stripe(angle=np.pi/2)));doubling=js(spectrum(a),spectrum(stripe(16)))
    assert self_distance<1e-10 and rotation>.1 and doubling>.1
    assert spectrum(np.ones((64,64,3),dtype='uint8')*128) is None
    sensitivity=[]
    references=read(root/'pattern_labels/reference_inputs.json')
    # 只取每张参考的固定首patch；所有参考均纳入，未看任何模型图/分数。
    for row in references:
        parts=reference_images(root/row['path']);patch=parts[0];base=spectrum(patch)
        if base is None:continue
        variants={'brightness_1.05':np.clip(patch.astype(float)*1.05,0,255).astype('uint8'),
            'translation_2px':np.roll(patch,2,axis=1),
            'crop_center60_resize64':np.asarray(Image.fromarray(patch[2:62,2:62]).resize((64,64),Image.Resampling.BILINEAR)),
            'blur_sigma0.5':cv2.GaussianBlur(patch,(0,0),.5)}
        for name,changed in variants.items():
            value=spectrum(changed)
            sensitivity.append(dict(reference_sha256=row['reference_sha256'],perturbation=name,gray_fft_js=js(base,value) if value is not None else None))
    csv_write(root/'audit/reference_sensitivity.csv',sensitivity)
    write(root/'audit/synthetic_controls.json',dict(passed=True,cases=cases,self_fft=self_distance,rotation90_fft=rotation,period_doubling_fft=doubling,
        constant_patch='spectrum N/A, not zero distance',source='synthetic controls and reference-only inputs; no generated model scores',
        caveat='positive controls check code behavior, not human perceptual validity'))


def release(project):
    root=project/'output_eval'/NAME
    initial=read(root/'protocol/initial_code.json')
    assert sha(SOURCE/'tools/pattern_fidelity_core.py')==initial['files']['tools/pattern_fidelity_core.py']
    annotation=labels();refs=read(root/'pattern_labels/reference_inputs.json')
    assert all(r['reference_sha256'] in annotation for r in refs)
    assert all(annotation[r['reference_sha256']]['category'] in PROTOCOL['category_rules'] for r in refs)
    frozen(root/'pattern_labels/reference_pattern_labels.json',[dict(r,**annotation[r['reference_sha256']]) for r in refs])
    csv_write(root/'pattern_labels/reference_pattern_labels.csv',read(root/'pattern_labels/reference_pattern_labels.json'))
    gt_sources={r['id']:r['gt'] for r in read(resolve(project,SERVER/'output_eval/t3_p0_p1_20261028/protocol/input_manifest.json')) if r['condition']=='original'}
    gt_sources.update({r['id']:r['gt'] for r in read(resolve(project,SERVER/'output_eval/t3_reference_response_20261027/splits/confirmation.json'))})
    gt_assets=[]
    for case in read(root/'manifests/identities.json'):
        relative,digest=asset(root,resolve(project,gt_sources[case['id']]))
        gt_assets.append(dict(split=case['split'],id=case['id'],path=relative,sha256=digest,usage='legacy structure/background evaluation only; not body mask or inference'))
    frozen(root/'manifests/gt_evaluation_assets.json',gt_assets)
    calibration(root)
    eligibility=[];cache={}
    for case in read(root/'manifests/identities.json'):
        for role in ['old','new']:
            key=case[role+'_sha256']
            if key not in cache:cache[key]=descriptors(reference_images(root/case[role]))
            desc=cache[key];category=annotation[key]['category'];regular=category in PROTOCOL['periodicity']['classes']
            stable=stable_period(desc)
            eligibility.append(dict(split=case['split'],id=case['id'],condition=role,reference_sha256=key,category=category,
                patch_n=case['patch_n'],valid_body=case['valid_body'],reference_regular_class=regular,
                reference_period=stable,reference_stable_period=stable is not None,
                primary_eligible=bool(case['valid_body'] and regular and stable is not None),
                reason=None if case['valid_body'] and regular and stable is not None else 'insufficient strict body patches' if not case['valid_body'] else 'nonregular/ambiguous reference class' if not regular else 'no stable reference period'))
    csv_write(root/'manifests/metric_validity.csv',eligibility)
    counts={}
    for split in sorted({r['split'] for r in eligibility}):
        selected=[r for r in eligibility if r['split']==split and r['condition']=='new']
        counts[split]=dict(identities=len(selected),body_valid=sum(r['valid_body'] for r in selected),primary_eligible=sum(r['primary_eligible'] for r in selected),
            new_reference_categories={category:sum(r['category']==category for r in selected) for category in PROTOCOL['category_rules']})
    extension=dict(color='CIE Lab D65 OpenCVfloat32, pooled fixed body patches,8x8x8 histogram bins L[0,100],ab[-128,128]; JSbase2 old-new margin',
        gray_dists='per64patch Rec709 luma; (luma-mean)/max(std,2/255)*.18+.5 clipped[0,1], repeat3channels; same normalization for all models',
        primary_eligibility='reference-only regular label+stable apparent period+>=4 strict body patches; generated-period reliability never selects primary FFT cases',
        flat_generated_fft='null with explicit invalid-output reason; per-model and matched denominators; no replacement',
        STSIM='not run: located supervised STSIM-C code has no explicit license in repository listing; original2013 equivalence not verified; no SSIM substitution',
        motif='not run: no validated detector/OCR for these low-resolution textile patches',
        main_endpoint='gray_fft_new, lower favorable; prespecified3contrast Holm; all evidence exploratory',
        all_new_metric_minimum_meaningful_difference=None)
    frozen(root/'protocol/metric_extension.json',extension)
    value=dict(allow_evaluation=True,reference_only_counts=counts,labels_sha256=sha(SOURCE/'tools/pattern_reference_labels.csv'),
        protocol_sha256=sha(root/'protocol/config.json'),metric_extension_sha256=sha(root/'protocol/metric_extension.json'),
        core_sha256=sha(SOURCE/'tools/pattern_fidelity_core.py'),evaluator_sha256=sha(Path(__file__)),
        commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        model_scores_viewed_before_freeze=False,independent_confirmation=False)
    frozen(root/'protocol/evaluation_release.json',value)
    print('RELEASED',counts,flush=True)


def lab_hist(parts):
    array=np.concatenate(parts,axis=0).astype('float32')/255
    lab=cv2.cvtColor(array,cv2.COLOR_RGB2LAB)
    hist=np.histogramdd(lab.reshape(-1,3),bins=[8,8,8],range=[(0,100),(-128,128),(-128,128)])[0]
    return hist/hist.sum()


def evaluate(project,split,models):
    import torch
    from tools import t3_response_run as response
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation,compute_texture_leakage
    root=project/'output_eval'/NAME;release=read(root/'protocol/evaluation_release.json')
    assert release['allow_evaluation'] and sha(SOURCE/'tools/pattern_fidelity_core.py')==release['core_sha256']
    assert sha(Path(__file__))==release['evaluator_sha256']
    response.ROOT=root/'runtime_assets'/split/models[0]
    model=response.load_dists('cuda');sanity=response.sanity(model)
    write(root/'audit'/('DISTS_sanity_'+split+'_'+models[0]+'.json'),sanity)
    torch.set_num_threads(2)
    def features(parts,gray=False):
        if gray:parts=[gray_rgb(im) for im in parts]
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().cuda()/255
        return model.forward_once(value)
    cases={(r['split'],r['id']):r for r in read(root/'manifests/identities.json')}
    if split=='P2b_swap32':
        targets={r['id']:r['gt'] for r in read(resolve(project,SERVER/'output_eval/t3_p0_p1_20261028/protocol/input_manifest.json')) if r['condition']=='original'}
    else:targets={r['id']:r['gt'] for r in read(resolve(project,SERVER/'output_eval/t3_reference_response_20261027/splits/confirmation.json'))}
    annotation=labels();refcache={};records=[];crop_manifest=[]
    with torch.inference_mode():
        for row in read(root/'manifests/outputs.json'):
            if row['split']!=split or row['model_id'] not in models:continue
            identity=row['identity_id'];arm=row['model_id'];case=cases[split,identity]
            gt=resolve(project,targets[identity]);gt_relative,gt_sha=asset(root,gt)
            assert gt_sha==row['metadata_old']['inputs']['gt'] if 'inputs' in row['metadata_old'] else True
            target=root/'metrics/per_image'/split/arm/(identity+'.json')
            if target.exists():records.extend(read(target));continue
            assert sha(root/case['mask'])==case['mask_sha256']
            category=annotation[case['new_sha256']]['category'];regular=category in PROTOCOL['periodicity']['classes']
            directional=regular
            refs={};refdesc={};refhist={};ref_features={}
            for role in ['old','new']:
                key=case[role+'_sha256']
                if key not in refcache:
                    parts=reference_images(root/case[role]);refcache[key]=(parts,descriptors(parts),lab_hist(parts))
                refs[role],refdesc[role],refhist[role]=refcache[key]
                if case['valid_body']:ref_features[role]={view:features(refs[role],view=='gray') for view in ['rgb','gray']}
            batch=[]
            for condition,output_role in [('original','old'),('swapped','new')]:
                path=Path(row['output_'+output_role+'_path']);assert sha(path)==row['output_'+output_role+'_sha256']
                image=np.asarray(Image.open(path).convert('RGB'));coords=case['coords']
                parts=[image[y:y+64,x:x+64] for x,y in coords]
                rec=dict(split=split,id=identity,model=arm,condition=condition,category=category,
                    output_sha256=sha(path),patch_n=len(parts),valid_body=case['valid_body'],
                    primary_eligible=bool(case['valid_body'] and regular and stable_period(refdesc['new']) is not None),
                    invalid_reason=None if case['valid_body'] else 'fewer than4 strict body patches')
                keys=['gray_fft','rgb_fft','orientation_hist','orientation_angle','period_value','period_x','period_y','dists_rgb','dists_gray','color']
                rec.update({key+'_'+role:None for key in keys for role in ['old','new','margin']})
                if case['valid_body']:
                    desc=descriptors(parts);hist=lab_hist(parts)
                    rec['orientation_confidence_gen']=float(np.median([d['orientation']['confidence'] for d in desc]))
                    rec['period_confidence_gen']=float(np.median([d['period']['confidence'] for d in desc]))
                    feature={view:features(parts,view=='gray') for view in ['rgb','gray']}
                    for role in ['old','new']:
                        rec['orientation_confidence_ref_'+role]=float(np.median([d['orientation']['confidence'] for d in refdesc[role]]))
                        rec['period_confidence_ref_'+role]=float(np.median([d['period']['confidence'] for d in refdesc[role]]))
                        comparison=compare(desc,refdesc[role],regular,directional)
                        for key,value in comparison.items():rec[key+'_'+role]=value
                        for view in ['rgb','gray']:rec['dists_'+view+'_'+role]=response.patch_distance(model,feature[view],ref_features[role][view])
                        rec['color_'+role]=js(hist,refhist[role])
                    for key in keys:
                        a=rec[key+'_old'];b=rec[key+'_new'];rec[key+'_margin']=a-b if a is not None and b is not None else None
                    rec['valid_gray_fft']=rec['gray_fft_new'] is not None
                    rec['valid_orientation']=rec['orientation_hist_new'] is not None
                    rec['valid_periodicity']=rec['period_value_new'] is not None
                    if rec['gray_fft_new'] is None:rec['fft_invalid_reason']='no non-DC generated/reference spectral energy; retained'
                    if regular and rec['period_value_new'] is None:rec['period_invalid_reason']='reference or generated patch periods not stable; retained N/A'
                    if directional and rec['orientation_angle_new'] is None:rec['orientation_invalid_reason']='insufficient dominant-direction confidence; histogram separate'
                    strip=Image.fromarray(np.concatenate(parts,axis=1));crop=root/'crops/generated_garment_regions'/split/arm/(identity+'_'+condition+'.png');crop.parent.mkdir(parents=True,exist_ok=True);strip.save(crop)
                    crop_manifest.append(dict(split=split,id=identity,model=arm,condition=condition,path=crop.relative_to(root).as_posix(),sha256=sha(crop),patch_n=len(parts),coords=json.dumps(coords)))
                bundle=prepare_evaluation_masks((384,512),sketch_path=str(root/case['sketch']),mask_policy='sketch_only')
                assert np.array_equal(bundle['garment'],np.asarray(Image.open(root/case['mask']))>127)
                kwargs=dict(gen_path=str(path),sketch_path=str(root/case['sketch']),target_path=str(gt),mask_bundle=bundle)
                structure=compute_structure_preservation(**kwargs);leak=compute_texture_leakage(**kwargs)
                rec.update(edge_f1=structure.get('edge_f1'),iou=structure.get('sketch_iou'),edge_l1=structure.get('edge_l1'),leak=leak.get('leak_colored_frac'),
                    GT_for_legacy_structure_background_only=gt_relative,GT_sha256=gt_sha,GT_used_in_generation=False)
                batch.append(rec)
            write(target,batch);records.extend(batch)
            print('PATTERN_METRIC',split,arm,identity,len(records),flush=True)
    name=split+'_'+models[0]
    csv_write(root/'metrics'/('combined_'+name+'.csv'),records)
    csv_write(root/'manifests'/('generated_patches_'+name+'.csv'),crop_manifest)
    write(root/'audit'/('evaluation_complete_'+name+'.json'),dict(complete=True,rows=len(records),models=models,split=split,
        code_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),metric_code_sha256=sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['release','evaluate'])
    parser.add_argument('--project',type=Path,default=SERVER);parser.add_argument('--split',default='P2b_swap32');parser.add_argument('--models',nargs='+',default=['T3'])
    args=parser.parse_args()
    if args.action=='release':release(args.project)
    else:evaluate(args.project,args.split,args.models)
