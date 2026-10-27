"""E2A-TEXTURE-EVAL-01：固定 patch、官方 DISTS、FFT/自相关和身份配对统计。

只读取已有 RGB 输出。合成图仅用于独立的估计器校验，不充当模型生成结果。
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, signal
import torch
from e2a_texture_prepare import ROOT, NEW, OLD, ARMS, sha, read, write, csv_write

def csv_read(p):
    return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))

def yes(v):return str(v).lower()=='true'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def angle_difference(a,b):
    return np.abs((np.asarray(a)-np.asarray(b)+90)%180-90)

def assignment(a,b):
    return min(itertools.permutations(range(len(a))),key=lambda p:float(np.mean(angle_difference(np.asarray(a)[list(p)],b))))

def make_config():
    return dict(protocol='E2A-TEXTURE-EVAL-01',created_utc=now(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        metric_source_sha256=sha(__file__),preparation_source_sha256=sha(Path(__file__).with_name('e2a_texture_prepare.py')),
        input_manifest_sha256=sha(ROOT/'input_manifest.csv'),labels_sha256=sha(ROOT/'reference_pattern_labels.csv'),
        mask_manifest_sha256=sha(ROOT/'mask_manifest.csv'),annotation_prior_exposure=True,development_only=True,
        reference_canvas=dict(size=[384,512],RGB=True,resampling='Pillow LANCZOS',crop='none',range=[0,1],
            rationale='Reconstruct BF RGB conditioning canvas; CLIP224 conditioning is a separate branch, not used for this metric.',
            source_pipeline_sha256=sha(NEW.parent.parent.parent/'pipelines/IMAGGarment_pipeline.py') if (NEW.parent.parent.parent/'pipelines/IMAGGarment_pipeline.py').exists() else None,
            caveat='Reconstruction rule, not a saved original conditioning tensor; local Pillow version recorded.'),
        generated_canvas=dict(size=[384,512],resampling='none',RGB=True,range=[0,1]),
        patch=dict(size=64,stride=32,interior_erosion_radius=5,erosion_kernel='square11',occupancy_min=.90,generated_max=8,generated_min=4,reference_max=16,reference_grid='4x4 endpoint-inclusive rounded linspace over384x512',generated_order='row-major candidates then rounded linspace subsample'),
        dists=dict(source='https://github.com/dingkeyan93/DISTS',commit='1267d8cb626c98706db3697422701c56a85ebf2e',
            source_sha256='69b63ebd4e1aece0f79a3b922ba803b5f6fec127babee94869e9eceaa42b4612',
            weights_sha256='f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218',
            backbone_sha256='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
            distance_direction='official forward: 1-(structure+texture), lower better',
            aggregation='.5*(mean_i min_j d(g_i,r_j)+mean_j min_i d(g_i,r_j))',
            feature_cache='float32 official forward_once; channelwise centered covariance bmm; compared against official forward',
            sanity_self_tolerance=1e-5,sanity_formula_tolerance=2e-6,sanity_same_pattern_win_fraction_min=.75,
            sanity_scope='16 independent procedural texture pairs, identical palette; synthetic calibration, not human motif validation',
            derived_diagnostic_not_standard_benchmark=True,T1_uses_DISTS_in_training=True),
        fft=dict(luminance=[.2126,.7152,.0722],remove_mean=True,window='outer(np.hanning(64),np.hanning(64))',
            min_cycles=2,max_cycles=20,min_luminance_sd=.015,min_angular_power_fraction=.35,
            angular_halfwidth_degrees=10,peak_to_annulus_median_min=5,orthogonal_axis_tolerance_degrees=20,
            second_axis_fraction_min=.20,fundamental_min_relative_peak=.25,
            plaid_harmonic_interpretation='compare raw FFT normals with (q1+q2)/2 and (q1-q2)/2 checker lattice normals; select using independent spatial gradient normal support, before target metrics',
            plaid_gradient_support_ratio=1.2,plaid_raw_frequency_separation_min_degrees=20,
            period_min_pixels=3.2,period_max_pixels=32,autocorrelation='IFFT2 of same Hann-windowed 2D power',
            acf_first_peak_min_height=.10,acf_first_peak_prominence=.025,acf_fft_relative_tolerance=.30,
            reference_patch_min_detected=4,generated_patch_min_detected='max(2,ceil(n/2))',
            orientation='frequency normal +90 degrees modulo180; x right,y down',
            axis_count='2 for plaid/check, dot arrays, and preannotated R095 houndstooth; otherwise1',
            identity_aggregation='circular medoid axes; median assigned-axis period; median patch errors',
            eligible_detection_failure='retain all reference-eligible identities, including reference estimator failures; no relabel toNA'),
        stats=dict(seed=221042,bootstrap=10000,signflip=10000,tie_tolerance=1e-8,CI='paired identity percentile95%',
            primary_metric='swap DISTS ResponseMargin, all32 common identities with sufficient patches',
            primary_comparisons=['C-T3M','C-T3'],primary_family_correction='Holm two-sided paired mean signflip +1 numerator/denominator',
            missing_policy='no imputation; report all planned denominator and failure rates; pairwise complete differences explicit',
            motif_subset='reference category changed with at least one pattern_evaluable reference; conservative documented category-change subset, not all within-category motif changes',
            same_category_swaps='cannot count as proven motif-identity change; distances still reported',
            human_trigger='both primary ResponseMargin differences have CI lower>0 and Holm p<.05, procedural DISTS sanity passed; genuine unexposed human validation still needed',
            tails='worst10% oriented C differences for every reported comparison/metric; ceil',
            no_confirmatory_LOCK=True),
        no_training=True,no_regeneration=True,old_stop_decision_unchanged=True)

def build_dists(config):
    p=OLD/'assets/DISTS/DISTS_pytorch/DISTS_pt.py'; assert sha(p)==config['dists']['source_sha256']
    wp=ROOT/'assets/dists_weights.pt';bp=ROOT/'assets/vgg16-397923af.pth'
    assert sha(wp)==config['dists']['weights_sha256'] and sha(bp)==config['dists']['backbone_sha256']
    spec=importlib.util.spec_from_file_location('official_frozen_dists',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    real=m.models.vgg16
    backbone=real(weights=None)
    backbone.load_state_dict(torch.load(bp,map_location='cpu',weights_only=True),strict=True)
    # 原文件保持只读；拦截构造器以加载同 SHA 离线权重，避免隐式联网或随机 backbone。
    m.models.vgg16=lambda **kwargs:backbone
    try:model=m.DISTS(load_weights=False)
    finally:m.models.vgg16=real
    weights=torch.load(wp,map_location='cpu',weights_only=True)
    assert set(weights)=={'alpha','beta'}
    with torch.no_grad():model.alpha.copy_(weights['alpha']);model.beta.copy_(weights['beta'])
    return model.eval().requires_grad_(False)

def tensor(patches):return torch.from_numpy(np.stack(patches).transpose(0,3,1,2).copy()).float()/255

def distances(model,fx,fy):
    # 官方逐通道公式的矩阵形式；不修改官方层、alpha/beta 或常数。
    weights=(model.alpha/(model.alpha.sum()+model.beta.sum())).flatten().split(model.chns)
    weights2=(model.beta/(model.alpha.sum()+model.beta.sum())).flatten().split(model.chns)
    score=torch.zeros((len(fx[0]),len(fy[0])))
    for x,y,a,b in zip(fx,fy,weights,weights2):
        x=x.flatten(2);y=y.flatten(2);mx=x.mean(-1);my=y.mean(-1)
        xc=x-mx[:,:,None];yc=y-my[:,:,None]
        vx=xc.square().mean(-1);vy=yc.square().mean(-1)
        cov=torch.bmm(xc.permute(1,0,2),yc.permute(1,2,0)).permute(1,2,0)/x.shape[-1]
        s1=(2*mx[:,None,:]*my[None,:,:]+1e-6)/(mx[:,None,:].square()+my[None,:,:].square()+1e-6)
        s2=(2*cov+1e-6)/(vx[:,None,:]+vy[None,:,:]+1e-6)
        score+=(s1*a+s2*b).sum(-1)
    return 1-score

def patch_set_distance(model,g,r):
    matrix=distances(model,g,r)
    a=float(matrix.min(1).values.mean());b=float(matrix.min(0).values.mean())
    return {'dists':.5*(a+b),'gen_to_ref':a,'ref_to_gen':b}

def procedural(kind,angle=0,period=12,size=128,phase=0):
    y,x=np.mgrid[:size,:size];t=np.deg2rad(angle)
    u=x*np.cos(t)+y*np.sin(t)+phase;v=-x*np.sin(t)+y*np.cos(t)+phase
    if kind=='stripe':a=np.sin(2*np.pi*u/period)>0
    elif kind=='plaid':a=(np.sin(2*np.pi*u/period)>0) ^ (np.sin(2*np.pi*v/period)>0)
    elif kind=='grid':a=(np.cos(2*np.pi*u/period)>.7)|(np.cos(2*np.pi*v/period)>.7)
    elif kind=='dot':a=(np.cos(2*np.pi*u/period)+np.cos(2*np.pi*v/period))>1
    elif kind=='noise':a=np.random.default_rng(221042).random((size,size))>.5
    else:raise ValueError(kind)
    return np.repeat(np.where(a,.85,.15)[:,:,None],3,axis=2)

def spectral(patch,axes,config,plaid=False):
    c=config['fft'];rgb=patch.astype(np.float64)/255;gray=rgb@np.array(c['luminance'])
    empty=dict(orientation_detected=False,period_detected=False,angles=[],periods=[],acf_periods=[],reason='low_contrast')
    if gray.std()<c['min_luminance_sd']:return empty
    v=(gray-gray.mean())*np.outer(np.hanning(64),np.hanning(64))
    power=np.abs(np.fft.fftshift(np.fft.fft2(v)))**2
    fy,fx=np.mgrid[-32:32,-32:32];rad=np.hypot(fx,fy);ang=np.degrees(np.arctan2(fy,fx))%180
    ann=(rad>=c['min_cycles'])&(rad<=c['max_cycles']);half=ann&((fy>0)|((fy==0)&(fx>0)))
    total=power[half].sum();median=np.median(power[ann])+1e-20
    peaks=(power==ndimage.maximum_filter(power,size=3))&half
    coords=sorted(zip(*np.where(peaks)),key=lambda ij:-power[ij]);chosen=[]
    for ij in coords:
        theta=float(ang[ij]);fraction=float(power[half&(angle_difference(ang,theta)<=c['angular_halfwidth_degrees'])].sum()/(total+1e-20))
        if power[ij]/median<c['peak_to_annulus_median_min']:continue
        minimum=c['min_angular_power_fraction'] if not chosen else c['second_axis_fraction_min']
        if fraction<minimum:continue
        if chosen:
            separation=float(angle_difference(theta,chosen[0]['normal']))
            if plaid:
                if separation<c['plaid_raw_frequency_separation_min_degrees']:continue
            elif abs(separation-90)>c['orthogonal_axis_tolerance_degrees']:continue
        # 选择同轴局部峰中达到主峰25%的最低频，保留明确的谐波规则。
        candidates=[k for k in coords if angle_difference(ang[k],theta)<=6 and power[k]>=power[ij]*c['fundamental_min_relative_peak']]
        fundamental=min(candidates,key=lambda k:rad[k]);frequency=float(rad[fundamental])
        chosen.append(dict(normal=theta,angle=(theta+90)%180,period=64/frequency,fraction=fraction))
        if len(chosen)==axes:break
    if len(chosen)<axes:return dict(empty,reason='missing_reliable_axes',detected_axes=len(chosen))
    interpretation='direct_frequency_normal'
    if plaid:
        # 棋盘格是两个一维信号的乘积，FFT 主峰为和/差频；格线不沿对角主峰。
        # 独立校验同时包含乘积棋盘格和加性网格，禁止一律旋转45度。
        q=[np.array([np.cos(np.deg2rad(r['normal'])),np.sin(np.deg2rad(r['normal']))])/r['period'] for r in chosen]
        alternate=[]
        for vector in [(q[0]+q[1])/2,(q[0]-q[1])/2]:
            normal=float(np.rad2deg(np.arctan2(vector[1],vector[0]))%180)
            alternate.append(dict(normal=normal,angle=(normal+90)%180,period=float(1/np.linalg.norm(vector)),fraction=0.))
        gy,gx=np.gradient(gray);ga=np.rad2deg(np.arctan2(gy,gx))%180;mag=np.hypot(gx,gy)
        def support(hypothesis):return sum(float(mag[angle_difference(ga,r['normal'])<=10].sum()) for r in hypothesis)/(float(mag.sum())+1e-20)
        raw_score,alt_score=support(chosen),support(alternate)
        alt_orthogonal=abs(float(angle_difference(alternate[0]['normal'],alternate[1]['normal']))-90)<=c['orthogonal_axis_tolerance_degrees']
        if alt_orthogonal and alt_score>raw_score*c['plaid_gradient_support_ratio']:
            chosen=alternate;interpretation='checker_sum_difference_frequencies'
        if abs(float(angle_difference(chosen[0]['normal'],chosen[1]['normal']))-90)>c['orthogonal_axis_tolerance_degrees']:
            return dict(empty,reason='no_orthogonal_spatial_axes_after_harmonic_check')
    acf=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(power)).real);acf/=acf[32,32]+1e-20
    acfs=[];ok=True
    lags=np.arange(0,32.01,.25)
    for item in chosen:
        t=np.deg2rad(item['normal']);profile=ndimage.map_coordinates(acf,[32+lags*np.sin(t),32+lags*np.cos(t)],order=1,mode='nearest')
        pk,pr=signal.find_peaks(profile,height=c['acf_first_peak_min_height'],prominence=c['acf_first_peak_prominence'])
        pk=[p for p in pk if lags[p]>=c['period_min_pixels']]
        first=float(lags[pk[0]]) if pk else None
        acfs.append(first)
        if first is None or abs(first-item['period'])/item['period']>c['acf_fft_relative_tolerance']:ok=False
    return dict(orientation_detected=True,period_detected=ok,angles=[r['angle'] for r in chosen],periods=[r['period']/64 for r in chosen],acf_periods=[None if x is None else x/64 for x in acfs],angular_power_fractions=[r['fraction'] for r in chosen],harmonic_interpretation=interpretation,reason='ok' if ok else 'fft_acf_disagree_or_no_acf_peak')

def axes_for(label):
    return 2 if label['category'] in ['plaid/check','dot'] or int(label['reference_index'])==95 else 1

def calibration(config):
    torch.set_num_threads(4);model=build_dists(config)
    drows=[];formula_error=0
    with torch.inference_mode():
        for i in range(16):
            kind=['stripe','plaid','dot','stripe'][i%4];foil={'stripe':'plaid','plaid':'dot','dot':'stripe'}[kind]
            a=(procedural(kind,angle=(i%4)*15,period=[8,12,16,20][i%4])*255).astype('uint8')
            b=(procedural(foil,angle=(i%4)*15,period=[8,12,16,20][i%4])*255).astype('uint8')
            patches=[a[0:64,0:64],a[7:71,11:75],b[0:64,0:64]];x=tensor(patches)
            f=model.forward_once(x);matrix=distances(model,f,f)
            direct=model(x[:1],x[1:2]);error=abs(float(matrix[0,1])-float(direct));formula_error=max(formula_error,error)
            drows.append(dict(case=i,kind=kind,foil=foil,self_distance=float(matrix[0,0]),same_pattern_crop=float(matrix[0,1]),same_palette_other_pattern=float(matrix[0,2]),same_pattern_closer=bool(matrix[0,1]<matrix[0,2]),cached_formula_error=error))
            for j,p in enumerate(patches):
                out=ROOT/f'dists_sanity_check/images/{i:02}_{j}.png';out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(p).save(out)
    passed=bool(max(abs(r['self_distance']) for r in drows)<config['dists']['sanity_self_tolerance'] and np.mean([r['same_pattern_closer'] for r in drows])>=config['dists']['sanity_same_pattern_win_fraction_min'])
    assert formula_error<config['dists']['sanity_formula_tolerance'],'矩阵公式与官方实现不符，停止计算'
    write(ROOT/'dists_sanity_check/results.json',dict(procedural_sanity_passed=bool(passed),motif_identity_validated_by_humans=False,formula_max_abs_error=formula_error,rows=drows))
    fftrows=[]
    for kind in ['stripe','plaid','grid']:
        for normal in [0,15,30,45,75,90,120]:
            for p in [8,12,16,24]:
                patch=(procedural(kind,normal,p,size=64,phase=3)*255).astype('uint8');out=spectral(patch,1 if kind=='stripe' else 2,config,plaid=kind!='stripe')
                truth=[(normal+90)%180] if kind=='stripe' else [(normal+90)%180,normal%180]
                if out['orientation_detected']:
                    perm=assignment(out['angles'],truth);err=float(np.mean(angle_difference(np.array(out['angles'])[list(perm)],truth)))
                else:err=None
                pe=float(np.mean(abs(np.array(out['periods'])*64-p)/p)*100) if out['period_detected'] else None
                fftrows.append(dict(kind=kind,normal=normal,true_period=p,angle_error=err,period_error_percent=pe,**out))
    noise=(procedural('noise',size=64)*255).astype('uint8');noisecheck=spectral(noise,1,config)
    write(ROOT/'calibration/fft_results.json',dict(parameters_frozen_before_target_metrics=True,synthetic_only=True,rows=fftrows,noise_foil=noisecheck,thresholds=config['fft']))
    # 未要求估计器每例必过；失败原样披露，禁止根据目标臂成绩调整阈值。
    print(json.dumps(dict(dists_sanity_passed=passed,formula_error=formula_error,fft_cases=len(fftrows),orientation_detected=sum(r['orientation_detected'] for r in fftrows),period_detected=sum(r['period_detected'] for r in fftrows),noise_detected=noisecheck['orientation_detected'])),flush=True)

def patch_arrays(image,coords):
    return [np.asarray(image.crop((int(x),int(y),int(x)+64,int(y)+64))) for x,y,*_ in coords]

def representative(estimates,need,period=False):
    good=[e for e in estimates if e['period_detected' if period else 'orientation_detected']]
    if len(good)<need:return None
    medoid=min(good,key=lambda a:sum(float(np.mean(angle_difference(np.array(b['angles'])[list(assignment(b['angles'],a['angles']))],a['angles']))) for b in good))
    periods=[]
    for b in good:periods.append(np.array(b['periods'])[list(assignment(b['angles'],medoid['angles']))])
    return dict(angles=medoid['angles'],periods=np.median(periods,axis=0).tolist())

def errors(estimates,rep,need,period=False):
    good=[e for e in estimates if e['period_detected' if period else 'orientation_detected']]
    if rep is None:return None,[],'reference_detection_failure'
    if len(good)<need:return None,[],'period_not_detected' if period else 'pattern_not_detected'
    values=[]
    for e in good:
        p=assignment(e['angles'],rep['angles'])
        if period:values.append((100*abs(np.array(e['periods'])[list(p)]-rep['periods'])/rep['periods']).tolist())
        else:values.append(angle_difference(np.array(e['angles'])[list(p)],rep['angles']).tolist())
    axis_medians=np.median(values,axis=0).tolist()
    return float(np.median(np.mean(values,axis=1))),axis_medians,'ok'

def run(config):
    assert sha(ROOT/'reference_pattern_labels.csv')==config['labels_sha256']
    assert sha(ROOT/'input_manifest.csv')==config['input_manifest_sha256']
    assert sha(__file__)==config['metric_source_sha256']
    assert not (ROOT/'original_reference/per_identity_metrics.csv').exists(),'不覆盖既有指标'
    torch.set_num_threads(4);model=build_dists(config)
    labels={r['reference_sha256']:r for r in csv_read(ROOT/'reference_pattern_labels.csv')}
    coords=read(ROOT/'generation_patch_coordinates.json');records=csv_read(ROOT/'input_manifest.csv')
    masks={r['identity']:r for r in csv_read(ROOT/'mask_manifest.csv')};pairs={r['identity']:r for r in csv_read(ROOT/'reference_pairs.csv')}
    ref_coords=[(int(x),int(y),1.) for y in np.linspace(0,448,4).round() for x in np.linspace(0,320,4).round()]
    patch_rows=[];spectra_rows=[];ref_specs={}
    for key,label in labels.items():
        img=Image.open(label['reference_path']).convert('RGB').resize((384,512),Image.Resampling.LANCZOS)
        arrs=patch_arrays(img,ref_coords);estimates=[spectral(a,axes_for(label),config,plaid=label['category']=='plaid/check') for a in arrs]
        ref_specs[key]=estimates
        for i,(p,c,e) in enumerate(zip(arrs,ref_coords,estimates)):
            patch_rows.append(dict(kind='reference',identity=label['reference_identity'],arm='',stage='reference',reference_sha256=key,output_sha256='',mask_sha256='',patch_index=i,x=c[0],y=c[1],mask_occupancy=1.,pixel_sha256=hashlib.sha256(p.tobytes()).hexdigest(),valid=True,reason='fixed_reference_grid',metric_version=config['source_commit']))
            spectra_rows.append(dict(kind='reference',identity=label['reference_identity'],arm='',stage='reference',reference_sha256=key,patch_index=i,**e))
    pair_characteristics=[]
    for identity,pair in pairs.items():
        oldkey,newkey=pair['old_reference_sha256'],pair['new_reference_sha256'];old,new=labels[oldkey],labels[newkey]
        record=dict(identity=identity,old_category=old['category'],new_category=new['category'],category_changed=old['category']!=new['category'],both_solid=old['category']==new['category']=='solid',within_category_motif_identity_change='not independently annotated',old_reference_sha256=oldkey,new_reference_sha256=newkey)
        for period,name in [(False,'orientation'),(True,'period')]:
            eligible=yes(old['has_repeat_period' if period else 'has_stable_orientation']) and yes(new['has_repeat_period' if period else 'has_stable_orientation'])
            record[name+'_both_applicable']=eligible
            a=representative(ref_specs[oldkey],4,period);b=representative(ref_specs[newkey],4,period)
            value=None
            if eligible and a is not None and b is not None and len(a['angles'])==len(b['angles']):
                p=assignment(b['angles'],a['angles'])
                value=float(np.mean(100*abs(np.array(b['periods'])[list(p)]-a['periods'])/a['periods'])) if period else float(np.mean(angle_difference(np.array(b['angles'])[list(p)],a['angles'])))
            record[name+'_reference_difference']=value
        pair_characteristics.append(record)
    write(ROOT/'reference_pair_characteristics.json',pair_characteristics)
    color_original={};color_swap={}
    for arm in ARMS:
        base=OLD if arm in ['T1','T2'] else NEW
        color_original[arm]={r['id']:r for r in read(base/f'evaluation/P2b/{arm}/per_identity_metrics.json')}
        color_swap[arm]={r['id']:r for r in read(base/f'evaluation/reference_swap/{arm}/per_identity_swap_metrics.json')}
    @lru_cache(maxsize=8)
    def ref_features(key):
        label=labels[key];img=Image.open(label['reference_path']).convert('RGB').resize((384,512),Image.Resampling.LANCZOS)
        return model.forward_once(tensor(patch_arrays(img,ref_coords)))
    output=[]
    with torch.inference_mode():
        for number,r in enumerate(records,1):
            cs=coords[r['identity']];label=labels[r['reference_sha256']];armspec=axes_for(label)
            arrs=patch_arrays(Image.open(r['output_path']).convert('RGB'),cs)
            estimate_cache={}
            def estim(axes,plaid=False):
                key=(axes,plaid)
                if key not in estimate_cache:estimate_cache[key]=[spectral(a,axes,config,plaid=plaid) for a in arrs]
                return estimate_cache[key]
            estimates=estim(armspec,label['category']=='plaid/check')
            base=dict(stage=r['stage'],arm=r['arm'],identity=r['identity'],reference_sha256=r['reference_sha256'],category=label['category'],mask_low_confidence=yes(masks[r['identity']]['mask_low_confidence']),patch_n=len(arrs),insufficient_interior_patches=len(arrs)<4,pattern_evaluable=yes(label['pattern_evaluable']),orientation_applicable=yes(label['has_stable_orientation']),period_applicable=yes(label['has_repeat_period']))
            if r['stage']=='original':base['tcf_lab_reused']=color_original[r['arm']][r['identity']]['tcf_lab']
            else:
                color=color_swap[r['arm']][r['identity']]
                base['tcf_lab_reused']=color['swapped_to_alternative_color']['tcf_lab_delta']
                base['color_response_margin_reused']=color['swapped_to_original_color']['tcf_lab_delta']-base['tcf_lab_reused']
            for i,(p,c,e) in enumerate(zip(arrs,cs,estimates)):
                patch_rows.append(dict(kind='generated',identity=r['identity'],arm=r['arm'],stage=r['stage'],reference_sha256=r['reference_sha256'],output_sha256=r['output_sha256'],mask_sha256=r['mask_sha256'],patch_index=i,x=c[0],y=c[1],mask_occupancy=c[2],pixel_sha256=hashlib.sha256(p.tobytes()).hexdigest(),valid=len(arrs)>=4,reason='ok' if len(arrs)>=4 else 'insufficient_interior_patches',metric_version=config['source_commit']))
                spectra_rows.append(dict(kind='generated',identity=r['identity'],arm=r['arm'],stage=r['stage'],reference_sha256=r['reference_sha256'],patch_index=i,**e))
            need=max(2,math.ceil(len(arrs)/2));rep=representative(ref_specs[r['reference_sha256']],4);repp=representative(ref_specs[r['reference_sha256']],4,True)
            for period,name,app in [(False,'orientation',base['orientation_applicable']),(True,'period',base['period_applicable'])]:
                if app:
                    value,axesval,status=errors(estimates,repp if period else rep,need,period)
                    base.update({name+'_error':value,name+'_axis_errors':axesval,name+'_status':status,name+'_detected':status=='ok'})
                else:base.update({name+'_error':None,name+'_axis_errors':[],name+'_status':'reference_not_applicable',name+'_detected':False})
                base[name+'_failure']=int(base[name+'_status']!='ok') if app else None
            if len(arrs)>=4:
                features=model.forward_once(tensor(arrs));base.update(patch_set_distance(model,features,ref_features(r['reference_sha256'])))
                base['dists_status']='ok'
                if r['stage']=='swap':
                    oldkey=pairs[r['identity']]['old_reference_sha256'];old=labels[oldkey]
                    distold=patch_set_distance(model,features,ref_features(oldkey))
                    base.update(dists_old=distold['dists'],gen_to_old_ref=distold['gen_to_ref'],old_ref_to_gen=distold['ref_to_gen'],response_margin=distold['dists']-base['dists'],old_category=old['category'],reference_category_changed=old['category']!=label['category'],motif_category_change_subset=old['category']!=label['category'] and (yes(old['pattern_evaluable']) or base['pattern_evaluable']))
                    for period,name in [(False,'orientation'),(True,'period')]:
                        app=yes(old['has_repeat_period' if period else 'has_stable_orientation']) and base[name+'_applicable']
                        base[name+'_response_applicable']=app
                        if app:
                            rp=representative(ref_specs[oldkey],4,period);err,_,status=errors(estim(axes_for(old),old['category']=='plaid/check'),rp,need,period)
                            base[name+'_old_error']=err;base[name+'_old_status']=status
                            base[name+'_response_margin']=err-base[name+'_error'] if err is not None and base[name+'_error'] is not None else None
            else:base.update(dists=None,gen_to_ref=None,ref_to_gen=None,dists_status='insufficient_interior_patches')
            output.append(base)
            if number%24==0:print('MEASURED',number,'/',len(records),flush=True)
    csv_write(ROOT/'patch_manifest.csv',patch_rows)
    def serial(r):return {k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()}
    csv_write(ROOT/'direction/per_patch_angles.csv',[serial({k:v for k,v in r.items() if k not in ['periods','acf_periods']}) for r in spectra_rows])
    csv_write(ROOT/'period/per_patch_periods.csv',[serial(r) for r in spectra_rows])
    for stage,folder in [('original','original_reference'),('swap','swapped_reference')]:
        rows=[r for r in output if r['stage']==stage];csv_write(ROOT/f'{folder}/per_identity_metrics.csv',[serial(r) for r in rows]);write(ROOT/f'{folder}/per_identity_metrics.json',rows)
    write(ROOT/'measurement_complete.json',dict(complete=True,utc=now(),outputs=len(output),patches=len(patch_rows),metric_config_sha256=sha(ROOT/'metric_config.json'),source_commit=config['source_commit'],all_six_arms=True,models_regenerated=False))

def summarize(values,rng):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return dict(n=0,mean=None,median=None,SD=None,CI95=None)
    sample=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),SD=float(values.std(ddof=1)) if len(values)>1 else None,CI95=np.quantile(sample,[.025,.975]).tolist())

def statistics(config):
    datasets={s:read(ROOT/f'{f}/per_identity_metrics.json') for s,f in [('original','original_reference'),('swap','swapped_reference')]}
    rng=np.random.default_rng(config['stats']['seed']);summary={};contrasts=[];tails=[]
    for stage,records in datasets.items():
        metrics=['dists','orientation_error','period_error','orientation_failure','period_failure','tcf_lab_reused']+(['response_margin','orientation_response_margin','period_response_margin','color_response_margin_reused'] if stage=='swap' else [])
        for arm in ARMS:
            rows=[r for r in records if r['arm']==arm];entry={}
            for metric in metrics:
                entry[metric]=summarize([r[metric] for r in rows if r.get(metric) is not None],rng)
            entry['planned_n']=len(rows);entry['low_confidence_mask_n']=sum(r['mask_low_confidence'] for r in rows);entry['insufficient_patch_n']=sum(r['insufficient_interior_patches'] for r in rows)
            for name in ['orientation','period']:
                eligible=[r for r in rows if r[name+'_applicable']]
                entry[name+'_detection']=dict(eligible_n=len(eligible),success_n=sum(r[name+'_detected'] for r in eligible),fail_n=sum(not r[name+'_detected'] for r in eligible),reference_failure_n=sum(r[name+'_status']=='reference_detection_failure' for r in eligible))
            summary[stage+'_'+arm]=entry
        for arm in ['T3M','T3','B','T1','T2']:
            a={r['identity']:r for r in records if r['arm']=='C'};b={r['identity']:r for r in records if r['arm']==arm}
            for metric in metrics:
                ids=[i for i in a if a[i].get(metric) is not None and b[i].get(metric) is not None]
                values=np.array([a[i][metric]-b[i][metric] for i in ids]);oriented=values if 'margin' in metric else -values
                stats=summarize(values,rng);n=len(values);tol=config['stats']['tie_tolerance']
                rec=dict(stage=stage,comparison='C-'+arm,metric=metric,difference='C minus comparator',higher_better='margin' in metric,**stats,paired_ids=ids,planned_n=len(a),missing_pair_n=len(a)-n,wins=int(sum(oriented>tol)),ties=int(sum(abs(oriented)<=tol)),losses=int(sum(oriented < -tol)))
                rec['win_tie_loss_proportions']=[rec[k]/n if n else None for k in ['wins','ties','losses']]
                if stage=='swap' and metric=='response_margin' and arm in ['T3M','T3'] and n:
                    signs=rng.choice([-1,1],(10000,n));null=(signs*values).mean(1)
                    rec['p_two_sided']=(int(sum(abs(null)>=abs(values.mean())))+1)/10001
                contrasts.append(rec)
                for k in np.argsort(oriented)[:math.ceil(n*.1)]:tails.append(dict(stage=stage,comparison='C-'+arm,metric=metric,identity=ids[k],raw_C_minus_comparator=float(values[k]),oriented_C_benefit=float(oriented[k])))
    prim=[r for r in contrasts if 'p_two_sided' in r];ordered=sorted(prim,key=lambda r:r['p_two_sided']);last=0
    for i,r in enumerate(ordered):last=max(last,min(1.,r['p_two_sided']*(len(ordered)-i)));r['p_holm']=last
    # 所有类别完整列出；类别分析及真正类别变化子集均为探索性。
    strata={}
    for stage,records in datasets.items():
        for category in sorted({r['category'] for r in records}):
            for arm in ARMS:
                rr=[r for r in records if r['category']==category and r['arm']==arm]
                strata[f'{stage}/{category}/{arm}']={m:summarize([r[m] for r in rr if r.get(m) is not None],rng) for m in ['dists','response_margin','orientation_error','period_error']}
    swap=datasets['swap'];motif={}
    for arm in ARMS:
        rr=[r for r in swap if r['arm']==arm and r.get('motif_category_change_subset')];motif[arm]=summarize([r['response_margin'] for r in rr if r.get('response_margin') is not None],rng)
    sanity=read(ROOT/'dists_sanity_check/results.json')['procedural_sanity_passed']
    trigger=bool(sanity and len(prim)==2 and all(r['CI95'][0]>0 and r['p_holm']<.05 for r in prim))
    write(ROOT/'statistics/paired_bootstrap.json',dict(protocol=config['protocol'],seed=config['stats']['seed'],iterations=10000,summary=summary,comparisons=contrasts,category_strata_exploratory=strata,motif_category_change_subset_exploratory=motif,DISTS_procedural_sanity_passed=sanity,human_review_triggered=trigger,human_ratings_n=0,development_only=True,annotation_exposed=True,old_stage1_stop_unchanged=True))
    csv_write(ROOT/'statistics/tail_cases.csv',tails)
    failures=[]
    for stage,records in datasets.items():
        for r in records:
            reasons=[]
            if r['insufficient_interior_patches']:reasons.append('insufficient_interior_patches')
            if r['mask_low_confidence']:reasons.append('low_confidence_mask_retained')
            for name in ['orientation','period']:
                if r[name+'_applicable'] and not r[name+'_detected']:reasons.append(name+':'+r[name+'_status'])
            if reasons:failures.append(dict(stage=stage,arm=r['arm'],identity=r['identity'],reasons=';'.join(reasons)))
    csv_write(ROOT/'failure_cases/index.csv',failures)
    print(json.dumps(dict(primary=prim,human_review_triggered=trigger,failures=len(failures)),allow_nan=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','calibrate','measure','statistics']);args=parser.parse_args()
    start=time.perf_counter();utc=now();cpu=time.process_time()
    torch.set_num_threads(4)
    if args.action=='freeze':
        assert not (ROOT/'metric_config.json').exists();config=make_config();write(ROOT/'metric_config.json',config)
        (ROOT/'code_commit.txt').write_text(config['source_commit']+'\n',encoding='utf-8')
    else:
        config=read(ROOT/'metric_config.json');assert sha(__file__)==config['metric_source_sha256']
        if args.action=='calibrate':calibration(config)
        if args.action=='measure':run(config)
        if args.action=='statistics':statistics(config)
    import PIL,torchvision,scipy
    ledger=ROOT/'runtime_resource_ledger.csv';rows=csv_read(ledger) if ledger.exists() else []
    rows.append(dict(action=args.action,start_utc=utc,end_utc=now(),wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,device='CPU',gpu_hours=0,cpu_threads=4,torch=torch.__version__,torchvision=torchvision.__version__,Pillow=PIL.__version__,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,pid=os.getpid(),status='completed'))
    csv_write(ledger,rows)

if __name__=='__main__':main()
