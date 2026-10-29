"""身份级配对统计与诊断图；不改模型、指标、样本和已冻结类别。"""
import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from tools.t3_p0p1 import read,write,sha,csv_write
from tools.pattern_fidelity_run import NAME,SERVER,SOURCE,frozen,reference_images,resolve
from tools.pattern_fidelity_core import PROTOCOL,spectrum,gray

ENDPOINTS=dict(gray_fft_new=-1,gray_fft_margin=1,rgb_fft_new=-1,rgb_fft_margin=1,
    dists_rgb_new=-1,dists_rgb_margin=1,dists_gray_new=-1,dists_gray_margin=1,
    color_new=-1,color_margin=1,orientation_hist_new=-1,orientation_hist_margin=1,
    orientation_angle_new=-1,period_value_new=-1,period_x_new=-1,period_y_new=-1,
    edge_f1=1,iou=1,edge_l1=-1,leak=-1,dists_gray_response=1,color_response=1)


def paired(values,key):
    values=np.asarray(values,float);n=len(values)
    if not n:return dict(n=0,delta=None,median_delta=None,ci_low=None,ci_high=None,p_raw=None,sd_difference=None)
    result=dict(n=n,delta=float(values.mean()),median_delta=float(np.median(values)),ci_low=None,ci_high=None,p_raw=None,sd_difference=float(values.std(ddof=1)) if n>1 else None)
    if n<2:return result
    seed=291042+int.from_bytes(__import__('hashlib').sha256(key.encode()).digest()[:3],'big')
    rng=np.random.default_rng(seed);bootstrap=values[rng.integers(0,n,(10000,n))].mean(1)
    lo,hi=np.quantile(bootstrap,[.025,.975]);signs=rng.choice([-1.,1.],(10000,n))
    p=(1+np.count_nonzero(abs((signs*values).mean(1))>=abs(values.mean())-1e-15))/10001
    result.update(ci_low=float(lo),ci_high=float(hi),p_raw=float(p),statistics_seed=seed)
    return result


def adjust(records):
    valid=sorted([r for r in records if r['p_raw'] is not None],key=lambda r:r['p_raw']);previous=0
    for rank,row in enumerate(valid):
        previous=max(previous,min(1.,(len(records)-rank)*row['p_raw']));row['p_holm']=previous
    for row in records:row.update(family_size=len(records),p_holm=row.get('p_holm'))


def input_sensitivity(root):
    rows=[]
    for case in read(root/'manifests/identities.json'):
        mask=np.asarray(Image.open(root/case['mask']))>127
        for kernel in [7,11,15]:
            body=cv2.erode(mask.astype('uint8'),np.ones((kernel,kernel),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            for size in [48,64,96,128]:
                candidates=sum(body[y:y+size,x:x+size].all() for y in range(0,513-size,32) for x in range(0,385-size,32))
                rows.append(dict(split=case['split'],id=case['id'],erosion_kernel=kernel,patch_size=size,candidates=int(candidates),valid_at_least4=candidates>=4,
                    use='input-only availability sensitivity; frozen64/kernel11 model scores unchanged'))
    csv_write(root/'audit/mask_patch_sensitivity.csv',rows)


def scientific_figures(root,contrasts,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    keys=['gray_fft_new','dists_gray_margin','color_margin','iou']
    fig,axes=plt.subplots(1,4,figsize=(13,3.6),constrained_layout=True)
    for ax,key in zip(axes,keys):
        rows=[r for r in contrasts if r['split']=='P2b_swap32' and r['baseline'] in ['B','T1','T2'] and r['metric']==key and r['subset']=='all_valid' and r['ci_low'] is not None]
        for y,row in enumerate(rows):
            ax.errorbar(row['delta'],y,xerr=[[row['delta']-row['ci_low']],[row['ci_high']-row['delta']]],fmt='o',color='#3464a1')
            ax.text(.02,y+.17,'n='+str(row['n'])+'; Holm p='+f"{row['p_holm']:.3f}",transform=ax.get_yaxis_transform(),fontsize=7)
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(len(rows)));ax.set_yticklabels(['T3-'+r['baseline'] for r in rows]);ax.set_title(key,fontsize=9);ax.set_xlabel('paired delta; unadjusted95% CI',fontsize=8);ax.set_ylim(-.5,2.6)
    fig.suptitle('Viewed32 exploratory data: metrics remain separate; no composite score',fontsize=11)
    fig.savefig(root/'figures/paired_effects.png',dpi=180);plt.close(fig)
    fixed=sorted(cases,key=lambda r:__import__('hashlib').sha256(('PATTERN_MASK_QA/'+r['split']+r['id']).encode()).hexdigest())[:4]
    invalid=next((r for r in cases if not r['valid_body']),None)
    if invalid and invalid not in fixed:fixed.append(invalid)
    fig,axes=plt.subplots(len(fixed),3,figsize=(9,3*len(fixed)),squeeze=False,constrained_layout=True)
    for row,case in enumerate(fixed):
        axes[row,0].imshow(Image.open(root/case['sketch']));axes[row,0].set_title(case['split']+' '+case['id'],fontsize=8)
        axes[row,1].imshow(Image.open(root/case['body']),cmap='gray');axes[row,1].set_title('fixed eroded body; n='+str(case['patch_n']),fontsize=8)
        for x,y in case['coords']:axes[row,1].add_patch(Rectangle((x,y),64,64,fill=False,edgecolor='red',lw=1))
        axes[row,2].imshow(Image.open(root/case['new']));axes[row,2].set_title('new reference; not model output',fontsize=8)
        for ax in axes[row]:ax.axis('off')
    fig.savefig(root/'figures/mask_patch_QA.png',dpi=130);plt.close(fig)
    # 展示参考的频谱与自相关，包括周期不可估计者，不挑清晰模型纹样。
    chosen=fixed[:4];fig,axes=plt.subplots(len(chosen),3,figsize=(9,2.8*len(chosen)),squeeze=False,constrained_layout=True)
    for row,case in enumerate(chosen):
        patch=reference_images(root/case['new'])[0];power=spectrum(patch);luma=gray(patch);luma=luma-luma.mean()
        ac=np.fft.fftshift(np.fft.ifft2(abs(np.fft.fft2(luma*np.outer(np.hanning(64),np.hanning(64)),s=(128,128)))**2).real)
        axes[row,0].imshow(patch);axes[row,0].set_title(case['id']+' reference patch',fontsize=8)
        axes[row,1].imshow(np.log10(power+1e-8) if power is not None else np.zeros((64,64)),cmap='magma');axes[row,1].set_title('normalized gray power spectrum',fontsize=8)
        axes[row,2].imshow(ac/(max(ac.max(),1e-20)),cmap='viridis');axes[row,2].set_title('linear2D autocorrelation; not physical scale',fontsize=8)
        for ax in axes[row]:ax.axis('off')
    fig.savefig(root/'figures/fft_autocorrelation_QA.png',dpi=140);plt.close(fig)


def panel(root,identities,split,kind,lookup,external=False):
    sources=[];arms=['T3','BF-Fashion','ControlNet+IP-Adapter'] if external else ['B','T1','T2','T3']
    cases={r['id']:r for r in read(root/'manifests/identities.json') if r['split']==split}
    manifest={(r['identity_id'],r['model_id']):r for r in read(root/'manifests/outputs.json') if r['split']==split}
    width=(3+len(arms)*2)*140;canvas=Image.new('RGB',(width,250*len(identities)),'white');draw=ImageDraw.Draw(canvas)
    for index,identity in enumerate(identities):
        case=cases[identity];y=index*250
        draw.text((5,y+3),kind+' | '+split+' | ID '+identity,fill='black')
        items=[('Sketch',root/case['sketch']),('Old ref',root/case['old']),('New ref',root/case['new'])]
        for arm in arms:
            if (identity,arm) not in manifest:continue
            m=manifest[identity,arm]
            for role,condition in [('old','original'),('new','swapped')]:items.append((arm.replace('ControlNet+IP-Adapter','CN+IPA')+' '+role,Path(m['output_'+role+'_path'])))
        for col,(label,path) in enumerate(items):
            draw.text((col*140+3,y+23),label,fill='black');im=ImageOps.contain(Image.open(path).convert('RGB'),(134,190))
            canvas.paste(im,(col*140+(140-im.width)//2,y+43+(190-im.height)//2))
            sources.append(dict(panel=kind,id=identity,split=split,label=label,source=str(path),sha256=sha(path),selection='input hash' if kind=='ordinary_hash4' else 'result-selected diagnostic candidate; actual visual interpretation required'))
    dest=root/'figures'/('external_' if external else '')/(kind+'.png');dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
    return sources


def analyze(project):
    root=project/'output_eval'/NAME
    plan=dict(endpoints=ENDPOINTS,primary='gray_fft_new restricted to frozen primary_eligible',
        statistical_unit='identity; new/swapped output; old output only used for response difference and structure safety',
        main_primary_family='3 comparisons',main_secondary_family='3 comparisons x22 endpoints, jointly Holm; frequency distances only reference-regular classes; period errors additionally require reliable peaks',
        category_family='all8 reference categories x3 baselines x22 endpoints, jointly Holm; n<2 no p',
        structure_safety='descriptive paired effects; inherited IoU.010 EdgeF1.020 Leak.005 margins; six original/swapped endpoints Bonferroni one-sided95%; no new confirmation claim',
        planning='precision grid only; no arbitrary minimum meaningful difference; no new generation or training automatically',
        model_score_read_before_plan=False)
    frozen(root/'protocol/analysis_plan.json',plan)
    input_sensitivity(root)
    records=[]
    for path in sorted((root/'metrics/per_image').rglob('*.json')):records.extend(read(path))
    assert len(records)==1472,('INCOMPLETE_METRICS',len(records))
    identities={(r['split'],r['id']):r for r in read(root/'manifests/identities.json')}
    annotation={r['reference_sha256']:r['category'] for r in read(root/'pattern_labels/reference_pattern_labels.json')}
    for row in records:
        case=identities[row['split'],row['id']]
        # 原始JSON保留描述子计算值；适用性只由冻结参考类别决定，非规则纹样表中记N/A。
        for key in ['gray_fft','rgb_fft']:
            for role in ['old','new']:
                if annotation[case[role+'_sha256']] not in PROTOCOL['periodicity']['classes']:row[key+'_'+role]=None
            a=row.get(key+'_old');b=row.get(key+'_new');row[key+'_margin']=a-b if a is not None and b is not None else None
    index={(r['split'],r['id'],r['model'],r['condition']):r for r in records}
    for row in records:
        if row['condition']=='swapped':
            old=index[row['split'],row['id'],row['model'],'original']
            for key in ['dists_gray','color']:
                a=row.get(key+'_margin');b=old.get(key+'_margin');row[key+'_response']=a-b if a is not None and b is not None else None
    csv_write(root/'metrics/combined_per_image.csv',records)
    for name,keys in [('color_metrics',['color_','color_response']),('dists_metrics',['dists_']),('orientation_metrics',['orientation_']),('periodicity_metrics',['period_','gray_fft_','rgb_fft_'])]:
        csv_write(root/'metrics'/(name+'.csv'),[{k:v for k,v in r.items() if k in ['split','id','model','condition','category','patch_n','valid_body','invalid_reason','primary_eligible'] or any(k.startswith(prefix) for prefix in keys)} for r in records])
    csv_write(root/'metrics/stsim_metrics.csv',[dict(split=r['split'],id=r['id'],model=r['model'],condition=r['condition'],stsim_new=None,stsim_old=None,stsim_margin=None,valid_stsim=False,invalid_reason='no verified2013-equivalent licensed implementation; optional metric not substituted with SSIM') for r in records])
    csv_write(root/'metrics/motif_metrics.csv',[dict(split=r['split'],id=r['id'],model=r['model'],condition=r['condition'],valid_motif_metric=False,invalid_reason='no validated classifier or OCR; no manufactured motif identity score') for r in records])
    means=[]
    for split in ['P2b_swap32','viewed128']:
        arms=sorted({r['model'] for r in records if r['split']==split})
        for arm in arms:
            group=[r for r in records if r['split']==split and r['model']==arm and r['condition']=='swapped']
            for metric in ENDPOINTS:
                values=[r[metric] for r in group if r.get(metric) is not None]
                means.append(dict(split=split,model=arm,metric=metric,n=len(values),planned_n=len(group),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None))
    csv_write(root/'statistics/model_means.csv',means)
    contrasts=[];stratified=[]
    new=[r for r in records if r['condition']=='swapped']
    for split,baselines in [('P2b_swap32',['B','T1','T2','C','BF-Fashion','ControlNet+IP-Adapter']),('viewed128',['C','T3-no-body','T3-no-rest'])]:
        cases={r['id']:r for r in new if r['split']==split and r['model']=='T3'}
        for baseline in baselines:
            comp={r['id']:r for r in new if r['split']==split and r['model']==baseline}
            for metric,direction in ENDPOINTS.items():
                for subset in ['all_valid','primary_eligible'] if metric=='gray_fft_new' else ['all_valid']:
                    pairs=[(cases[i],comp[i]) for i in cases.keys()&comp.keys() if cases[i].get(metric) is not None and comp[i].get(metric) is not None and (subset!='primary_eligible' or cases[i]['primary_eligible'])]
                    contrasts.append(dict(split=split,baseline=baseline,comparison='T3-'+baseline,metric=metric,favorable_direction=direction,subset=subset,
                        **paired([a[metric]-b[metric] for a,b in pairs],split+baseline+metric+subset)))
                if split=='P2b_swap32' and baseline in ['B','T1','T2']:
                    for category in PROTOCOL['category_rules']:
                        pairs=[(cases[i],comp[i]) for i in cases.keys()&comp.keys() if cases[i]['category']==category and cases[i].get(metric) is not None and comp[i].get(metric) is not None]
                        stratified.append(dict(split=split,baseline=baseline,metric=metric,category=category,favorable_direction=direction,**paired([a[metric]-b[metric] for a,b in pairs],baseline+metric+category)))
    families={}
    for row in contrasts:
        if row['split']=='P2b_swap32' and row['baseline'] in ['B','T1','T2']:family='primary3' if row['subset']=='primary_eligible' else 'secondary66'
        elif row['baseline'] in ['BF-Fashion','ControlNet+IP-Adapter']:family='external2_'+row['metric']+'_'+row['subset']
        elif row['baseline'].startswith('T3-no'):family='ablation2_'+row['metric']+'_'+row['subset']
        else:family=row['split']+'_C_'+row['metric']+'_'+row['subset']
        row['family']=family;families.setdefault(family,[]).append(row)
    for group in families.values():adjust(group)
    adjust(stratified)
    csv_write(root/'statistics/paired_statistics.csv',contrasts);csv_write(root/'statistics/category_stratified_statistics.csv',stratified)
    # 所有无效记录都保留；不能把patch不足、非适用类别当作生成失败。
    invalid=[]
    for r in records:
        for metric in ENDPOINTS:
            if r.get(metric) is None:invalid.append(dict(split=r['split'],id=r['id'],model=r['model'],condition=r['condition'],metric=metric,
                reason='response difference is stored only on swapped identity record' if metric.endswith('_response') and r['condition']=='original' else r.get('invalid_reason') or 'reference class/energy/direction confidence/period stability not applicable; see frozen rules'))
    csv_write(root/'manifests/metric_invalid_records.csv',invalid)
    # 精度场景仅用于规划；h不是事后定义的实际意义门槛。
    planning=[]
    for row in contrasts:
        if row['split']=='P2b_swap32' and row['baseline'] in ['B','T1','T2'] and row['metric'] in ['gray_fft_new','dists_gray_margin'] and row['subset']=='all_valid' and row['n']>1:
            for halfwidth in [.01,.02,.05]:
                planning.append(dict(baseline=row['baseline'],metric=row['metric'],observed_sd=row['sd_difference'],precision_halfwidth=halfwidth,
                    valid_fraction=row['n']/32,approx_valid_identities=math.ceil((1.959964*row['sd_difference']/halfwidth)**2),
                    approximate_screened_identities=math.ceil((1.959964*row['sd_difference']/halfwidth)**2/(row['n']/32)),
                    note='normal approximation; precision scenario, not meaningful effect or authorized confirmation sample; per-class coverage still required'))
    csv_write(root/'statistics/confirmation_precision_scenarios.csv',planning)
    # 结构非劣保护，沿用历史预设幅度；这里仍为已查看数据的探索性诊断。
    safety=[]
    for split,baseline in [('P2b_swap32',b) for b in ['B','T1','T2']]+[('viewed128','C')]:
        for condition in ['original','swapped']:
            for metric,margin,sign in [('iou',.010,1),('edge_f1',.020,1),('leak',.005,-1)]:
                pairs=[(a,index[split,a['id'],baseline,condition]) for a in records if a['split']==split and a['model']=='T3' and a['condition']==condition]
                values=np.array([(a[metric]-b[metric])*sign for a,b in pairs if a.get(metric) is not None and b.get(metric) is not None])
                rng=np.random.default_rng(291042);boots=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
                lower=float(np.quantile(boots,.05/6));safety.append(dict(split=split,baseline=baseline,condition=condition,metric=metric,n=len(values),oriented_mean=float(values.mean()),noninferiority_margin=margin,simultaneous_lower=lower,passed=lower>-margin,scope='exploratory inherited bound; not new confirmation'))
    csv_write(root/'statistics/structure_safety.csv',safety)
    provenance=[];main={r['id']:r for r in new if r['split']=='P2b_swap32' and r['model']=='T3'}
    stable=lambda i:__import__('hashlib').sha256(('PATTERN_FIDELITY_291042/'+i).encode()).hexdigest()
    ordinary=sorted(main,key=stable)[:4]
    provenance+=panel(root,ordinary,'P2b_swap32','ordinary_hash4',index)
    provenance+=panel(root,ordinary,'P2b_swap32','ordinary_hash4',index,True)
    def delta(i,key):
        a=main[i].get(key);b=index['P2b_swap32',i,'B','swapped'].get(key)
        return a-b if a is not None and b is not None else None
    eligible=[i for i in main if delta(i,'gray_fft_new') is not None]
    selections={
        'frequency_improvement_candidate':sorted([i for i in eligible if delta(i,'gray_fft_new')<0],key=lambda i:delta(i,'gray_fft_new'))[:3],
        'color_response_pattern_failure_candidate':sorted([i for i in main if main[i].get('color_margin') is not None and main[i]['color_margin']>0 and main[i].get('dists_gray_margin') is not None and main[i]['dists_gray_margin']<=0 and main[i]['category'] in ['floral_irregular','text_logo','plaid_grid','geometric_repeat']],key=lambda i:-main[i]['color_margin'])[:3],
        'structure_regression_candidate':sorted([i for i in eligible if delta(i,'gray_fft_new')<0 and ((delta(i,'iou') or 0)<-.01 or (delta(i,'leak') or 0)>.005)],key=lambda i:delta(i,'iou'))[:3]}
    for name,ids in selections.items():
        if ids:provenance+=panel(root,ids,'P2b_swap32',name,index)
    csv_write(root/'figures_manifest.csv',provenance)
    scientific_figures(root,contrasts,read(root/'manifests/identities.json'))
    write(root/'report/analysis_summary.json',dict(complete=True,rows=len(records),output_pairs=736,new_generation=0,new_training=0,
        release=read(root/'protocol/evaluation_release.json'),primary_results=[r for r in contrasts if r['family']=='primary3'],
        diagnostic_selections=selections,body_valid=dict(P2b_swap32=27,viewed128=105),
        inference='all previously viewed; primary main endpoint not estimable if frozen reference period gate has zero eligible identities; no independent success claim',
        STSIM='N/A; no verified equivalent licensed implementation',motif='N/A; no validated detector/OCR',
        confirmation_decision='do not start training or confirmation automatically; use complementary evidence, reference-only feasibility and precision planning first'))
    print('ANALYZED',len(records),'records',len(contrasts),'paired contrasts',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=SERVER)
    analyze(parser.parse_args().project)
