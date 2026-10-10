"""先完成 S0/S1；科学门槛不通过即结项，通过后继续已授权的后续阶段。"""
import csv
import os
import shutil
import time
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tools.e37_protocol import *
from tools.e37_runtime import Generator,FixedInputs,predict,decode_x0,digest
from tools.e37_proxy import tensor_image,masks,components,combined,gradient,association


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def preview(image,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    im=(image.detach().float().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(im).save(p)


def smoke(g,data):
    adapter=g.load_adapter(800);rows=read(OUT/'splits/dev32.json')[:2];checks=[]
    for row in rows:
        sid=row['id'];g.bridge.reset();off=[];off_eps=[]
        for rep in range(2):
            r,_,eps=g.generate(row,'A0_E5_OFF',OUT/'s0/images'/('off%d_%s.png'%(rep,sid)),capture_first=True)
            off.append(r);off_eps.append({k:digest(v) for k,v in eps.items()});del eps
        old=read(E36/'p1/images/A0_E5_OFF'/(sid+'.json'))
        keys=['png_sha256','final_latent_sha256','initial_latent_sha256']
        assert all(off[0][k]==off[1][k]==old[k] for k in keys)
        assert off_eps[0]==off_eps[1]
        g.bridge.reset()
        on,_,eps=g.generate(row,'B1_CONV',OUT/'s0/images'/('on_'+sid+'.png'),capture_first=True)
        on_eps={k:digest(v) for k,v in eps.items()};del eps
        assert on['png_sha256']==sha(E36/'p1/images/B1_CONV'/(sid+'.png'))
        assert on_eps['unconditional']==off_eps[0]['unconditional']
        assert on_eps['conditional']!=off_eps[0]['conditional'] and on['png_sha256']!=off[0]['png_sha256']
        trace=g.bridge.report();write(OUT/'s0'/('trace_'+sid+'.json'),trace)
        applied=[r for r in trace['trace'] if r['applied']]
        assert applied and all(r['branch']=='conditional' and r['index']==0 and r['shape']==[1,640,32,24] for r in applied)
        checks.append(dict(id=sid,off_repeats_exact=True,off_matches_original=True,on_actuates=True,
            on_matches_E36=True,unconditional_off=True,eps_hashes=dict(off=off_eps[0],on=on_eps)))
    # 固定名单顺序中的首个可信 mask；不换身份池或随机重抽。
    probe=read(OUT/'splits/train_probe96.json')
    valid_index=next(i for i,row in enumerate(probe) if masks(row)[1]['valid'])
    rec=next(r for r in read(OUT/'splits/fixed_noisy_records.json') if r['index']==valid_index and r['range']==0)
    state={n:digest(p) for n,p in adapter.named_parameters()}
    args,kw,noise,sketch,gt,ref,audit=data.sample(rec)
    prediction=predict(g,args,kw,sketch);rgb=decode_x0(g,data,args,prediction)
    region,mask_audit=masks(read(OUT/'splits/train_probe96.json')[rec['index']])
    assert mask_audit['valid'],'first fixed sample needs valid proxy mask; do not resample'
    losses=components(rgb,gt,ref,region)
    ge=gradient(F.mse_loss(prediction.float(),noise.float()),adapter)
    gr=gradient(losses['mom']+losses['refgram']+losses['bg']+losses['bd'],adapter,retain_graph=False)
    assert ge.norm()>0 and gr.norm()>0 and rgb.requires_grad
    preview(rgb,OUT/'s0/images/x0_preview.png');preview(gt,OUT/'s0/images/gt_preprocessed.png');preview(ref,OUT/'s0/images/reference_plain_resize.png')
    offset=0;layer_grads={}
    for name,p in adapter.named_parameters():
        layer_grads[name]=float(gr[offset:offset+p.numel()].norm());offset+=p.numel()
    assert layer_grads['head.weight']>0 and layer_grads['head.bias']>0
    write(OUT/'s0/report.json',dict(pass_s0=True,checks=checks,training_record=audit,mask=mask_audit,
        epsilon_grad_norm=float(ge.norm()),image_grad_norm=float(gr.norm()),image_grad_by_parameter=layer_grads,vae_decode_has_grad=rgb.requires_grad,
        adapter_unchanged={n:digest(p) for n,p in adapter.named_parameters()}==state,
        optimizer_updates=0,vae_decode_range=[float(rgb.min()),float(rgb.max())],
        normalization='GT original dataset [-1,1] -> [0,1]; R original BICUBIC plain_resize',
        original_E36_metrics_not_rejudged=True))
    decision(s0_pass=True,phase='S1')
    print('E37 S0 PASS',flush=True)


def calibrate(g,data,started):
    adapter=g.load_adapter(800);rows=read(OUT/'splits/train_probe96.json')
    mask_audits=[];regions={}
    for row in rows:
        region,audit=masks(row);mask_audits.append(audit)
        if audit['valid']:regions[row['id']]=region
    write(OUT/'s1/train_mask_audit.json',mask_audits)
    if len(regions)/96<.8:
        return None,dict(reason='train_probe_mask_coverage',valid_n=len(regions),fixed_n=96)
    records=[];raw=[];saturated=0
    for i,record in enumerate(read(OUT/'splits/fixed_noisy_records.json')):
        if time.perf_counter()-started>7200:raise TimeoutError('S0-S3 2 GPU-hour diagnostics budget reached')
        args,kw,noise,sketch,gt,ref,audit=data.sample(record)
        prediction=predict(g,args,kw,sketch);rgb=decode_x0(g,data,args,prediction)
        finite=bool(torch.isfinite(rgb).all());fraction=float(((rgb<0)|(rgb>1)).float().mean())
        saturated+=int(fraction>.5)
        item=dict(**audit,finite=finite,out_of_range_fraction=fraction,
            rgb_min=float(rgb.min()),rgb_max=float(rgb.max()),mask_valid=record['id'] in regions)
        if not finite:
            write(OUT/'s1/x0_audit.json',records+[item]);return None,dict(reason='nonfinite_x0')
        if record['id'] in regions:
            c=components(rgb,gt,ref,regions[record['id']])
            bg=gradient(c['bg'],adapter);bd=gradient(c['bd'],adapter,retain_graph=False)
            item.update({k:float(v.detach()) for k,v in c.items()})
            item.update(bg_grad_norm=float(bg.norm()),bd_grad_norm=float(bd.norm()))
            raw.append(item)
        if i<4:preview(rgb,OUT/'s1/x0_previews'/('%s_t%d.png'%(record['id'],record['t'])))
        records.append(item);write(OUT/'s1/x0_audit.json',records)
        if (i+1)%16==0:print('E37 S1 calibration',i+1,'/192',flush=True)
        del prediction,rgb
    if saturated>96:return None,dict(reason='majority_x0_severely_saturated',saturated_n=saturated,fixed_n=192)
    med=lambda k:float(np.median([r[k] for r in raw]))
    denominators=[med(k) for k in ['bd_grad_norm','tex','refgram']]
    if min(denominators)<=0:return None,dict(reason='zero_calibration_denominator',values=denominators)
    weights=dict(lambda_bd=med('bg_grad_norm')/denominators[0],lambda_tex=med('color')/denominators[1],lambda_R=med('mom')/denominators[2])
    csv_write(OUT/'s1/calibration.csv',raw)
    result=dict(weights=weights,valid_n=len(raw),fixed_n=192,identities=len(regions),checkpoint=800,
        calibrated_before_dev=True,optimizer_updates=0,medians={k:med(k) for k in
        ['bg','bd','color','tex','mom','refgram','bg_grad_norm','bd_grad_norm']})
    path=OUT/'s1/calibration_locked.json'
    if path.exists():assert read(path)==result
    write(path,result)
    return weights,dict(reason=None,severely_saturated_n=saturated,fixed_n=192)


@torch.no_grad()
def proxy_on_full_rgb(weights):
    rows=read(OUT/'splits/dev32.json');records=[];audits=[];hashes={}
    for row in rows:
        region,audit=masks(row);audits.append(audit)
        gt=tensor_image(row['gt']);ref=tensor_image(row['reference'],bicubic=True)
        for arm in ['A0_E5_OFF','B1_CONV']:
            path=E36/'p1/images'/arm/(row['id']+'.png');meta=read(path.with_suffix('.json'))
            assert sha(path)==meta['png_sha256'];hashes[str(path)]=sha(path)
            target=OUT/'s1/full_rgb/images'/arm/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);shutil.copy2(path.with_suffix('.json'),target.with_suffix('.json'))
            if audit['valid']:
                c=components(tensor_image(path),gt,ref,region)
                losses={k:float(v) for k,v in combined(c,weights).items()}
                natural=components(gt,gt,ref,region)
                natural_gap=float(natural['mom']+weights['lambda_R']*natural['refgram'])
            else:losses=dict(LS=None,LAGT=None,LAR=None);natural_gap=None
            records.append(dict(id=row['id'],arm=arm,mask_valid=audit['valid'],**losses,GT_reference_natural_gap=natural_gap))
    write(OUT/'s1/dev_mask_audit.json',audits);write(OUT/'s1/reused_rgb_hashes.json',hashes)
    write(OUT/'s1/dev_proxy_records.json',records);csv_write(OUT/'s1/dev_proxy_records.csv',records)
    metrics=read(E36/'p1/metrics.json');index={(r['arm'],r['id']):r for r in metrics}
    pindex={(r['arm'],r['id']):r for r in records};results={};pairs=[]
    comparisons=[('LS','struct_iou',-1),('LS','leak_colored_frac',1),('LAGT','lpips_gt',1),
                 ('LAR','tcf_lab_delta',1),('LAR','tpf_gram_l1',1)]
    for loss,metric,sign in comparisons:
        x=[];y=[]
        for row in rows:
            sid=row['id'];b=pindex['A0_E5_OFF',sid];a=pindex['B1_CONV',sid]
            mb=index['A0_E5_OFF',sid];ma=index['B1_CONV',sid]
            if b[loss] is None or a[loss] is None or mb[metric] is None or ma[metric] is None:continue
            dx=a[loss]-b[loss];dy=sign*(ma[metric]-mb[metric]);x.append(dx);y.append(dy)
            pairs.append(dict(id=sid,loss=loss,metric=metric,delta_loss=dx,aligned_delta_metric=dy))
        result=association(x,y);result.update(fixed_n=32,invalid_n=32-len(x),metric_direction=sign)
        results[loss+'__'+metric]=result
    write(OUT/'s1/paired_bootstrap.json',results);csv_write(OUT/'s1/paired_changes.csv',pairs)
    # 对所有有效身份画固定散点，不删除负向/最差身份。
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(20,4))
    for ax,(loss,metric,_) in zip(axes,comparisons):
        rr=[p for p in pairs if p['loss']==loss and p['metric']==metric]
        ax.scatter([r['delta_loss'] for r in rr],[r['aligned_delta_metric'] for r in rr],s=12)
        for r in rr:
            if r['id']=='160097987':ax.annotate(r['id'],(r['delta_loss'],r['aligned_delta_metric']),fontsize=6)
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
        ax.set(xlabel='delta '+loss,ylabel='aligned delta '+metric,title='valid %d/32'%len(rr))
    fig.tight_layout();fig.savefig(OUT/'s1/proxy_full_rgb_scatter.png',dpi=140);plt.close(fig)
    coverage=sum(a['valid'] for a in audits)/32
    structure=any(results['LS__'+key]['pass_gate'] for key in ['struct_iou','leak_colored_frac'])
    appearance=results['LAGT__lpips_gt']['pass_gate']
    reference=any(results['LAR__'+key]['pass_gate'] for key in ['tcf_lab_delta','tpf_gram_l1'])
    passed=coverage==28/32 and structure and appearance and reference and not any(r['inconclusive'] for r in results.values())
    from tools.e35_freeu_stats import panels
    folder=OUT/'s1/full_rgb';arms=['A0_E5_OFF','B1_CONV']
    write(folder/'manifest.json',rows);write(folder/'params.json',{a:dict(parameters=143166 if a=='B1_CONV' else 0) for a in arms})
    write(folder/'metrics.json',[r for r in metrics if r['arm'] in arms])
    panels(folder,rows,arms,tag='all32')
    worst=sorted([dict(id=row['id'],delta=index['B1_CONV',row['id']]['struct_iou']-index['A0_E5_OFF',row['id']]['struct_iou'])
        for row in rows if index['B1_CONV',row['id']]['struct_iou'] is not None],key=lambda r:r['delta'])[:10]
    write(folder/'worst10.json',worst)
    bad={r['id'] for r in worst}|{'160097987'}
    panels(folder,[r for r in rows if r['id'] in bad],arms,tag='worst_and_E36_failure')
    report=dict(pass_s1=bool(passed),results=results,coverage=coverage,
        group_gates=dict(structure=structure,GT_appearance=appearance,reference_appearance=reference),
        E36_failure_identity_proxy=[r for r in records if r['id']=='160097987'],
        reuse='E36 full DDIM50 E5 and B1-800 RGB; SHA verified',
        no_single_step_eps_substitution=True,weights=weights)
    write(OUT/'s1/report.json',report)
    return report


def run():
    from tools.e34_sarr_protocol import seed_all
    torch.set_num_threads(2);seed_all(42);prepare();started=time.perf_counter()
    job_id=os.environ.get('SLURM_JOB_ID','manual')
    write(OUT/'audit'/('implementation_'+job_id+'.json'),dict(commit=commit(),
        files={str(p):sha(p) for p in Path('tools').glob('e37_*.py')}))
    attempts_path=OUT/'resource_attempts.json'
    attempts=read(attempts_path) if attempts_path.exists() else []
    previous_seconds=sum(r['seconds'] for r in attempts)
    assert previous_seconds<7200,'Cumulative diagnostics budget exhausted; report bottleneck before continuation'
    g=Generator();data=FixedInputs(g)
    try:
        if not (OUT/'s0/report.json').exists():smoke(g,data)
        else:assert read(OUT/'s0/report.json')['pass_s0']
        if (OUT/'s1/calibration_locked.json').exists():
            weights=read(OUT/'s1/calibration_locked.json')['weights'];x0=read(OUT/'s1/x0_gate.json')
        else:
            weights,x0=calibrate(g,data,started-previous_seconds);write(OUT/'s1/x0_gate.json',x0)
        if weights is None:
            decision(s1_pass=False,phase='complete',stop_reason='S1_fail: '+x0['reason']);return
        report=proxy_on_full_rgb(weights)
        if not report['pass_s1']:
            inconclusive=any(r['inconclusive'] for r in report['results'].values())
            decision(s1_pass=False,phase='complete',stop_reason='S1_inconclusive' if inconclusive else 'S1_fail: proxy does not meet frozen correlation/sign gates')
        else:
            decision(s1_pass=True,phase='S2_ready',stop_reason=None)
            print('S1 PASS: continue authorized S2; experiment is not concluded',flush=True)
    finally:
        g.close();verify_inputs()
        runtime=dict(seconds=time.perf_counter()-started,gpu_hours=(time.perf_counter()-started)/3600,
            job_id=os.environ.get('SLURM_JOB_ID'),gpu_count=1,optimizer_updates=0,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,commit=commit())
        write(OUT/'resource_runtime.json',runtime);attempts.append(runtime)
        write(attempts_path,attempts)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        decision(phase='engineering_error',engineering_error=repr(exc));raise
