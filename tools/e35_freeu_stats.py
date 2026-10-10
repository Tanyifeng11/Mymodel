"""统一最终RGB评测、身份配对统计和预注册停止规则；不在确认集调参。"""
import argparse
import csv
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.e35_freeu_protocol import *


def finite(value):
    return value is not None and isinstance(value,(float,int)) and math.isfinite(value)

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value

def csv_write(path,records):
    keys=['id','arm','mask_valid']+METRICS
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)

@torch.inference_mode()
def evaluate(split,folder=None):
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import (compute_structure_preservation,compute_texture_color_fidelity,
        compute_texture_leakage,compute_texture_pattern_fidelity,compute_clip_i_values)
    from tools.e33tmoc_appearance_eval import frozen_lpips,model_hash
    folder=Path(folder) if folder is not None else OUT/('a_screen' if split=='dev32' else 'a_confirm');rows=read(folder/'manifest.json')
    arms=read(folder/'params.json');records=[]
    lpips,info=frozen_lpips();lpips=lpips.cuda()
    info['state_sha256']=model_hash(lpips)
    def tensor(im):return torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    masks={r['id']:prepare_evaluation_masks(SIZE,sketch_path=r['sketch'],mask_policy='sketch_only') for r in rows}
    for arm in arms:
        current=[]
        for row in rows:
            sid=row['id'];path=folder/'images'/arm/(sid+'.png');assert path.exists()
            bundle=masks[sid];stats=bundle['stats']
            valid=bool(stats.get('mask_source')=='sketch_flood_fill' and not stats.get('mask_low_confidence',True))
            gen=Image.open(path).convert('RGB');gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            record=dict(id=sid,arm=arm,mask_source='sketch_only',mask_valid=valid,mask_stats=stats,
                lpips_gt=float(lpips(tensor(gen),tensor(gt)).mean()),warnings=[])
            # 结构Edge独立于mask；IoU与区域指标只在可信Sketch-only mask下计算。
            record.update(compute_structure_preservation(str(path),row['sketch'],mask_bundle=bundle))
            if valid:
                record.update(compute_texture_color_fidelity(str(path),row['reference'],mask_bundle=bundle))
                record.update(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle))
                record.update(compute_texture_pattern_fidelity(str(path),row['reference'],mask_bundle=bundle))
            else:
                record.update({k:None for k in MASK_METRICS})
                record['warnings'].append('unreliable sketch mask: retain identity, mask-derived metrics NA; no GT fallback')
            current.append(clean(record))
        gen_paths=[str(folder/'images'/arm/(r['id']+'.png')) for r in rows]
        for key,target in [('clip_texture','reference'),('clip_real','gt')]:
            values=compute_clip_i_values(gen_paths,[r[target] for r in rows],batch_size=4,
                device='cuda',model_name='models/clip')
            assert len(values)==len(rows)
            for record,value in zip(current,values):record[key]=float(value)
        records.extend(current)
        print('EVALUATED',split,arm,len(current),flush=True)
    write(folder/'metrics.json',records);csv_write(folder/'metrics.csv',records)
    write(folder/'metric_protocol.json',dict(mask_source='sketch_only',mask_valid='sketch_flood_fill and not low confidence',
        mask_valid_n=sum(r['mask_valid'] for r in records if r['arm']=='A0_E5_OFF'),
        fixed_n=len(rows),gt_fallback_calls=0,lpips=info,clip_model='models/clip',
        lab_backend='existing eval.metrics._rgb_to_lab, fixed installed backend',
        edge_definition='existing struct_edge_f1, includes sketch internal lines; not outer-contour-only F1',
        iou_definition='existing estimated generated foreground vs reliable sketch-only mask',
        distribution_metrics='not computed at32/96',failure_denominator=len(rows),
        failures_n=0,caption_unchanged=True))
    del lpips
    summarize(folder)
    panels(folder,rows,list(arms),tag='fixed_all')


def paired(records,arm,key):
    base={r['id']:r for r in records if r['arm']=='A0_E5_OFF'}
    return [dict(id=r['id'],delta=r[key]-base[r['id']][key]) for r in records
        if r['arm']==arm and finite(r.get(key)) and finite(base[r['id']].get(key))]

def summarize(folder):
    records=read(folder/'metrics.json');arms=list(read(folder/'params.json'));n=len(read(folder/'manifest.json'))
    summary={};deltas={}
    for arm in arms:
        group=[r for r in records if r['arm']==arm]
        summary[arm]={}
        for key in METRICS:
            vals=[r[key] for r in group if finite(r.get(key))]
            summary[arm][key]=dict(mean=float(np.mean(vals)) if vals else None,valid_n=len(vals),invalid_n=n-len(vals),fixed_n=n)
        times=[read((folder/'images'/arm/(r['id']+'.png')).with_suffix('.json')) for r in group]
        summary[arm]['seconds']=dict(mean=float(np.mean([t['seconds'] for t in times])),median=float(np.median([t['seconds'] for t in times])))
        summary[arm]['peak_memory_gib']=max(t['peak_memory_gib'] for t in times)
        if arm!='A0_E5_OFF':
            deltas[arm]={key:dict(**bootstrap([r['delta'] for r in paired(records,arm,key)]),
                invalid_n=n-len(paired(records,arm,key)),fixed_n=n) for key in METRICS}
    write(folder/'summary.json',summary);write(folder/'paired_deltas.json',deltas)
    # 身份散点图只展示有效配对，分母和缺失数在配套JSON中明确记录。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for arm in deltas:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        base={r['id']:r for r in records if r['arm']=='A0_E5_OFF'}
        for ax,key in zip(axes,['struct_iou','lpips_gt','tcf_lab_delta']):
            group=[r for r in records if r['arm']==arm and finite(r.get(key)) and finite(base[r['id']].get(key))]
            x=[base[r['id']][key] for r in group];y=[r[key] for r in group]
            ax.scatter(x,y,s=14,alpha=.7)
            if x:lo=min(x+y);hi=max(x+y);ax.plot([lo,hi],[lo,hi],'k--',linewidth=.7)
            ax.set(xlabel='E5',ylabel=arm,title='%s valid=%d/%d'%(key,len(group),n))
        fig.tight_layout();fig.savefig(folder/(arm+'_paired.png'),dpi=140);plt.close(fig)
    return deltas


def safety(deltas):
    limits=CONFIG['thresholds']
    def below(key,value):return deltas[key]['mean'] is not None and deltas[key]['mean']<=value
    def above(key,value):return deltas[key]['mean'] is not None and deltas[key]['mean']>=value
    return dict(edge=above('struct_edge_f1',-limits['edge_drop']),
        lab=below('tcf_lab_delta',limits['lab_rise']),clip=above('clip_texture',-limits['clip_drop']),
        leak=below('leak_colored_frac',limits['leak_rise']),lpips=below('lpips_gt',limits['lpips_rise']))


def select():
    folder=OUT/'a_screen';records=read(folder/'metrics.json');deltas=read(folder/'paired_deltas.json')
    checks={arm:safety(value) for arm,value in deltas.items()}
    outliers={}
    for arm in deltas:
        bad={}
        for key,sign,limit in [('struct_iou',-1,.15),('struct_edge_f1',-1,.15),('leak_colored_frac',1,.10)]:
            for r in paired(records,arm,key):
                if sign*r['delta']>limit:bad.setdefault(r['id'],[]).append(key)
        outliers[arm]=bad;checks[arm]['no_catastrophic_proxy']=not bad
    eligible=[arm for arm in deltas if all(checks[arm].values()) and deltas[arm]['struct_iou']['mean'] is not None]
    chosen=None;tied=[]
    if eligible:
        chosen=max(eligible,key=lambda arm:deltas[arm]['struct_iou']['mean'])
        best={r['id']:r['struct_iou'] for r in records if r['arm']==chosen}
        for arm in eligible:
            gap=[r['struct_iou']-best[r['id']] for r in records if r['arm']==arm and finite(r.get('struct_iou')) and finite(best[r['id']])]
            ci=bootstrap(gap)['ci95']
            if ci and ci[0]<=0<=ci[1]:tied.append(arm)
        chosen=min(tied,key=lambda arm:(read(folder/'summary.json')[arm]['lpips_gt']['mean'],list(ARMS).index(arm)))
    result=dict(selected=chosen,config=ARMS[chosen] if chosen else None,safety=checks,eligible=eligible,tied_on_iou=tied,
        selection_split='dev32',exploratory_only=True,git_commit=commit(),locked=True,
        catastrophic_proxies=outliers)
    path=folder/'selected_candidate.json'
    if path.exists():assert read(path)==result,'选参冻结后禁止更改'
    write(path,result)
    decision(screen_n=32,screen_selected=chosen,a_result='pending_confirm' if chosen else 'A_stop',
        stop_reason=None if chosen else 'all three prespecified candidates violate screen safety limits',
        confirm_n=0,b_authorized=False)
    print('SELECTED',chosen,checks,flush=True)
    return result


def confirm():
    folder=OUT/'a_confirm';chosen=read(OUT/'a_screen/selected_candidate.json')['selected']
    deltas=read(folder/'paired_deltas.json')[chosen];limits=CONFIG['thresholds'];checks=safety(deltas)
    primary=deltas['struct_iou']
    checks['iou_gain']=primary['mean'] is not None and primary['mean']>=limits['iou_gain']
    checks['iou_ci_positive']=primary['ci95'] is not None and primary['ci95'][0]>0
    passed=all(checks.values())
    # Trade-off必须有可重复LPIPS改善及可接受的其他安全项；不会因此自动启动B训练。
    perceptual=deltas['lpips_gt'];ci=perceptual['ci95']
    tradeoff=not passed and ci is not None and ci[1]<0 and all(safety(deltas).values())
    status='A_pass' if passed else ('A_tradeoff' if tradeoff else 'A_fail')
    write(folder/'decision.json',dict(status=status,checks=checks,deltas=deltas,n=96,
        primary_valid_n=primary['n'],primary_missing_n=96-primary['n'],no_retuning=True))
    decision(a_result=status,confirm_n=96,confirm_delta_sketch_iou=primary['mean'],confirm_ci_sketch_iou=primary['ci95'],
        confirm_delta_contour_f1=deltas['struct_edge_f1']['mean'],confirm_delta_lpips=perceptual['mean'],
        confirm_delta_texture_clip=deltas['clip_texture']['mean'],confirm_mask_valid_n=primary['n'],
        b_authorized=passed,b_result='pending_novelty_review' if passed else 'not_run',
        official_validation_result='not_run',novel_method_claim_supported=False)
    worst=sorted(paired(read(folder/'metrics.json'),chosen,'struct_iou'),key=lambda r:r['delta'])[:10]
    write(folder/'worst10.json',worst)
    lookup={r['id']:r for r in read(folder/'manifest.json')}
    panels(folder,[lookup[r['id']] for r in worst],['A0_E5_OFF',chosen],tag='worst10')
    print('CONFIRM',status,checks,json.dumps(deltas),flush=True)


def panels(folder,rows,arms,tag):
    target=folder/'paired_images'/tag;target.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),4):
        canvas=Image.new('RGB',((3+len(arms))*160,4*246),'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(rows[start:start+4]):
            y=i*246;draw.text((3,y+2),row['id'],fill='black')
            entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
            entries.extend((arm,str(folder/'images'/arm/(row['id']+'.png'))) for arm in arms)
            for j,(label,path) in enumerate(entries):
                draw.text((j*160+3,y+18),label,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((160,213),Image.Resampling.BILINEAR),(j*160,y+32))
        canvas.save(target/('page%02d.png'%(start//4)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['screen','confirm','select']);args=p.parse_args()
    if args.action=='screen':evaluate('dev32');select()
    elif args.action=='confirm':evaluate('confirm96');confirm()
    else:select()
