"""身份配对统计；显著性、纹样可读性和机制门禁分别呈现。"""
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tools.e53_protocol import *
from tools.e53_texture import score
from tools.e52_data import signed_distance
from tools.e44_metrics import external_contour,contour_f1

RULES=dict(primary_G='mean within-identity Pearson boundary reference-following rho G1 vs G0; undefined rho retained NA',primary_C='ORACLE minus ORIGINAL F1',primary_D='EARLY minus LATE F1; expected negative for early damage',family=3,
    C_support='requires F1 rescue beyond SUPPORT, UV texture rescue, SHIFT texture and F1 damage, all paired CI in expected directions',
    G_support='requires primary positive rho plus reduced reference-following amplitude after restoring V0 under G1 Oracle',
    D_support='requires significant early damage at audited equal energy and Oracle texture correctness above chance and above WRONG_TEXTURE',
    no_effect='non-significance is INCONCLUSIVE, never proof of NOT_PRIMARY',scope='controlled renderer geometric-UV routing and fixed Writer; not mathematical purity of Writer features or unique universal root cause')

def paired(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],np.float64)
    if not len(x):return dict(n=0,mean=None,ci95=None,p_two_sided=None)
    rng=np.random.default_rng(32042);draws=np.concatenate([rng.choice(x,(500,len(x)),replace=True).mean(1) for _ in range(20)])
    p=min(1.,2*min((np.count_nonzero(draws<=0)+1)/10001,(np.count_nonzero(draws>=0)+1)/10001))
    return dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),p_two_sided=float(p),method='identity paired percentile bootstrap; bootstrap sign-tail p')

def holm(tests):
    ordered=sorted((v['p_two_sided'] if v['p_two_sided'] is not None else 1.,k) for k,v in tests.items());running=0.;out={}
    for i,(p,k) in enumerate(ordered):running=max(running,min(1.,(3-i)*p));out[k]=running
    return out

def response(reference,output,region):
    a=reference[region].astype('float64');b=output[region].astype('float64')
    amplitude=float(np.dot(a,b)/max(float(np.dot(a,a)),1e-12)) if len(a) else None
    rho=float(np.corrcoef(a,b)[0,1]) if len(a)>8 and np.std(a)>1e-8 and np.std(b)>1e-8 else None
    return dict(rho=rho,amplitude=amplitude,n_boundary_points=len(a))

def panels(rows,arms,stage):
    folder=OUT/'figures'/stage/'outputs';folder.mkdir(parents=True,exist_ok=True)
    for ai in range(0,len(arms),6):
        aa=arms[ai:ai+6]
        for start in range(0,len(rows),4):
            page=rows[start:start+4];canvas=Image.new('RGB',((len(aa)+1)*128,len(page)*196),'white');draw=ImageDraw.Draw(canvas)
            for y,row in enumerate(page):
                for x,(label,path) in enumerate([('Sketch',Path(row['sketch']))]+[(a,image_path(stage,a,row['id'])) for a in aa]):
                    draw.text((x*128+2,y*196+2),row['id']+' '+label,fill='black');canvas.paste(Image.open(path).convert('RGB').resize((128,171)),(x*128,y*196+23))
            canvas.save(folder/('arms%02d_ids%03d.png'%(ai,start)))

def evaluate(stage):
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage
    from tools.e44_b1_mask import overlay
    rows=read(OUT/'splits'/(stage+'.json'));arms=PILOT_ARMS if stage=='pilot' else LOCK_ARMS;records=[];directions=[];energy_checks=[]
    folder=OUT/'metrics'/stage;folder.mkdir(parents=True,exist_ok=True);freeze(OUT/'configs/statistics.json',RULES)
    for row in rows:
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only');garment=bundle['garment'];assert garment is not None
        geometry=np.load(row['geometry']);true_mask=geometry['target_labels']>0;mask_by_arm={};initial=set();base_refs=set()
        for arm in arms:
            path=image_path(stage,arm,row['id']);meta=read(path.with_suffix('.json'));trace=read(path.with_name(row['id']+'_trace.json'));initial.add(meta['initial_sha256']);base_refs.add(meta['base_reference_sha256'])
            assert len(trace['steps'])==50 and not meta['gt_used_in_inference']
            image=Image.open(path).convert('RGB');fg=estimate_foreground_mask(image,SIZE)
            assert fg is not None,'FOREGROUND_EXTRACTION_NA'
            mask_by_arm[arm]=fg;out=folder/'foreground'/arm;out.mkdir(parents=True,exist_ok=True)
            Image.fromarray(fg.astype('uint8')*255).save(out/(row['id']+'.png'));overlay(image,fg,garment).save(out/(row['id']+'_overlay.png'))
            distance=signed_distance(fg)-signed_distance(true_mask);boundary=external_contour(true_mask)
            rec=dict(id=row['id'],arm=arm,seed=42,f1_3px=contour_f1(fg,garment),iou=float((fg&garment).sum()/max((fg|garment).sum(),1)),leak=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac'),f1_renderer_mask=contour_f1(fg,true_mask),boundary_displacement_abs=float(np.abs(distance[boundary]).mean()),boundary_displacement_signed=float(distance[boundary].mean()),base_reference_sha256=meta['base_reference_sha256'],generated_png_sha256=sha(path),gt_used_in_inference=False,**score(image,row))
            rec['Q_input_identity']=int(rec['readable'] and rec['predicted_texture']==ARMS[arm]['texture']);records.append(rec)
            budget=sum(x*x for x in trace['energy_targets']) if trace['energy_targets'] is not None else 0.
            error=abs(trace['total_injection_energy']/budget-1) if budget else 0.
            if arm!='E5':assert budget>0 and error<1e-4
            energy_checks.append(dict(id=row['id'],arm=arm,total_energy=trace['total_injection_energy'],budget=budget,relative_error=error,max_gain=max(x['energy']['gain'] for x in trace['steps']),max_step_error=max(x['energy']['relative_error'] for x in trace['steps'])))
        assert len(initial)==1 and len(base_refs)==1
        reference=geometry['expected_shift'];region=(cv2.dilate(external_contour(true_mask).astype('uint8'),np.ones((19,19),np.uint8))>0)&(np.abs(reference)>=1)
        baseline=signed_distance(mask_by_arm['G0'])
        d=dict(id=row['id'],kind=row['kind'],sign=row['sign'],**response(reference,signed_distance(mask_by_arm['G1'])-baseline,region))
        restored=response(reference,signed_distance(mask_by_arm['G1_V0'])-baseline,region);d['V0_restored']=restored
        d['V0_response_reduction']=d['amplitude']-restored['amplitude'] if d['amplitude'] is not None and restored['amplitude'] is not None else None
        d['V_swap']=response(reference,signed_distance(mask_by_arm['V_SWAP'])-signed_distance(mask_by_arm['G0_ORIGINAL']),region)
        d['K_swap']=response(reference,signed_distance(mask_by_arm['K_SWAP'])-signed_distance(mask_by_arm['G0_ORIGINAL']),region)
        directions.append(d);print('E53 EVAL',stage,row['id'],flush=True)
    write(folder/'per_image.json',records);write(folder/'geometry_response.json',directions);write(folder/'energy_audit.json',energy_checks)
    with (folder/'per_image.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    byid={row['id']:{r['arm']:r for r in records if r['id']==row['id']} for row in rows}
    keys=['f1_3px','iou','leak','Q_tex_corr','Q_tex_identity','Q_tex_margin','boundary_displacement_abs','boundary_displacement_signed']
    means={arm:{key:float(np.mean(v)) if v else None for key in keys for v in [[r[key] for r in records if r['arm']==arm and r[key] is not None]]} for arm in arms}
    def delta(a,b,key):return paired([d[a][key]-d[b][key] for d in byid.values() if d[a][key] is not None and d[b][key] is not None])
    contrasts={a:{key:delta(a,'E5',key) for key in keys} for a in arms if a!='E5'}
    c=dict(rescue={k:delta('ORACLE','ORIGINAL',k) for k in keys},wrong={k:delta('SHIFT','ORACLE',k) for k in keys},beyond_support={k:delta('ORACLE','SUPPORT',k) for k in keys},correct_vs_wrong_texture={k:delta('ORACLE','WRONG_TEXTURE',k) for k in keys})
    primary=dict(G=paired([d['rho'] for d in directions]),C=c['rescue']['f1_3px'],D=delta('EARLY','LATE','f1_3px'));adjusted=holm(primary)
    v_reduce=paired([d['V0_response_reduction'] for d in directions]);positive=lambda x:x['ci95'] is not None and x['ci95'][0]>0;negative=lambda x:x['ci95'] is not None and x['ci95'][1]<0
    texture_above_chance=paired([d['ORACLE']['Q_tex_identity']-.125 for d in byid.values()])
    clean_texture=positive(texture_above_chance) and positive(c['correct_vs_wrong_texture']['Q_tex_identity']) and positive(paired([d['ORACLE']['Q_tex_corr'] for d in byid.values()]))
    verdicts=dict(H_G='SUPPORTED_IN_CONTROLLED_GEOMETRY_RANGE' if adjusted['G']<.05 and positive(primary['G']) and positive(v_reduce) else 'INCONCLUSIVE',H_C='SUPPORTED_FOR_GEOMETRIC_UV_ROUTING' if adjusted['C']<.05 and positive(primary['C']) and positive(c['rescue']['Q_tex_identity']) and negative(c['wrong']['Q_tex_identity']) and negative(c['wrong']['f1_3px']) and positive(c['beyond_support']['f1_3px']) else 'INCONCLUSIVE',H_D='SUPPORTED_CONTROLLED_TIME_EFFECT' if adjusted['D']<.05 and negative(primary['D']) and clean_texture else 'INCONCLUSIVE')
    summary=dict(stage=stage,n_identities=len(rows),arms=arms,engineering_complete=True,means=means,contrasts_vs_E5=contrasts,primary=primary,holm_p=adjusted,correspondence=c,V0_geometry_response_reduction=v_reduce,oracle_texture_above_chance=texture_above_chance,clean_texture_readout_gate=clean_texture,verdicts=verdicts,semantic_feature_oracle_proven=False,MODEL_STATUS='NOT_AN_UPGRADE_TRIAL',rules=RULES,max_total_energy_error=max(x['relative_error'] for x in energy_checks),max_gain=max(x['max_gain'] for x in energy_checks),training_updates=0)
    write(folder/'summary.json',summary);panels(rows,arms,stage)
    decision(**({stage:'COMPLETE',**verdicts} if stage=='lock' else {stage:'COMPLETE'}));print('E53 SUMMARY',json_safe(summary['verdicts']),flush=True)
    return summary
