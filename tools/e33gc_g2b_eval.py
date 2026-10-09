"""预冻结独立DFT+Hann与旧E26并列；固定分母、不可读失败，训练N=4无泛化CI。"""
import cv2,numpy as np,torch,hashlib
from PIL import Image,ImageDraw
from tools.e33gc_g2b_protocol import *
from data.e33gc_g2b_renderer import load
from tools.e33tm_metrics import Evaluator
from tools.e33tmif_metrics import pair
from tools.e33tmoc_appearance_eval import patches,measure,frozen_lpips,model_hash
from tools.e33gc_g2b_prepare import panels

def dft_block(image,box):
    a=np.asarray(image.crop(tuple(box)).convert('L'),float)/255;h,w=a.shape
    power=abs(np.fft.fft2((a-a.mean())*np.outer(np.hanning(h),np.hanning(w))))**2
    fy,fx=np.meshgrid(np.fft.fftfreq(h),np.fft.fftfreq(w),indexing='ij');radius=np.hypot(fx,fy)
    p=power*((radius>=CONFIG['dft']['low_frequency'])&(radius<=CONFIG['dft']['high_frequency']))
    z=np.sum(p*np.exp(2j*np.arctan2(fy,fx)))/max(float(p.sum()),1e-12)
    theta=(np.degrees(np.angle(z))/2+90)%180;std=float(a.std());coherence=float(abs(z))
    return dict(angle=float(theta),std=std,coherence=coherence,
        readable=std>=CONFIG['dft']['min_std'] and coherence>=CONFIG['dft']['min_axis_coherence'])

def dft_pair(images,boxes):
    blocks={arm:[dft_block(image,box) for box in boxes] for arm,image in images.items()};result={}
    for name,arm,rotation in [('r90','R90',90),('r180','R180',0)]:
        valid=[i for i in range(len(boxes)) if blocks['R0'][i]['readable'] and blocks[arm][i]['readable']]
        errors=[float(abs((blocks[arm][i]['angle']-blocks['R0'][i]['angle']+rotation+90)%180-90)) for i in valid]
        error=float(np.mean(errors)) if errors else None;readable=len(valid)>=3
        result.update({name+'_error':error,name+'_readable':readable,name+'_valid_blocks':len(valid),
            name+'_support_fraction':len(valid)/max(len(boxes),1),name+'_success':bool(readable and error<=15)})
    return dict(**result,blocks=blocks,fixed_boxes=boxes,fixed_block_count=len(boxes))

def run():
    init();torch.set_num_threads(2);cv2.setNumThreads(1)
    evaluator=Evaluator();clip_hash=model_hash(evaluator.model);lpips,lpips_init=frozen_lpips();lpips_hash=model_hash(lpips)
    identities=read(OUT/'protocol/g2b_fit_probe_ids.json');cases=[];blind=[];blind_key=[];learning=[]
    for group,rows in identities.items():
        if group not in ['fit','probe']:continue
        for row in rows:
            sid=row['id'];value=load(row,True);inner=value['inner']
            fixed=cv2.resize(inner.astype(np.float32),(48,64),interpolation=cv2.INTER_AREA)>=.95;weight=np.ones((64,48),np.float32)
            oracle={a:v for a,v in zip(ARMS,value['targets'])}
            oracle_dft=dft_pair(oracle,row['dft_boxes']);assert oracle_dft['r90_success'] and oracle_dft['r180_success'],'oracle fixed-block DFT contract invalid'
            sets={'A0_E5':OUT/'G2b_smoke/cache'/sid,'A1_step160':OUT/('G2b_train' if group=='fit' else 'G2b_probe')/'step160'/sid}
            if group=='fit':
                sets.update({'step40':OUT/'G2b_train/step40'/sid,'step80':OUT/'G2b_train/step80'/sid,
                    'constant_RF':OUT/'G2b_train/constant_RF'/sid})
            result=dict(id=sid,label=row['label'],group=group,oracle_DFT=oracle_dft,methods={})
            for method,folder in sets.items():
                if not folder.exists():continue
                images={a:Image.open(folder/(a+'_A0.png' if method=='A0_E5' else a+'.png')).convert('RGB') for a in ARMS}
                dft=dft_pair(images,row['dft_boxes']);metrics={};geometries={};target_boxes=patches(value['references'][0],inner,sid,'shared','target')
                for i,arm in enumerate(ARMS):
                    scores,geometry=evaluator.evaluate(images[arm],row['caption'],value['references'][i],Image.fromarray(value['mask']*255),value['sketch'])
                    reference=value['references'][i];source_mask=np.any(np.asarray(reference)<245,axis=2)
                    scores.update(measure(images[arm],reference,inner,source_mask,target_boxes,patches(reference,source_mask,sid,arm,'source'),lpips))
                    diff=(np.asarray(images[arm],float)-np.asarray(value['targets'][i],float))/255
                    scores['masked_charbonnier']=float(np.sqrt(diff[inner]**2+1e-6).mean())
                    baseline=Image.open(OUT/'G2b_smoke/cache'/sid/(arm+'_A0.png')).convert('RGB')
                    delta=np.abs(np.asarray(images[arm],float)-np.asarray(baseline,float))/255
                    scores['outside_MAE_vs_A0']=float(delta[~inner].mean())
                    metrics[arm]=scores;geometries[arm]=geometry
                result['methods'][method]=dict(DFT=dft,E26=pair(geometries,fixed,weight),arms=metrics)
                if method in ['A0_E5','A1_step160']:
                    token=hashlib.sha256(('E33GC-G2b/blind/'+sid+'/'+method).encode()).hexdigest()[:12]
                    blind_key.append(dict(token=token,id=sid,label=row['label'],group=group,method=method))
                    blind.append((token,row['caption'],[('ref0',value['references'][0]),('ref90',value['references'][1]),
                        ('truth0',value['targets'][0]),('truth90',value['targets'][1]),('sketch',value['sketch'])]+
                        [(arm,images[arm]) for arm in ARMS]))
                if method in ['step40','step80']:
                    learning.append((row['label']+' '+method,row['caption'],[(arm,images[arm]) for arm in ARMS]))
            cases.append(result);print('EVAL',row['label'],result['methods']['A1_step160']['DFT']['r90_success'],flush=True)
    panels(sorted(blind,key=lambda v:v[0]),OUT/'visual_audit/blind_endpoint')
    panels(learning,OUT/'visual_audit/learning_curve')
    write(OUT/'protocol/blind_method_key.json',blind_key)
    write(OUT/'G2b_eval/identity_metrics.json',cases)
    fit=[c for c in cases if c['group']=='fit'];assert len(fit)==4
    def count(key):return sum(c['methods']['A1_step160']['DFT'][key] for c in fit)
    baseline=np.mean([m['masked_charbonnier'] for c in fit for m in c['methods']['A0_E5']['arms'].values()])
    endpoint=np.mean([m['masked_charbonnier'] for c in fit for m in c['methods']['A1_step160']['arms'].values()])
    reduction=float((baseline-endpoint)/baseline*100)
    checks=dict(r90=count('r90_success')>=3,r180=count('r180_success')>=3,readable=count('r90_readable')>=3,RGB_fit=reduction>=20)
    write(OUT/'G2b_eval/numerical_gate.json',dict(checks=checks,numerical_pass=all(checks.values()),N=4,
        r90_success_n=count('r90_success'),r180_success_n=count('r180_success'),readable_n=count('r90_readable'),
        masked_charbonnier_baseline=float(baseline),masked_charbonnier_endpoint=float(endpoint),reduction_pct=reduction,
        statistical_unit='identity; three arms repeated',generalization_CI=None,AI_visual_gate_pending=True,
        human_gate=None,lpips_init=lpips_init,lpips_sha256=lpips_hash,CLIP_sha256=clip_hash))
    decision(g2b_fit_final_image_r90_n_of_4=count('r90_success'),g2b_fit_r180_n_of_4=count('r180_success'),
        g2b_fit_readable_n_of_4=count('r90_readable'),g2b_fit_masked_charbonnier_reduction_pct=reduction,
        g2b_fit_pass=False if not all(checks.values()) and identities['fit'][0].get('route')!='B' else None,
        g2b_numerical_fit_pass=all(checks.values()),next_route='two_AI_endpoint_review')
    assert clip_hash==model_hash(evaluator.model) and lpips_hash==model_hash(lpips)
    verify_frozen();bundle('endpoint')

if __name__=='__main__':run()
