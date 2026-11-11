"""复用冻结 v3 逐图指标，附加边界、频谱及检索机制诊断。"""
import gc
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_compat import load_trainable_strict, compute_losses
from tools.e2_fft_experiment import read, write, sha, setup
from tools.e2_fft_training import dataset
from tools.m1_experiment import ARMS, load_m, m_environment


def training_audit(out):
    streams=[];records={}
    for arm in ARMS:
        folder=out/'training'/arm
        complete=read(folder/'complete.json')
        assert complete['status']=='completed' and complete['updates']==2000 and complete['frozen_unchanged']
        assert sha(folder/'latest.pt')==complete['checkpoint_sha256']
        assert complete['config']['initialization_sha256']==read(out/'comparison/M0.json')['initialization_sha256']
        rows=[json.loads(s) for s in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['step'] for r in rows]==list(range(1,2001))
        assert all(np.isfinite(r['grad_norm']) and all(np.isfinite(v) for v in r['losses'].values()) for r in rows)
        streams.append([(r['tokens'],r['chain']) for r in rows])
        records[arm]=dict(checkpoint_sha256=complete['checkpoint_sha256'],updates=len(rows),
            total_seconds=sum(r['seconds'] for r in rows),peak_gib=max(r['peak_gib'] for r in rows),
            training_input_map_sha256=complete['config']['training_input_map_sha256'])
    assert streams[0]==streams[1]
    assert len({r['training_input_map_sha256'] for r in records.values()})==1
    write(out/'audit/training_completion.json',dict(passed=True,same_legacy_stream=True,arms=records))


def generate(args):
    core,out=setup(args)
    assert read(out/'audit/training_inputs_final.json')['passed']
    assert read(out/'audit/mask_inputs_final.json')['passed']
    audit=read(out/'audit/training_completion.json')
    assert audit['passed']
    arm=args.arm;complete=read(out/'training'/arm/'complete.json')
    assert sha(out/'training'/arm/'latest.pt')==complete['checkpoint_sha256']==audit['arms'][arm]['checkpoint_sha256']
    pipe,network,_=load_m(args,core,arm)
    state=torch.load(out/'training'/arm/'latest.pt',map_location='cpu',weights_only=False)
    load_trainable_strict(network,state['network']);del state
    network.eval().requires_grad_(False)
    pipe.unet.half();pipe.vae.half();network.adapter.half()
    pipe.image_encoder.to(args.device);pipe.bf_texture_conditioner.to(args.device)
    spec=read(out/'strict_pairs.json');records=[]
    with torch.inference_mode():
        for pair_index,pair in enumerate(spec['pairs']):
            for condition in ['old','new']:
                key='original_reference' if condition=='old' else 'swapped_reference'
                sketch=Path(args.data)/pair['sketch'];reference=Path(args.data)/pair[key]
                assert sha(sketch)==pair['hashes']['sketch'] and sha(reference)==pair['hashes'][key]
                path=out/'outputs'/arm/pair['case_id']/(condition+'.png')
                if path.exists():
                    meta=read(path.with_suffix('.json'))
                    assert meta['output_sha256']==sha(path) and meta['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert meta['manifest_sha256']==sha(out/'strict_pairs.json')
                    records.append(meta);continue
                modules=[] if arm=='M0' else [getattr(network.adapter.fusion,'s2t_patch_attn%d'%i) for i in range(2,6)]
                capture=arm=='M1' and pair_index<4 and condition=='new'
                for m in modules:m.capture=capture
                network.adapter.condition(core.tensor(sketch).to(args.device).half(),core.tensor(reference).to(args.device).half())
                if capture:
                    from eval.eval_utils import prepare_evaluation_masks
                    mask=prepare_evaluation_masks((384,512),sketch_path=str(sketch),mask_policy='sketch_only')['garment']
                    summaries=[];arrays={}
                    for scale,m in enumerate(modules):
                        d=dict(m.diagnostic);arrays['scale%d'%scale]=d.pop('attention')
                        qh,qw=d['query_hw']
                        d['query_mask_occupancy']=[bool(mask[min(511,int((y+.5)/qh*512)),min(383,int((x+.5)/qw*384))]) for x,y in d['query_xy']]
                        summaries.append(d);m.capture=False
                    folder=out/'diagnostics/evaluation'/pair['case_id'];folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/'attention.npz',**arrays)
                    write(folder/'result.json',dict(case_id=pair['case_id'],condition=condition,scales=summaries,
                        mask_analysis_only=True,GT_used_in_condition=False,checkpoint_sha256=complete['checkpoint_sha256']))
                latent=torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                latent_hash=core.digest(latent)
                image=pipe(ref_image=core.tensor(sketch).to(args.device).half(),prompt=pair['caption'],
                    texture_clip_image=None,texture_embeds=None,null_prompt='',negative_prompt=' worst quality, low quality',
                    width=384,height=512,num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']),latents=latent,num_inference_steps=50)[0]
                path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                meta=dict(arm=arm,condition=condition,case_id=pair['case_id'],index=pair['index'],seed=pair['seed'],
                    caption=pair['caption'],initial_latent_sha256=latent_hash,output_sha256=sha(path),
                    checkpoint_sha256=complete['checkpoint_sha256'],input_sketch_sha256=sha(sketch),
                    input_reference_sha256=sha(reference),manifest_sha256=sha(out/'strict_pairs.json'),GT_used_in_inference=False)
                write(path.with_suffix('.json'),meta);records.append(meta);network.adapter.clear()
                print('M_GENERATED',arm,pair['case_id'],condition,flush=True)
    write(out/'generation'/(arm+'.json'),dict(status='completed',outputs=len(records),environment=m_environment()))


def spectral_metrics(image,reference,coordinates,pattern):
    from tools.pattern_fidelity_core import descriptors,compare,nearest
    generated=descriptors([image[y:y+64,x:x+64] for x,y in coordinates])
    reference=descriptors([reference[y:y+64,x:x+64] for y in np.linspace(0,448,4).round().astype(int)
                           for x in np.linspace(0,320,4).round().astype(int)])
    scores=compare(generated,reference,regular=True,directional=True)
    def peak(d):
        power=d['gray_fft']
        if power is None or power.max()<5*np.median(power):return None
        y,x=np.unravel_index(power.argmax(),power.shape)
        return np.array([x-32,y-32],float)/64
    distance=lambda a,b:float(min(np.linalg.norm(a-b),np.linalg.norm(a+b)))
    scores['fft_peak_distance']=nearest([peak(d) for d in generated],[peak(d) for d in reference],distance)
    return dict(fft_peak_distance=scores['fft_peak_distance'],gray_fft_js=scores['gray_fft'],
        autocorrelation_period_log2_error=scores['period_value'],orientation_error_degrees=scores['orientation_angle'],
        generated_period=scores['period_gen_value'],reference_period=scores['period_ref_value'],
        generated_patches=len(generated),reference_patches=len(reference))


def panels(args,pairs,order):
    out=Path(args.out)
    for rank,pair in enumerate(pairs):
        canvas=Image.new('RGB',(7*192,284),'white');draw=ImageDraw.Draw(canvas)
        files=[Path(args.data)/pair[k] for k in ['sketch','original_reference','swapped_reference']]
        files += [out/'outputs'/arm/pair['case_id']/(condition+'.png') for arm in ARMS for condition in ['old','new']]
        labels=['sketch','old reference','new reference','M0 old','M0 new','M1 old','M1 new']
        for i,(path,label) in enumerate(zip(files,labels)):
            draw.text((i*192+4,4),label,fill='black')
            with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((192,256)),(i*192,24))
        folder=out/'metrics/panels';folder.mkdir(parents=True,exist_ok=True)
        canvas.save(folder/(pair['case_id']+'.jpg'),quality=92)
    rows=[]
    for case_id,delta in order[:4]:
        rows.append(dict(case_id=case_id,primary_delta=delta,file='metrics/panels/'+case_id+'.jpg'))
    write(out/'metrics/worst_cases.json',rows)


def evaluate(args):
    from tools import e2_b_screen_evaluation as legacy
    from tools.bf_t3_three_arm_report import paired
    from tools.bf_t3_ablation_report import holm
    from tools.e2_followup_data import rgb
    from tools.e2_a_evaluation import boundary_f1
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask
    legacy.evaluate(args,arms=ARMS,audit=training_audit)
    out=Path(args.out);pairs=read(out/'strict_pairs.json')['pairs']
    rows=read(out/'metrics/per_image.json');lookup={p['case_id']:p for p in pairs}
    for row in rows:
        pair=lookup[row['case_id']];path=out/'outputs'/row['arm']/pair['case_id']/(row['condition']+'.png')
        with Image.open(path) as im:
            im=im.convert('RGB');image=np.asarray(im)
            foreground=estimate_foreground_mask(im,(384,512))
        mask=prepare_evaluation_masks((384,512),sketch_path=str(Path(args.data)/pair['sketch']),mask_policy='sketch_only')['garment']
        row['boundary_f1_r2']=boundary_f1(foreground,mask,2)
        key='original_reference' if row['condition']=='old' else 'swapped_reference'
        reference=np.asarray(rgb(Path(args.data)/pair[key],(384,512)))
        row['spectral']=spectral_metrics(image,reference,pair['patch_coordinates'],row['pattern'])
    write(out/'metrics/per_image.json',rows)
    images={(r['arm'],r['case_id'],r['condition']):r for r in rows}
    protections=[];spectra=[]
    directions=dict(edge_f1=1,iou=1,edge_l1=-1,leak=-1,tcf_lab=-1,boundary_f1_r2=1)
    margins=read(out/'protocol.json')['inherited_exploratory_noninferiority']
    for condition in ['old','new']:
        for metric,direction in dict(directions,**({'lpips':-1} if condition=='old' else {})).items():
            delta=[images['M1',p['case_id'],condition][metric]-images['M0',p['case_id'],condition][metric]
                for p in pairs if all(images[a,p['case_id'],condition][metric] is not None for a in ARMS)]
            result=paired(delta);margin=margins.get(metric,0.)
            lower=None if result['ci95'] is None else min(direction*x for x in result['ci95'])
            protections.append(dict(condition=condition,metric=metric,better_sign=direction,missing=32-len(delta),
                exploratory_margin=margin,protection_pass=lower is not None and lower>=-margin,**result))
        for metric in ['fft_peak_distance','gray_fft_js','autocorrelation_period_log2_error','orientation_error_degrees']:
            for pattern in ['all']+sorted({r['pattern'] for r in rows}):
                chosen=[p for p in pairs if pattern=='all' or p['qualification']['old_pattern']==pattern]
                delta=[]
                valid={a:0 for a in ARMS}
                for p in chosen:
                    values=[images[a,p['case_id'],condition]['spectral'][metric] for a in ARMS]
                    finite=[v is not None and bool(np.isfinite(v)) for v in values]
                    for arm,eligible in zip(ARMS,finite):
                        valid[arm]+=int(eligible)
                    if all(finite):
                        delta.append(values[1]-values[0])
                spectra.append(dict(condition=condition,pattern=pattern,metric=metric,planned_n=len(chosen),
                    valid_by_arm=valid,missing_pairs=len(chosen)-len(delta),**paired(delta)))
    holm(protections)
    write(out/'metrics/protection_statistics.json',protections)
    write(out/'metrics/spectral_statistics.json',spectra)
    responses=read(out/'metrics/responses.json');response={(r['arm'],r['case_id']):r for r in responses}
    stratified=[]
    for pattern in sorted({r['pattern'] for r in responses}):
        chosen=[p for p in pairs if p['qualification']['old_pattern']==pattern]
        for metric in ['gray_strict_response_change','rgb_strict_response_change','gray_tolerant_response_change','rgb_tolerant_response_change']:
            values=[response['M1',p['case_id']][metric]-response['M0',p['case_id']][metric] for p in chosen]
            stratified.append(dict(pattern=pattern,metric=metric,**paired(values)))
    write(out/'metrics/stratified_statistics.json',stratified)
    primary=next(r for r in read(out/'metrics/paired_statistics.json') if r['family']=='primary')
    evidence=primary['ci95'][0]>0 and primary['p']<.05
    safe=all(r['protection_pass'] for r in protections)
    harmful=any(r['ci95'] is not None and max(r['better_sign']*v for v in r['ci95']) < -r['exploratory_margin'] for r in protections)
    decision='effective_screening' if evidence and safe else 'harmful_screening' if harmful else 'no_supported_overall_improvement'
    write(out/'metrics/decision.json',dict(decision=decision,texture_evidence=evidence,protection_pass=safe,
        primary=primary,predefined_before_outputs=True,single_seed_exploratory=True,
        no_significant_harm_is_not_proof_of_noninferiority=True))
    order=sorted([(p['case_id'],response['M1',p['case_id']]['gray_strict_response_change']-
        response['M0',p['case_id']]['gray_strict_response_change']) for p in pairs],key=lambda r:r[1])
    panels(args,pairs,order)
    write(out/'metrics/extended_version.json',dict(environment=m_environment(),
        spectral_source_sha256=sha(Path(__file__).with_name('pattern_fidelity_core.py')),
        spectral='unchanged Hann/linear-autocorrelation/structure-tensor descriptors; extra signed FFT peak distance cycles/pixel',
        period='absolute log2 apparent image-period ratio, no physical-scale interpretation',
        missing='pairwise complete; all planned/valid/failure denominators retained',
        directional_proxy_caveat='one dominant direction may not summarize plaid or dot patterns'))
    print('M_EVALUATE_PASS',decision,primary,flush=True)


def diagnose(args):
    from train_GAM_texture_joint import VGGGramStyleLoss
    from tools.e2_time_gradient_common import gradient_pair
    args.action='calibrate';core,out=setup(args);args.action='diagnose'
    records=[]
    for stage in ['initial','trained']:
        pipe,network,checked=load_m(args,core,'M1')
        if stage=='trained':
            state=torch.load(out/'training/M1/latest.pt',map_location='cpu',weights_only=False)
            load_trainable_strict(network,state['network']);del state
        network.train();data=dataset(pipe,core)
        style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
        watched=[(n,p) for n,p in network.named_parameters() if '.s2t_patch_attn' in n and n.endswith('.weight')
            and any(part in n for part in ['q_proj','patch_embed','kv_proj','out_proj'])]
        modules=[getattr(network.adapter.fusion,'s2t_patch_attn%d'%i) for i in range(2,6)]
        for m in modules:m.capture=True
        for selection in read(out/'split.json')['smoke'][:4]:
            index=selection['index'];batch={k:v[None].to(args.device) for k,v in data[index].items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                total,raw,token,captured,_=compute_losses(pipe,network,batch,
                    torch.Generator(device=args.device).manual_seed(42000),style,core,audit_mode=True)
            parameters=[p for n,p in watched]
            first=torch.autograd.grad(raw['mse'],parameters,retain_graph=True,allow_unused=True)
            reference=.1*raw['color']+.1*raw['body']
            second=torch.autograd.grad(reference,parameters,retain_graph=True,allow_unused=True)
            full=torch.autograd.grad(total,parameters,allow_unused=True)
            gradients={n:dict(mse_vs_reference=gradient_pair(a,b),total_norm=None if c is None else float(c.float().norm()))
                for (n,p),a,b,c in zip(watched,first,second,full)}
            folder=out/'diagnostics'/stage/str(index);folder.mkdir(parents=True,exist_ok=True)
            summaries=[];arrays={}
            sketch=((batch['vae_sketch'][0].detach().float().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')
            texture=((batch['texture_image'][0].detach().float().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8')
            for scale,m in enumerate(modules):
                d=dict(m.diagnostic);a=d.pop('attention');arrays['scale%d'%scale]=a
                mask=torch.nn.functional.adaptive_avg_pool2d(batch['garment_mask'].float(),d['query_hw'])[0,0]
                d['query_mask_occupancy']=[float(mask[y,x]) for x,y in d['query_xy']]
                summaries.append(d)
                # 九个固定 query 各自显示平均头检索热图和 top5 位置。
                panel=Image.new('RGB',(9*128,360),'white');draw=ImageDraw.Draw(panel)
                kh,kw=d['key_hw'];qh,qw=d['query_hw']
                for j,(x,y) in enumerate(d['query_xy']):
                    sim=Image.fromarray(sketch).resize((128,171));sd=ImageDraw.Draw(sim)
                    sx,sy=int((x+.5)/qw*128),int((y+.5)/qh*171)
                    sd.ellipse((sx-4,sy-4,sx+4,sy+4),fill='red');panel.paste(sim,(j*128,18))
                    ref=Image.fromarray(texture).resize((128,171));rd=ImageDraw.Draw(ref)
                    avg=a[0,:,j].mean(0);top=np.argsort(avg)[-5:][::-1]
                    for rank,k in enumerate(top):
                        ky,kx=divmod(int(k),kw);cx,cy=(kx+.5)/kw*128,(ky+.5)/kh*171
                        rd.rectangle((cx-4,cy-4,cx+4,cy+4),outline='red' if rank==0 else 'yellow',width=2)
                    panel.paste(ref,(j*128,189));draw.text((j*128+2,2),'Q%d mask %.2f'%(j,d['query_mask_occupancy'][j]),fill='black')
                panel.save(folder/('retrieval_scale%d.jpg'%scale),quality=92)
            np.savez_compressed(folder/'attention.npz',**arrays)
            record=dict(stage=stage,index=index,token=token,scales=summaries,gradients=gradients,
                reference_objective='.1 RGB+.1 body; unchanged E2 contains no FFT supervision',
                mask_analysis_only=True,initial_zero_qkv_gradients_expected=stage=='initial',initialization=checked)
            write(folder/'result.json',record);records.append(record)
            network.adapter.clear();del batch,raw,captured,total,first,second,full,reference
            print('M_DIAGNOSE',stage,index,flush=True)
        network.adapter.unbind();del network,pipe,data,style;gc.collect();torch.cuda.empty_cache()
    write(out/'diagnostics/result.json',dict(passed=True,records=records,environment=m_environment()))


def bundle(args):
    out=Path(args.out)
    assert read(out/'audit/training_completion.json')['passed']
    assert read(out/'diagnostics/result.json')['passed']
    assert len(read(out/'metrics/per_image.json'))==128
    files=sorted({p for prefix in ['inputs','audit','calibration','smoke','comparison','training','generation','outputs',
        'metrics','logs','tests','regression','diagnostics'] for p in (out/prefix).rglob('*') if p.is_file()
        and p.suffix in ['.json','.jsonl','.png','.jpg','.txt','.out','.err','.csv','.npz']})
    files += [out/n for n in ['strict_pairs.json','split.json','protocol.json']]
    write(out/'bundle_manifest.json',{p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with zipfile.ZipFile(out/'review_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'bundle_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    write(out/'bundle_info.json',dict(bytes=(out/'review_results.zip').stat().st_size,sha256=sha(out/'review_results.zip'),files=len(files)))
    print('M_BUNDLE',sha(out/'review_results.zip'),flush=True)


def status(args):
    out=Path(args.out)
    for arm in ARMS:
        log=out/'training'/arm/'training.jsonl'
        rows=[] if not log.exists() else log.read_text(encoding='utf-8').splitlines()
        if rows:
            last=json.loads(rows[-1]);print(arm,'step',last['step'],'seconds',round(last['seconds'],2),'GiB',round(last['peak_gib'],2))
        else:print(arm,'not started')
    for name in ['smoke/result.json','audit/training_completion.json','generation/M0.json','generation/M1.json',
                 'diagnostics/result.json','metrics/decision.json','bundle_info.json']:
        print(name,(read(out/name).get('decision') or 'done') if (out/name).exists() else 'pending')
