"""冻结官方验证配对的五臂v3评测；与旧v1/v2分开，全部模型重算。"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.e2_compat import load_trainable_strict
from tools.e2_b_screen_training import ARMS
from tools.e2_fft_experiment import read, write, sha, setup, load, environment
from tools.e2_followup_data import rgb


def generation_metadata(args):
    out=Path(args.out);complete=read(out/'training'/args.arm/'complete.json')
    assert complete['status']=='completed' and complete['updates']==2000
    checkpoint=out/'training'/args.arm/'latest.pt'
    assert sha(checkpoint)==complete['checkpoint_sha256']
    return complete,checkpoint


def generate(args):
    core,out=setup(args);spec=read(out/'strict_pairs.json')
    assert spec['sufficient_for_planned_screening']
    complete,checkpoint=generation_metadata(args)
    pipe,network,_=load(args,core)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    load_trainable_strict(network,state['network']);del state
    network.eval().requires_grad_(False)
    pipe.unet.half();pipe.vae.half();network.adapter.half()
    pipe.image_encoder.to(args.device);pipe.bf_texture_conditioner.to(args.device)
    records=[]
    with torch.inference_mode():
        for pair in spec['pairs']:
            for condition in ['old','new']:
                sketch=Path(args.data)/pair['sketch']
                key='original_reference' if condition=='old' else 'swapped_reference'
                reference=Path(args.data)/pair[key]
                assert sha(sketch)==pair['hashes']['sketch'] and sha(reference)==pair['hashes'][key]
                path=out/'outputs'/args.arm/pair['case_id']/(condition+'.png')
                if path.exists():
                    meta=read(path.with_suffix('.json'))
                    assert meta['output_sha256']==sha(path) and meta['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert meta['manifest_sha256']==sha(out/'strict_pairs.json')
                    records.append(meta);continue
                network.adapter.condition(core.tensor(sketch).to(args.device).half(),core.tensor(reference).to(args.device).half())
                latent=torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                latent_hash=core.digest(latent)
                image=pipe(ref_image=core.tensor(sketch).to(args.device).half(),prompt=pair['caption'],
                    texture_clip_image=None,texture_embeds=None,null_prompt='',negative_prompt=' worst quality, low quality',
                    width=384,height=512,num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']),latents=latent,num_inference_steps=50)[0]
                path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                meta=dict(arm=args.arm,condition=condition,case_id=pair['case_id'],index=pair['index'],
                    seed=pair['seed'],caption=pair['caption'],initial_latent_sha256=latent_hash,
                    output_sha256=sha(path),checkpoint_sha256=complete['checkpoint_sha256'],
                    input_sketch_sha256=sha(sketch),input_reference_sha256=sha(reference),
                    manifest_sha256=sha(out/'strict_pairs.json'),GT_used_in_inference=False)
                write(path.with_suffix('.json'),meta);records.append(meta);network.adapter.clear()
                print('B_GENERATED',args.arm,pair['case_id'],condition,flush=True)
    write(out/'generation'/(args.arm+'.json'),dict(status='completed',outputs=len(records),records=records,environment=environment()))


def training_audit(out):
    reference=None;records={}
    for arm in ARMS:
        complete=read(out/'training'/arm/'complete.json')
        assert complete['updates']==2000 and complete['status']=='completed'
        assert sha(out/'training'/arm/'latest.pt')==complete['checkpoint_sha256']
        rows=[__import__('json').loads(s) for s in (out/'training'/arm/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['step'] for r in rows]==list(range(1,2001))
        assert all(np.isfinite(r['grad_norm']) and all(np.isfinite(v) for v in r['losses'].values()) for r in rows)
        stream=[r['tokens'] for r in rows]
        if reference is None:reference=stream
        else:assert stream==reference,'五臂训练旧样本/噪声/t/dropout流不同'
        records[arm]=dict(updates=2000,checkpoint_sha256=complete['checkpoint_sha256'],
            input_map_sha256=complete['config']['training_input_map_sha256'],
            valid_auxiliary_samples=sum(r['patch_counts']['valid_n'] for r in rows),
            training_seconds=sum(r['seconds'] for r in rows))
    assert len({r['input_map_sha256'] for r in records.values()})==1
    write(out/'audit/training_completion.json',dict(passed=True,same_legacy_stream=True,arms=records))


def response_comparisons(responses, pairs, arms):
    """首臂是已审计基线；A0/A5复用时不再依赖B0这个名称。"""
    from tools.bf_t3_three_arm_report import paired
    from tools.bf_t3_ablation_report import holm
    values={(r['arm'],r['index']):r for r in responses};comparisons=[]
    baseline=arms[0]
    for arm in arms[1:]:
        for mode in ['gray','rgb']:
            for scale in ['strict','tolerant']:
                metric=mode+'_'+scale+'_response_change'
                delta=[values[arm,p['index']][metric]-values[baseline,p['index']][metric] for p in pairs]
                comparisons.append(dict(arm=arm,comparison=arm+'_minus_'+baseline,metric=metric,
                    family='primary' if metric=='gray_strict_response_change' else 'secondary',**paired(delta)))
    for family in ['primary','secondary']:holm([r for r in comparisons if r['family']==family])
    return comparisons


def evaluate(args):
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_leakage, compute_texture_color_fidelity
    from tools.e2a_attr_eval import frozen_lpips
    from tools.bf_t3_three_arm_report import gray
    from tools import t3_response_run as response
    core,out=setup(args);training_audit(out)
    assert read(out/'audit/training_inputs_final.json')['passed']
    spec=read(out/'strict_pairs.json');pairs=spec['pairs']
    assert len(pairs)==32 and len({p['identity_group'] for p in pairs})==32
    response.ROOT=out;dists=response.load_dists(args.device);lpips,lpips_assets=frozen_lpips()
    references={};records=[]
    def feature(parts):
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().to(args.device)/255.
        return dists.forward_once(value)
    def ref(path,scale,is_gray):
        key=(str(path),scale,is_gray)
        if key not in references:
            size=(round(384*scale),round(512*scale));image=np.asarray(rgb(path,size))
            patches=[image[y:y+64,x:x+64] for y in np.linspace(0,size[1]-64,4).round().astype(int)
                     for x in np.linspace(0,size[0]-64,4).round().astype(int)]
            references[key]=feature([gray(p) for p in patches] if is_gray else patches)
        return references[key]
    def distance(a,b):
        # v3使用固定两组所有patch配对的均值，避免只匹配少数最接近纹元。
        return float(response.distances(dists,a,b).mean())
    with torch.inference_mode():
        for pair in pairs:
            for key,source in [('sketch','sketch'),('gt','gt'),('original_reference','original_reference'),('swapped_reference','swapped_reference')]:
                assert sha(Path(args.data)/pair[source])==pair['hashes'][key]
            masks=prepare_evaluation_masks((384,512),sketch_path=str(Path(args.data)/pair['sketch']),mask_policy='sketch_only')
            coords=pair['patch_coordinates'];assert len(coords)>=4
            latent_hashes=[]
            for arm in ARMS:
                complete=read(out/'training'/arm/'complete.json')
                for condition in ['old','new']:
                    path=out/'outputs'/arm/pair['case_id']/(condition+'.png');meta=read(path.with_suffix('.json'))
                    ref_key='original_reference' if condition=='old' else 'swapped_reference'
                    assert meta['output_sha256']==sha(path) and meta['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert meta['manifest_sha256']==sha(out/'strict_pairs.json') and not meta['GT_used_in_inference']
                    assert (meta['arm'],meta['condition'],meta['case_id'],meta['index'],meta['seed'],meta['caption'])==(
                        arm,condition,pair['case_id'],pair['index'],pair['seed'],pair['caption'])
                    assert meta['input_sketch_sha256']==pair['hashes']['sketch'] and meta['input_reference_sha256']==pair['hashes'][ref_key]
                    latent_hashes.append(meta['initial_latent_sha256'])
                    kwargs=dict(gen_path=str(path),sketch_path=str(Path(args.data)/pair['sketch']),
                        target_path=str(Path(args.data)/pair['gt']),mask_bundle=masks)
                    structure=compute_structure_preservation(**kwargs)
                    row=dict(arm=arm,index=pair['index'],case_id=pair['case_id'],identity_group=pair['identity_group'],
                        condition=condition,patch_n=len(coords),pattern=pair['qualification']['old_pattern'],
                        edge_f1=structure['edge_f1'],iou=structure['sketch_iou'],edge_l1=structure['edge_l1'],
                        leak=compute_texture_leakage(**kwargs)['leak_colored_frac'],lpips=None,
                        tcf_lab=compute_texture_color_fidelity(texture_path=str(Path(args.data)/pair[ref_key]),**kwargs)['tcf_lab_delta'],scores={})
                    if condition=='old':row['lpips']=float(lpips(core.tensor(path).to(args.device),core.tensor(Path(args.data)/pair['gt']).to(args.device)).mean())
                    row={k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in row.items()}
                    image=np.asarray(Image.open(path).convert('RGB'));parts=[image[y:y+64,x:x+64] for x,y in coords]
                    for is_gray in [False,True]:
                        f=feature([gray(p) for p in parts] if is_gray else parts)
                        old={str(s):distance(f,ref(Path(args.data)/pair['original_reference'],s,is_gray)) for s in [.75,1.,1.33]}
                        new={str(s):distance(f,ref(Path(args.data)/pair['swapped_reference'],s,is_gray)) for s in [.75,1.,1.33]}
                        row['scores']['gray' if is_gray else 'rgb']=dict(old=old,new=new,
                            strict_margin=old['1.0']-new['1.0'],tolerant_margin=min(old.values())-min(new.values()),
                            old_best_scale=min(old,key=old.get),new_best_scale=min(new,key=new.get))
                    records.append(row);print('B_EVALUATED',arm,pair['case_id'],condition,flush=True)
            assert len(set(latent_hashes))==1,'十个条件初始latent不同'
            write(out/'metrics/per_image_progress.json',records)
    lookup={(r['arm'],r['case_id'],r['condition']):r for r in records};responses=[]
    for pair in pairs:
        for arm in ARMS:
            a,b=lookup[arm,pair['case_id'],'old'],lookup[arm,pair['case_id'],'new']
            row=dict(arm=arm,index=pair['index'],identity_group=pair['identity_group'],case_id=pair['case_id'],pattern=pair['qualification']['old_pattern'])
            for mode in ['gray','rgb']:
                for scale in ['strict','tolerant']:
                    row[mode+'_'+scale+'_response_change']=b['scores'][mode][scale+'_margin']-a['scores'][mode][scale+'_margin']
            responses.append(row)
    comparisons=response_comparisons(responses,pairs,ARMS)
    write(out/'metrics/per_image.json',records);write(out/'metrics/responses.json',responses)
    write(out/'metrics/paired_statistics.json',comparisons)
    write(out/'metrics/version.json',dict(protocol=spec['protocol'],outputs=len(records),identities=32,
        manifest_sha256=sha(out/'strict_pairs.json'),primary_comparisons=4,secondary_comparisons=12,
        gray_preprocess='legacy bounded three-channel contrast normalized proxy; not pure texture ground truth',
        dists_distance='all fixed cross-patch distance mean; distinct v3, no v1/v2 number mixing',
        independent_upstream_pretraining_test=False,lpips_assets=lpips_assets,environment=environment()))
