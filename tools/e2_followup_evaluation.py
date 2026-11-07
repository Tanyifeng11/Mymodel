"""固定资格清单上的新v2换料诊断；旧v1指标与产物不覆盖。"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.e2_compat import load_trainable_strict
from models.e2_geometry import InteriorPatchSampler, build_geometry
from tools.e2_followup_data import SIZE, read, write, sha, freeze, rgb


def ref_path(args, pair, condition):
    if condition=='old':return Path(args.data)/pair['original_reference']
    root=Path(args.out) if pair.get('swapped_reference_root')=='out' else Path(args.data)
    return root/pair['swapped_reference']


def generate(args):
    from tools.e2_fft_experiment import setup, load, environment
    core,_=setup(args)
    parent,out=Path(args.parent),Path(args.out)
    manifest=read(out/'strict_pairs.json')
    complete=read(parent/'training'/args.arm/'complete.json')
    assert complete['updates']==2000 and complete['status']=='completed'
    checkpoint=parent/'training'/args.arm/'latest.pt'
    assert sha(checkpoint)==complete['checkpoint_sha256']
    pipe,network,_=load(args,core)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    load_trainable_strict(network,state['network']);del state
    network.eval().requires_grad_(False)
    pipe.unet.half();pipe.vae.half();network.adapter.half()
    pipe.image_encoder.to(args.device);pipe.bf_texture_conditioner.to(args.device)
    entries=[]
    with torch.inference_mode():
        for pair in manifest['pairs']:
            for condition in ['old','new']:
                sketch_path=Path(args.data)/pair['sketch']
                reference_path=ref_path(args,pair,condition)
                assert sha(sketch_path)==pair['hashes']['sketch']
                assert sha(reference_path)==pair['hashes']['original_reference' if condition=='old' else 'swapped_reference']
                dest=out/'outputs'/args.arm/pair['case_id']/(condition+'.png')
                if dest.exists():
                    meta=read(dest.with_suffix('.json'))
                    assert meta['output_sha256']==sha(dest) and meta['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert meta['manifest_sha256']==sha(out/'strict_pairs.json')
                    entries.append(meta);continue
                sketch=core.tensor(sketch_path).to(args.device).half()
                reference=core.tensor(reference_path).to(args.device).half()
                network.adapter.condition(sketch,reference)
                latent=torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                latent_hash=core.digest(latent)
                im=pipe(ref_image=sketch,prompt=pair['caption'],texture_clip_image=None,texture_embeds=None,
                    null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,
                    num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']),latents=latent,num_inference_steps=50)[0]
                dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
                meta=dict(arm=args.arm,case_id=pair['case_id'],index=pair['index'],kind=pair['kind'],condition=condition,
                    seed=pair['seed'],caption=pair['caption'],initial_latent_sha256=latent_hash,output_sha256=sha(dest),
                    input_sketch_sha256=sha(sketch_path),input_reference_sha256=sha(reference_path),
                    checkpoint_sha256=complete['checkpoint_sha256'],manifest_sha256=sha(out/'strict_pairs.json'),GT_used_in_inference=False)
                write(dest.with_suffix('.json'),meta);entries.append(meta)
                network.adapter.clear();print('GENERATED_V2',args.arm,pair['case_id'],condition,flush=True)
    write(out/('generation/%s.json'%args.arm),dict(status='completed',outputs=len(entries),records=entries,environment=environment()))


def evaluate(args):
    from tools.e2_fft_experiment import setup, environment
    from tools import t3_response_run as response
    from tools.bf_t3_three_arm_report import gray, paired
    from tools.bf_t3_ablation_report import holm
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation,compute_texture_leakage
    core,_=setup(args)
    out=Path(args.out);manifest=read(out/'strict_pairs.json');pairs=manifest['pairs']
    # 未找到合格真实参考时保留n=0；衍生控制另列，绝不伪装成真实面料配对。
    response.ROOT=out
    dists=response.load_dists(args.device)
    references={};records=[];patch_table={}

    def feature(parts):
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().to(args.device)/255.
        return dists.forward_once(value)

    def reference(path,scale,is_gray):
        key=(str(path),scale,is_gray)
        if key not in references:
            size=(round(384*scale),round(512*scale))
            image=np.array(rgb(path,size))
            parts=[image[y:y+64,x:x+64] for y in np.linspace(0,size[1]-64,4).round().astype(int)
                   for x in np.linspace(0,size[0]-64,4).round().astype(int)]
            references[key]=feature([gray(p) for p in parts] if is_gray else parts)
        return references[key]

    def distance(a,b):
        matrix=response.distances(dists,a,b)
        return float(.5*(matrix.min(0).values.mean()+matrix.min(1).values.mean()))

    with torch.inference_mode():
        for pair in pairs:
            masks=prepare_evaluation_masks(SIZE,sketch_path=str(Path(args.data)/pair['sketch']),mask_policy='sketch_only')
            geo=build_geometry(torch.from_numpy(masks['garment'].astype(np.float32))[None,None],details=False)
            coords,candidates=InteriorPatchSampler(110803+pair['index']).coordinates(geo.body[0],64,8)
            patch_table[str(pair['index'])]=dict(coordinates=coords,candidates=candidates,minimum=4,fully_inside_body=True)
            for arm in ['E2','E2_FFT']:
                complete=read(Path(args.parent)/'training'/arm/'complete.json')
                for condition in ['old','new']:
                    path=out/'outputs'/arm/pair['case_id']/(condition+'.png')
                    meta=read(path.with_suffix('.json'))
                    assert sha(path)==meta['output_sha256'] and not meta['GT_used_in_inference']
                    assert meta['checkpoint_sha256']==complete['checkpoint_sha256']
                    assert meta['manifest_sha256']==sha(out/'strict_pairs.json')
                    assert meta['input_reference_sha256']==sha(ref_path(args,pair,condition))
                    kwargs=dict(gen_path=str(path),sketch_path=str(Path(args.data)/pair['sketch']),
                                target_path=str(Path(args.data)/pair['gt']),mask_bundle=masks)
                    structure=compute_structure_preservation(**kwargs)
                    row=dict(arm=arm,case_id=pair['case_id'],index=pair['index'],kind=pair['kind'],condition=condition,
                        patch_n=len(coords),edge_f1=structure['edge_f1'],iou=structure['sketch_iou'],edge_l1=structure['edge_l1'],
                        leak=compute_texture_leakage(**kwargs)['leak_colored_frac'],scores={},metric_warnings=structure.get('metric_warnings',[]))
                    row={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
                    if len(coords)>=4:
                        image=np.asarray(Image.open(path).convert('RGB'))
                        patches=[image[y:y+64,x:x+64] for x,y in coords]
                        for is_gray in [False,True]:
                            f=feature([gray(p) for p in patches] if is_gray else patches)
                            old={str(s):distance(f,reference(ref_path(args,pair,'old'),s,is_gray)) for s in [.75,1.,1.33]}
                            new={str(s):distance(f,reference(ref_path(args,pair,'new'),s,is_gray)) for s in [.75,1.,1.33]}
                            name='gray' if is_gray else 'rgb'
                            row['scores'][name]=dict(old=old,new=new,strict_margin=old['1.0']-new['1.0'],
                                tolerant_margin=min(old.values())-min(new.values()),
                                old_best_scale=min(old,key=old.get),new_best_scale=min(new,key=new.get))
                    records.append(row)
                    print('EVALUATED_V2',arm,pair['case_id'],condition,len(coords),flush=True)
    # 全部四个生成条件应共用一个latent，不能只比较两臂new图。
    for pair in pairs:
        metas=[read(out/'outputs'/a/pair['case_id']/(c+'.json')) for a in ['E2','E2_FFT'] for c in ['old','new']]
        assert len({v['initial_latent_sha256'] for v in metas})==1
    lookup={(v['arm'],v['case_id'],v['condition']):v for v in records}
    responses=[]
    for pair in pairs:
        for arm in ['E2','E2_FFT']:
            a,b=lookup[arm,pair['case_id'],'old'],lookup[arm,pair['case_id'],'new']
            value=dict(arm=arm,index=pair['index'],case_id=pair['case_id'],kind=pair['kind'],patch_n=b['patch_n'])
            for mode in ['rgb','gray']:
                for scale in ['strict','tolerant']:
                    key=mode+'_'+scale
                    value[key+'_new_margin']=b['scores'].get(mode,{}).get(scale+'_margin')
                    value[key+'_response_change']=(b['scores'][mode][scale+'_margin']-a['scores'][mode][scale+'_margin']) if mode in b['scores'] else None
            responses.append(value)
    comparisons=[]
    for kind in ['real_reference_pair','derived_rotate90','derived_period2x']:
        family=[]
        for metric in ['gray_strict_response_change','rgb_strict_response_change','gray_tolerant_response_change','rgb_tolerant_response_change']:
            # 同服装多个控制条件先聚合，不能把参考变换/patch当独立服装。
            values=defaultdict(dict)
            for row in responses:
                if row['kind']==kind and row[metric] is not None:values[row['index']][row['arm']]=row[metric]
            differences=[v['E2_FFT']-v['E2'] for v in values.values() if set(v)=={'E2','E2_FFT'}]
            family.append(dict(kind=kind,metric=metric,comparison='E2_FFT_minus_E2',**paired(differences)))
        holm(family);comparisons+=family
    write(out/'metrics/patch_table.json',patch_table)
    write(out/'metrics/per_image.json',records)
    write(out/'metrics/responses.json',responses)
    write(out/'metrics/paired_statistics.json',comparisons)
    write(out/'metrics/version.json',dict(protocol=manifest['protocol'],manifest_sha256=sha(out/'strict_pairs.json'),
        outputs=len(records),identities=len({v['index'] for v in responses}),
        real_reference_targets=manifest['actual_real_targets'],derived_control_conditions=manifest['actual_derived_controls'],
        sampling='input sketch only, full-interior64, fixed RNG per identity, max8/min4',
        dists_gray_preprocess='legacy bounded three-channel contrast-normalized proxy; not pure texture truth',
        statistics='10000 identity-paired bootstrap/signflip seed110142; Holm4 per real/rotation/period family',
        no_new_training=True,independent_generalization_test=False,human_blind_reviewers=0,environment=environment()))
