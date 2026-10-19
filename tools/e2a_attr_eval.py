"""固定逐身份 latent/输入/评价后端，生成 RGB 终图并保存真实 LPIPS。"""
import argparse
import gc
import hashlib
import importlib.util
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
import torch
from tools.e2a_attr_common import *


def model_path(arm):
    if arm=='E1': return E1
    seed,factor=arm.split('_F')
    return checkpoint(int(seed[1:]),factor)


def image_seed(identity): return CONFIG['inference_seed']+int(hashlib.sha256(identity.encode()).hexdigest()[:7],16)


def input_rows(stage):
    if stage=='LOCK': return read(OUT/'selection/LOCK_protocol.json')['rows']
    return read(OUT/'splits'/(stage+'.json'))


def frozen_lpips():
    import lpips
    path=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-4df8aa71.pth'
    digest=sha(path); assert digest.startswith('4df8aa71')
    model=lpips.LPIPS(net='alex',pnet_rand=True)
    pretrained=torch.load(path,map_location='cpu',weights_only=False)
    model.net.load_state_dict({k:pretrained['features.'+k.split('.',1)[1]] for k in model.net.state_dict()},strict=True)
    model.eval().requires_grad_(False)
    return model.cuda(),dict(name='LPIPS v0.1 AlexNet',trunk_sha256=digest,
        calibration='installed lpips package pretrained v0.1/alex.pth',
        comparison='full generated image vs GT, both [-1,1], 384x512')


@torch.inference_mode()
def generate(stage,arm):
    folder=OUT/'evaluation'/stage/arm
    if (folder/'generation_complete.json').exists(): return
    rows=input_rows(stage); path=model_path(arm)
    if arm!='E1': assert read(path.parent.parent/'checkpoint_verification.json')['passed']
    else: assert sha(E1)==read(OUT/'checkpoint_manifest.json')[str(E1)]['sha256']
    spec=importlib.util.spec_from_file_location('e2a_historical_inference',SOURCE/'inference_IMAGGarment-1.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sd=PROJECT/'models/stable-diffusion-v1-5'
    args=SimpleNamespace(GAM_model_ckpt=str(path),texture_ckpt=str(path),base_model_path=str(sd),
        vae_model_path=str(sd/'vae'),image_encoder_path=str(PROJECT/'models/clip'),device='cuda:0',
        width=384,height=512,texture_num_tokens=16,force_texture_num_tokens_override=False,
        texture_condition_mode='token',alpha1=2.,alpha2=2.,alpha3=1.5,alpha4=1.)
    pipe,_=module.prepare(args)
    for model in [pipe.unet,pipe.reference_unet,pipe.vae,pipe.text_encoder,pipe.image_encoder,pipe.bf_texture_conditioner]:
        model.eval().requires_grad_(False)
    pipe.set_layer_group_enabled(True); pipe._setup_layer_groups(); pipe.set_progress_bar_config(disable=True)
    routing={n:dict(group=p.layer_group,text_scale=p.detail_text_scale) for n,p in pipe.unet.attn_processors.items()
             if hasattr(p,'layer_group')}
    assert sum(v['group']=='detail' for v in routing.values())>0
    freeze(folder/'routing.json',routing)
    folder.mkdir(parents=True,exist_ok=True)
    for row in rows:
        image_path=folder/'images'/(row['id']+'.png'); metadata_path=image_path.with_suffix('.json')
        if image_path.exists() and metadata_path.exists():
            assert read(metadata_path)['output_sha256']==sha(image_path)
            continue
        seed=image_seed(row['id']); torch.manual_seed(seed)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        latent=torch.randn((1,4,64,48),generator=generator,device='cuda',dtype=torch.float16)
        latent_sha=hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
        sketch=Image.open(row['sketch']).convert('RGB').resize((384,512),Image.BILINEAR)
        sketch_tensor=torch.from_numpy(np.asarray(sketch,np.float32).transpose(2,0,1)/127.5-1)[None]
        reference=Image.open(row['reference']).convert('RGB')
        image=pipe(ref_image=sketch_tensor,prompt=row['caption'],texture_clip_image=reference,texture_embeds=None,
            null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,
            num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,generator=generator,
            latents=latent,num_inference_steps=50,texture_mode='patch_resampled',texture_num_tokens=16,
            texture_scale=1.,texture_condition_mode='token',fusion_type='minimal',texture_preprocess_mode='plain_resize',
            spatial_mask=None,debug_spatial=False,force_texture_num_tokens_override=False)[0]
        image_path.parent.mkdir(parents=True,exist_ok=True); image.save(image_path)
        write(metadata_path,dict(id=row['id'],arm=arm,seed=seed,initial_latent_sha256=latent_sha,
            checkpoint=str(path),gt_used_in_inference=False,output_sha256=sha(image_path),
            inputs={k:sha(row[k]) for k in ['gt','sketch','reference']}))
        print('RGB',stage,arm,row['id'],flush=True)
    write(folder/'generation_complete.json',dict(complete=True,n=len(rows),arm=arm,commit=commit()))
    del pipe; gc.collect(); torch.cuda.empty_cache()


def grid(stage,arm,rows,folder):
    page=Image.new('RGB',(4*128,8*(171+24)),'white'); draw=ImageDraw.Draw(page)
    for j,row in enumerate(rows[:8]):
        entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt']),
                 (arm,folder/'images'/(row['id']+'.png'))]
        for k,(label,path) in enumerate(entries):
            x=k*128; y=j*195
            draw.text((x+2,y+3),row['id']+' '+label,fill='black')
            page.paste(Image.open(path).convert('RGB').resize((128,171)),(x,y+24))
    page.save(folder/'output_grid.png')


@torch.inference_mode()
def measure(stage,arm):
    from eval.eval_utils import prepare_evaluation_masks,json_safe
    from eval.metrics import compute_structure_preservation,compute_texture_leakage,compute_texture_color_fidelity
    from eval.metrics import compute_texture_pattern_fidelity,compute_clip_i_values,compute_fid_from_paths
    from garment_mask_utils import mask_backend_info
    folder=OUT/'evaluation'/stage/arm
    if (folder/'evaluation_complete.json').exists(): return
    rows=input_rows(stage); lpips_model,lpips_info=frozen_lpips()
    backend=mask_backend_info(); assert backend['mask_backend']=='opencv'
    freeze(OUT/'evaluation/metric_protocol.json',dict(mask_backend=backend,lpips=lpips_info,
        mask_policy='sketch_only',evaluation_code={p:sha(SOURCE/p) for p in ['eval/metrics.py','eval/eval_utils.py','garment_mask_utils.py']},
        size=[384,512],missing_policy='retain identity, store NA and valid denominator; do not substitute samples',
        motif_identity_proven=False))
    records=[]; gen_paths=[]
    tensor=lambda image: torch.from_numpy(np.asarray(image,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    for row in rows:
        path=folder/'images'/(row['id']+'.png'); gen_paths.append(str(path))
        bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
        maskdir=OUT/'masks/evaluation'/row['id']; maskdir.mkdir(parents=True,exist_ok=True)
        for name in ['garment','outside','boundary']:
            if bundle.get(name) is not None:
                target=maskdir/(name+'.png'); array=bundle[name].astype('uint8')*255
                if target.exists(): assert np.array_equal(np.asarray(Image.open(target)),array)
                else: Image.fromarray(array).save(target)
        kwargs=dict(gen_path=str(path),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=bundle)
        structure=compute_structure_preservation(**kwargs); leakage=compute_texture_leakage(**kwargs)
        color=compute_texture_color_fidelity(texture_path=row['reference'],**kwargs)
        pattern=compute_texture_pattern_fidelity(texture_path=row['reference'],**kwargs)
        image=Image.open(path).convert('RGB'); gt=Image.open(row['gt']).convert('RGB').resize((384,512),Image.BILINEAR)
        record=dict(id=row['id'],arm=arm,edge_f1=structure.get('edge_f1'),iou=structure.get('sketch_iou'),
            edge_l1=structure.get('edge_l1'),leak=leakage.get('leak_colored_frac'),
            lpips=float(lpips_model(tensor(image),tensor(gt)).mean()),tcf_lab=color.get('tcf_lab_delta'),
            tcf_hsv=color.get('tcf_hsv_l1'),tcf_rgb_l2=color.get('tcf_rgb_l2'),tpf_patch=pattern.get('tpf_patch_sim'),
            mask_source=bundle['stats'].get('mask_source'),mask_low_confidence=bundle['stats'].get('mask_low_confidence'),
            warnings=structure.get('metric_warnings',[])+leakage.get('metric_warnings',[])+color.get('metric_warnings',[]))
        records.append(json_safe(record))
    for key,kind in [('clip_i_real','gt'),('clip_i_texture','reference')]:
        values=compute_clip_i_values(gen_paths,[r[kind] for r in rows],batch_size=4,model_name=str(PROJECT/'models/clip'))
        for record,value in zip(records,values): record[key]=float(value)
    auxiliary={}
    try:
        value,implementation=compute_fid_from_paths(gen_paths,[r['gt'] for r in rows],batch_size=8,return_backend=True)
        auxiliary=dict(fid=value,fid_backend=implementation,fid_unit='set; exploratory at small N')
    except Exception as error: auxiliary=dict(fid=None,fid_unavailable=str(error))
    write(folder/'per_identity_metrics.json',records); csv_write(folder/'per_identity_metrics.csv',records)
    write(folder/'auxiliary_metrics.json',auxiliary); grid(stage,arm,rows,folder)
    write(folder/'evaluation_complete.json',dict(complete=True,n=len(rows),arm=arm,commit=commit(),
        valid_counts={k:sum(r.get(k) is not None for r in records) for k in PRIMARY+APPEARANCE}))
    print('EVALUATION COMPLETE',stage,arm,len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['P1','P2','LOCK'],required=True)
    parser.add_argument('--arm',required=True); args=parser.parse_args()
    torch.set_num_threads(2)
    generate(args.stage,args.arm); measure(args.stage,args.arm)
