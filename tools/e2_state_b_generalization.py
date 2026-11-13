"""冻结B400与Lab约束，验证独立新身份和原始真实纹理。"""
import argparse
import gc
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_lab_anchor import LabAnchor
from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_generalization_data as inputs
from tools import e2_state_b_pilot as pilot
from tools import e2_state_b_repair as repair
from tools.e2_fft_experiment import SOURCE, read, write, sha
from tools.e2_oracle_texture import projected, weights_digest, encode, decode
from tools.e2_state_b_lab_anchor import CHECKPOINT, png

ARMS = ('m0', 'raw', 'ab_l')
CONDITIONS = ('v16', 'h16', 'v32', 'h32', 'real_a', 'real_b')


def source_hashes():
    names = ('tools/e2_state_b_generalization.py', 'tools/e2_state_b_generalization_data.py',
        'tools/e2_state_b_generalization_metrics.py', 'models/e2_lab_anchor.py',
        'models/e2_state_texture_b.py', 'tools/e2_state_b_pilot.py', 'tools/e2_state_b_repair.py',
        'tools/e2_b_repair_losses.py', 'tools/e2_oracle_texture.py', 'tools/e2_oracle_color.py',
        'tools/t3_response_run.py', 'tools/bf_t3_three_arm_report.py', 'tools/e2_state_b_lab_anchor.py',
        'pipelines/IMAGGarment_pipeline.py')
    return {name:sha(SOURCE/name) for name in names}


def freeze(args):
    assert not (args.out/'protocol.json').exists(), '输入清单已经冻结'
    complete=read(args.repair/'training/complete.json')
    assert complete['frozen_unchanged'] and complete['checkpoint_sha256']==CHECKPOINT
    assert sha(args.repair/'training/step400.pt')==CHECKPOINT
    spec=inputs.freeze(args)
    spec.update(code=source_hashes(), checkpoint_sha256=CHECKPOINT, frozen_hash=complete['frozen_hash'],
        arms=list(ARMS), conditions=list(CONDITIONS), seeds=2, shards=2,
        sampler=dict(steps=50,cfg=7.,sketch_scale=.6), active_steps=[25,50], sigma=24.,
        anchor='same-seed M0(v16) for all synthetic conditions; M0(real_a) for real_a/real_b',
        real_reference='original RGB, only model-required 384x512 bicubic resize; no recolouring/rotation',
        metrics=dict(synthetic='unchanged strict FFT/orientation/ACF and original-target old-template projection',
            real='RGB and per-patch contrast-normalized gray DISTS target/foil distances and bidirectional margins',
            palette='32 fixed angles, 1024 quantiles, at most 4096 evenly spaced pixels of Lab ab; reference versus garment interior',
            brightness='mean L and masked sigma24 low-frequency L drift versus fixed same-seed M0 anchor',
            interpretation='DISTS margins and old-reference similarity are proxies, not pattern classification or residual ground truth',
            collapse='reference ab 10-90% spread >=12 and output spread <25% reference spread'),
        scope='16 new target identities, 16 independent real reference donors, 8 low-chroma-spread and 8 multicolour; exploratory',
        gates='synthetic: retain raw strict switches and old removal, improve anchor colour/L; '
              'real: retain gray response and improve reference palette/L versus raw; both colour strata checked separately')
    for p in spec['pairs']:
        folder=args.out/'cases'/p['case_id']
        p['mask_sha256']=sha(folder/'mask.png')
        p['sketch_copy_sha256']=sha(folder/'sketch.jpg')
    write(args.out/'protocol.json',spec)
    print('NEW_REAL_FROZEN',len(spec['pairs']),sha(args.out/'protocol.json'),flush=True)


def generate(args):
    spec=read(args.out/'protocol.json')
    assert spec['code']==source_hashes()
    checkpoint=args.repair/'training/step400.pt'
    assert sha(checkpoint)==CHECKPOINT
    core,pipe,net=pilot.load_core(args)
    b=StateTextureB(color_channels=2).to(args.device)
    b.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=False)['b'])
    b.eval().requires_grad_(False)
    before_m0,before_b=weights_digest(net),weights_digest(b)
    assert before_m0==spec['frozen_hash']
    with torch.inference_mode():
        for p in spec['pairs'][args.shard::2]:
            folder=args.out/'cases'/p['case_id']
            assert sha(args.data/p['sketch'])==p['source_hashes']['target']['sketch']
            assert sha(folder/'mask.png')==p['mask_sha256']
            assert all(sha(folder/(c+'_ref.png'))==p['reference_hashes'][c] for c in CONDITIONS)
            rows=[]
            sketch=core.tensor(args.data/p['sketch']).to(args.device).half()
            mask=np.asarray(Image.open(folder/'mask.png'))>0
            for seed in range(2):
                pair=dict(p,seed=p['seed']+100000*seed)
                base=torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                digest=core.digest(base)
                m0={}
                def sample(condition,module=None,layer=None):
                    features=projected(net,sketch,core.tensor(folder/(condition+'_ref.png')).to(args.device).half())
                    values,_=pilot.inputs(args,pair,condition,args.device)
                    if layer is not None:values['color']=layer.ab.expand(1,2,64,48)/100
                    captured={}
                    def transform(value):
                        assert not captured
                        captured['raw']=png(pipe,value)
                        result=layer(value,'ab_l')
                        captured['audit']=dict(layer.last_audit)
                        return result
                    net.adapter.features=features
                    with repair.Tap(pipe,module,values,end=50) as tap:
                        image=pipe(ref_image=sketch,prompt=pair['caption'],texture_clip_image=None,
                            texture_embeds=None,null_prompt='',negative_prompt=' worst quality, low quality',
                            width=384,height=512,num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,
                            ipa_scale=0.,latents=base.clone(),
                            generator=torch.Generator(device=args.device).manual_seed(pair['seed']),
                            num_inference_steps=50,**({'decoded_output_transform':transform} if layer is not None else {}))[0]
                    net.adapter.clear()
                    assert tap.index==50 and core.digest(base)==digest
                    return image,captured
                # M0先生成；公共锚点不读取B输出、换布目标或评分统计。
                for c in CONDITIONS:
                    m0[c],_=sample(c)
                    m0[c].save(folder/f'{c}_s{seed}_m0.png')
                layers={group:LabAnchor(np.asarray(m0[c],np.float32)/255,mask,sigma=24.).to(args.device).eval()
                        for group,c in [('synthetic','v16'),('real','real_a')]}
                write(folder/f'anchors_s{seed}.json',{k:v.describe() for k,v in layers.items()})
                for c in CONDITIONS:
                    layer=layers['real' if c.startswith('real') else 'synthetic']
                    picture,captured=sample(c,b,layer)
                    captured['raw'].save(folder/f'{c}_s{seed}_raw.png')
                    picture.save(folder/f'{c}_s{seed}_ab_l.png')
                    if c in pilot.VARIANTS:
                        # 与前轮相同的原目标/VAE正控制，仅评价使用。
                        target=pilot.target_images(args,pair,c,m0['v16'],f'_s{seed}')
                        decode(pipe.vae,encode(pipe.vae,target,args.device)).save(folder/f'{c}_s{seed}_vae.png')
                    rows.append(dict(condition=c,seed=pair['seed'],digest=digest,
                        output_sha256={a:sha(folder/f'{c}_s{seed}_{a}.png') for a in ARMS},projection=captured['audit']))
                print('NEW_REAL_GENERATED',p['case_id'],seed,flush=True)
            write(folder/'generation.json',dict(rows=rows,paired_noise=True,updates=0))
    assert before_m0==weights_digest(net) and before_b==weights_digest(b) and sha(checkpoint)==CHECKPOINT
    write(args.out/f'shard{args.shard}.json',dict(updates=0,frozen_unchanged=True,m0_hash=before_m0,
        b_hash=before_b,checkpoint_sha256=CHECKPOINT,protocol_sha256=sha(args.out/'protocol.json')))
    del core,pipe,net,b
    gc.collect();torch.cuda.empty_cache()
    from tools.e2_state_b_generalization_metrics import evaluate
    evaluate(args)


def panel(args,p,group):
    folder=args.out/'cases'/p['case_id']
    conditions=CONDITIONS[:4] if group=='synthetic' else CONDITIONS[4:]
    canvas=Image.new('RGB',(5*194,len(conditions)*282),'white');draw=ImageDraw.Draw(canvas)
    for row,c in enumerate(conditions):
        items=[('reference',folder/(c+'_ref.png'))]
        if group=='synthetic':items.append(('target',folder/f'{c}_s0_target.png'))
        else:items.append(('sketch',folder/'sketch.jpg'))
        items += [(a,folder/f'{c}_s0_{a}.png') for a in ARMS]
        for col,(name,path) in enumerate(items):
            x,y=col*194,row*282
            draw.text((x+3,y+3),c+' '+name,fill='black')
            canvas.paste(Image.open(path).convert('RGB').resize((192,256)),(x,y+24))
    canvas.save(folder/(group+'_panel.jpg'),quality=95)


def summarize(args):
    from tools.e2_state_b_generalization_metrics import aggregate
    spec=read(args.out/'protocol.json')
    assert spec['code']==source_hashes()
    audits=[read(args.out/f'shard{s}.json') for s in range(2)]
    assert audits[0]==audits[1] and audits[0]['frozen_unchanged'] and audits[0]['updates']==0
    result=aggregate(args,spec)
    for p in spec['pairs']:
        for group in ('synthetic','real'):panel(args,p,group)
    result.update(protocol=spec,training=audits[0],condition_seed_groups=192,generated_images=576,
        interpretation='zero-training generalization test; real old-pattern removal has no pixel ground truth; '
            'uniform ab projection may destroy multicolour fabric even when anchor mean colour improves')
    write(args.out/'summary.json',result)
    archive=args.out/'newid_real_review.zip'
    files=[args.out/'protocol.json',args.out/'summary.json',*args.out.glob('shard*.json'),
           *args.out.glob('metrics*/**/*.json'),*args.out.glob('cases/*/*'),*args.out.glob('inputs/*.json')]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):
            if path.is_file():z.write(path,path.relative_to(args.out))
    print('NEW_REAL_RESULTS',result['groups'],flush=True)
    print('NEW_REAL_ARCHIVE',archive.stat().st_size,sha(archive),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','freeze','generate','evaluate','summarize'))
    parser.add_argument('--project',type=Path,default=pilot.PROJECT)
    parser.add_argument('--data',type=Path,default=pilot.PROJECT.parent/'datasets/BF')
    parser.add_argument('--repair',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_repair_20261112')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--pool-review',type=Path)
    parser.add_argument('--pool-only',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--shard',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    if args.action=='prepare':inputs.prepare(args)
    elif args.action=='evaluate':
        from tools.e2_state_b_generalization_metrics import evaluate
        evaluate(args)
    else:globals()[args.action](args)


if __name__=='__main__':main()
