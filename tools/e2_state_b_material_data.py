"""固定原4个目标身份、7张训练裁片；11个验证目标及供体均禁止进入训练。"""
import copy
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.e2_state_texture_b import StateTextureB
from tools.e2_fft_experiment import SOURCE,read,write,sha,environment
from tools.e2_oracle_texture import RGB_LUMA
from tools.e2_state_b_generalization_data import perceptual_hash
from tools.e2_b_material_losses import feather

TRAIN=(621,1641,870,1526)
REFERENCES={621:'stripes',1641:'stripes',870:'dots',1526:'checks',6210:'stripes',7826:'checks',8269:'dots'}
VARIANTS=('native','r90','s50')
CONDITIONS=tuple(f'r{i}_{v}' for i in REFERENCES for v in VARIANTS)


def prepare(args,code):
    from eval.eval_utils import prepare_evaluation_masks
    assert not (args.out/'protocol.json').exists()
    evaluation=read(args.validation/'protocol.json')
    assert sha(args.validation/'protocol.json')=='466a2cf17aebe651b76071586501b01c00fdf6acd9fd987d4cf7c00fdedc9117'
    original=read(args.baseline/'protocol.json')
    registry_file=args.project/'output_eval/e2_b_screen_20261108/inputs/registry.json'
    assert sha(registry_file)==evaluation['input_audit']['registry_sha256']
    registry=read(registry_file)['records'];by_id={r['index']:r for r in registry}
    protected={p[k] for p in evaluation['pairs'] for k in ('identity_group','donor_identity_group')}
    trainroots={by_id[i]['identity_group'] for i in REFERENCES}|{by_id[i]['identity_group'] for i in TRAIN}
    assert len(protected)==22 and not protected&trainroots
    held=[r for r in registry if r['identity_group'] in protected];assert len(held)==22
    used=[by_id[i] for i in REFERENCES]
    pixel=lambda p:hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()
    protected_hashes={pixel(args.data/r[k]) for r in held for k in ('gt','texture')}
    phash={r['identity_group']:perceptual_hash(args.data/r['gt']) for r in held}
    independence=[]
    for row in used:
        for k in ('sketch','gt','texture'):assert sha(args.data/row[k])==row['hashes'][k]
        assert all(pixel(args.data/row[k]) not in protected_hashes for k in ('gt','texture'))
        fingerprint=perceptual_hash(args.data/row['gt'])
        nearest=min(bin(fingerprint^v).count('1') for v in phash.values())
        assert nearest>4,(row['index'],nearest)
        independence.append(dict(index=row['index'],identity_group=row['identity_group'],nearest_heldout_gt_phash=nearest))
    refs=args.out/'references';refs.mkdir(parents=True,exist_ok=True)
    for i in REFERENCES:
        native=Image.open(args.data/by_id[i]['texture']).convert('RGB').resize((384,512),Image.Resampling.BICUBIC)
        rotated=native.transpose(Image.Transpose.ROTATE_90).resize((384,512),Image.Resampling.BICUBIC)
        half=native.resize((192,256),Image.Resampling.BICUBIC);small=Image.new('RGB',(384,512))
        for x,y in ((0,0),(192,0),(0,256),(192,256)):small.paste(half,(x,y))
        for v,img in zip(VARIANTS,(native,rotated,small)):img.save(refs/f'r{i}_{v}.png')
    train=[]
    for i in TRAIN:
        p=copy.deepcopy(next(p for p in original['pairs'] if p['index']==i));p['split']='train'
        folder=args.out/'cases'/p['case_id'];folder.mkdir(parents=True,exist_ok=True)
        mask=prepare_evaluation_masks((384,512),sketch_path=str(args.data/p['sketch']),mask_policy='sketch_only')['garment']
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
        shutil.copyfile(args.data/p['sketch'],folder/'sketch.jpg')
        for c in CONDITIONS:shutil.copyfile(refs/(c+'.png'),folder/(c+'_ref.png'))
        p.update(mask_sha256=sha(folder/'mask.png'),reference_hashes={c:sha(folder/(c+'_ref.png')) for c in CONDITIONS})
        train.append(p)
    for p in evaluation['pairs']:
        src=args.validation/'cases'/p['case_id'];dst=args.out/'cases'/p['case_id'];dst.mkdir(parents=True,exist_ok=True)
        names=['mask.png','sketch.jpg']+[c+'_ref.png' for c in evaluation['conditions']]
        names += [f'{c}_s{s}_{arm}.png' for c in evaluation['conditions'] for s in range(2) for arm in ('m0','raw','ab_l')]
        names += ['generation.json','result.json']
        for name in names:
            target=name.replace('_raw.png','_old_raw.png').replace('_ab_l.png','_old_lab.png')
            if name in ('generation.json','result.json'):target='previous_'+name
            shutil.copyfile(src/name,dst/target);assert sha(src/name)==sha(dst/target)
    spec=dict(source=environment(),code=code,train=train,holdout=copy.deepcopy(evaluation['pairs']),
        references={str(i):dict(family=f,source=by_id[i]['texture'],identity_group=by_id[i]['identity_group'],sha256=by_id[i]['hashes']['texture']) for i,f in REFERENCES.items()},
        conditions=list(CONDITIONS),reference_hashes={c:sha(refs/(c+'.png')) for c in CONDITIONS},
        validation_conditions=evaluation['conditions'],validation_protocol_sha256=sha(args.validation/'protocol.json'),
        protected_product_roots=sorted(protected),independence=independence,input_only_review=True,
        training_updates=400,prior_b_updates=400,initialization='warm start repaired B400; zero new spatial-ab input weights; reference mean ab replaces M0 mean ab',
        reference='64 pixel-unshuffled luma channels plus128 spatial ab channels, two reference mean-ab maps',
        postprocessing='none; learned B raw output is primary',sampler=dict(steps=50,cfg=7.,sketch_scale=.6,active=[25,50]),
        target='original real RGB/Lab texture plus zero-mean sigma64 M0 illumination bounded to5 L; per-pixel gamut-safe L; outside M0',
        loss='latent, spatial RGB, full Lab highpass, spatial ab, sliced ab palette, low-frequency L residual to full target, old-template residual, outside protection',
        evaluation='same11 identities/2seeds/6refs; step0 vs400; old cached B/raw/Lab context; full-reference and visible-target palette separately',
        gates=dict(reference_response='real gray bidirectional counts and mean margin >=step0',
            pattern='real mean highpass Lab correlation and contrast error improve; at least75% positive-control contrast and correlation>=.5 on both refs for a strict pair',
            residue='mean old-template residual projection to known controlled goal decreases',
            color='palette SWD decreases and all12 multicolour real outputs retain >=75% VAE-positive chroma spread',
            protection='outside deviation from same-condition M0 does not increase'),
        interpretation='controlled targets constructed from real crops, not captured counterfactual garment ground truth; 11 reused cases are regression/generalization probes, not a new confirmation set')
    write(args.out/'protocol.json',spec)
    print('MATERIAL_PREPARED',len(train),len(CONDITIONS),len(evaluation['pairs']),flush=True)


def inputs(args,p,c,device):
    d=args.out/'cases'/p['case_id'];rgb=np.asarray(Image.open(d/(c+'_ref.png')).convert('RGB'),np.float32)/255
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    mask=np.asarray(Image.open(d/'mask.png'))>0;weight=feather(mask)
    sketch=np.asarray(Image.open(args.data/p['sketch']).convert('RGB').resize((384,512)),np.float32)@RGB_LUMA/255
    tensor=lambda v:torch.from_numpy(np.ascontiguousarray(v)).float().to(device)
    luma=tensor(rgb@RGB_LUMA)[None,None]*2-1
    chroma=tensor(lab[...,1:].transpose(2,0,1))[None]/100
    reference=torch.cat((F.pixel_unshuffle(luma,8),F.pixel_unshuffle(chroma,8)),1)
    mean=chroma.mean((2,3),keepdim=True)
    return dict(reference=reference,mask=F.avg_pool2d(tensor(weight)[None,None],8),
        structure=F.avg_pool2d(tensor(1-sketch)[None,None],8),color=mean.expand(1,2,64,48)),weight


def initialize(args,device):
    old=torch.load(args.repair/'training/step400.pt',map_location=device,weights_only=False)['b']
    b=StateTextureB(color_channels=2,reference_channels=192).to(device)
    state={k:v.clone() for k,v in old.items()}
    kernel=torch.zeros_like(b.input.weight)
    kernel[:,:72]=old['input.weight'][:,:72]
    kernel[:,200:]=old['input.weight'][:,72:]
    state['input.weight']=kernel;b.load_state_dict(state)
    assert sum(p.numel() for p in b.parameters())==274372
    return b
