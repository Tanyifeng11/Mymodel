"""E32 StageA 离线特征：缓存中严格区分输入与监督。"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask

from data.e32_target_pseudogt import (FrozenFeatures, image_at, masks, structure_input,
                                     upsample_geometry, TAU, COARSE, unit)
from models.apacc_features import load_dino
from tools.e32_common import read, write, sha


def cache_path(out, sid):
    return out/'A_geometry/cache'/(sid+'.npz')


def cpu_prepare(task):
    out, dataset, row, rotation = task
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    reference, sketch, target = [image_at(dataset/row[k]) for k in ('reference','sketch','target')]
    input_mask, interior, info = masks(sketch,target)
    foreground=np.asarray(estimate_cloth_foreground_mask(target,384,512)[0])>127
    extractor = FrozenFeatures(None,'cpu')
    source=extractor.extract(reference,perceptual=False)
    gt=extractor.extract(target,interior,perceptual=False)
    structure=cv2.resize(structure_input(sketch,input_mask),(96,128),interpolation=cv2.INTER_AREA).transpose(2,0,1)
    source_geometry=source['geometry'];target_geometry=gt['geometry']
    readable=(target_geometry[...,3]>=TAU)&(gt['occupancy']>=.95)
    ref_readable=source_geometry[...,3]>=TAU
    if readable.any() and ref_readable.any():
        g=target_geometry[readable];r=source_geometry[ref_readable]
        ori=np.clip(unit(g[:,:2])@unit(r[:,:2]).T,-1,1)
        geometry=float(((ori.max(1)+1)/2).mean())
    else:
        geometry=.5
    color=float(source['histogram']@gt['histogram'])
    weight=np.clip(.25+.75*(.5*color+.5*geometry),.25,1.)
    values=dict(reference=source['reference'],structure=structure.astype(np.float16),
                supervision_geometry=upsample_geometry(target_geometry).transpose(2,0,1).astype(np.float16),
                supervision_interior=cv2.resize(interior.astype(np.float32),(48,64),interpolation=cv2.INTER_AREA).astype(np.float16),
                supervision_foreground=cv2.resize(foreground.astype(np.float32),(48,64),interpolation=cv2.INTER_AREA).astype(np.float16),
                supervision_unmasked_geometry=upsample_geometry(gt['unmasked_geometry']).transpose(2,0,1).astype(np.float16),
                pair_weight=np.float32(weight),reference_histogram=source['histogram'])
    pixels=[cv2.resize(np.asarray(reference),(448,448),interpolation=cv2.INTER_AREA)]
    if rotation:
        # 保留像素尺度与90度角度；固定canvas裁切/补白，不做长宽交换后的拉伸。
        matrix=cv2.getRotationMatrix2D((191.5,255.5),90,1.)
        rotated=Image.fromarray(cv2.warpAffine(np.asarray(reference),matrix,reference.size,
                            flags=cv2.INTER_LINEAR,borderValue=(255,255,255)))
        rotated_feat=extractor.extract(rotated,perceptual=False)
        values['reference_rot90']=rotated_feat['reference']
        pixels.append(cv2.resize(np.asarray(rotated),(448,448),interpolation=cv2.INTER_AREA))
    return row['id'],values,np.stack(pixels),info


@torch.inference_mode()
def dino_batch(pixels,model,device):
    t=torch.from_numpy(pixels).permute(0,3,1,2).float().to(device)/255
    mean=t.new_tensor([.485,.456,.406])[None,:,None,None]
    std=t.new_tensor([.229,.224,.225])[None,:,None,None]
    maps=torch.stack(model.get_intermediate_layers((t-mean)/std,n=4,reshape=True)).mean(0)
    maps=maps.cpu().numpy().transpose(0,2,3,1)
    return [unit(cv2.resize(v,COARSE)).astype(np.float16) for v in maps]


def prepare(out,dataset,weights):
    assert read(out/'decision_summary.json')['pair_dependence_pass']
    split=read(out/'split_manifest.json')
    audit_path=out/'audits/target_group_split_audit.json'
    if not audit_path.exists():
        known=read(out/'audits/stage0_input_hashes.json')
        holdouts=split['dev']+split['confirmation_all']
        heldout_hashes={known[r['id']]['target'] for r in holdouts}
        target_hashes={};reserved=[]
        for i,row in enumerate(split['train']):
            digest=sha(dataset/row['target']);target_hashes[row['id']]=digest
            if digest in heldout_hashes:reserved.append(row)
            if (i+1)%4096==0:print('[E32 target split hash]',i+1,'/',len(split['train']),flush=True)
        # 重复target归到保留组，不改dev选取、不删原始数据、不按confidence删样本。
        reserved_ids={r['id'] for r in reserved}
        split['train']=[r for r in split['train'] if r['id'] not in reserved_ids]
        split['reserved_duplicate_target_rows']=reserved
        write(out/'split_manifest.json',split)
        write(audit_path,dict(training_target_hashes=target_hashes,heldout_hashes=sorted(heldout_hashes),
              reserved_duplicate_target_ids=sorted(reserved_ids),train_after_group_guard=len(split['train']),
              **{'pass':True},note='distinct target file SHA256 groups; references may share seen pattern; no visual-near-duplicate claims'))
    write(out/'A_geometry/preregistered_protocol.json',dict(steps=8000,batch=8,seeds=[42,43,44],
          lr=1e-4,weight_decay=1e-4,warmup=500,scheduler='cosine',reference_dropout=.1,
          loss_weights=dict(orientation=1.,period=1.,confidence=.1,smoothness=.05),
          fixed_StageA_ablations_seed42=['A_no_target_supervision','E_no_interior','G_no_E26','G_dino_only'],
          gate='dev: ori advantage>=10deg and period>=.15log2 with CI95 lower>0; rot90 success>=.7; zero ori/period error degradation CI95 lower>0; >=2/3 seeds',
          known_causal_cases='diagnostic evaluation only; no hyperparameter adjustment',
          decision_after='all fixed geometry-stage ablations, no appearance-stage training before Gate'))
    records=split['train']+split['dev']+split['confirmation_all']
    evaluation={r['id'] for r in split['dev']+split['confirmation_all']}
    tasks=[(out,dataset,r,r['id'] in evaluation) for r in records if not cache_path(out,r['id']).exists()]
    device='cuda'
    model,_=load_dino(device,weights)
    # CPU算E26，GPU仅批量计算冻结DINO；不在登录节点做计算。
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator=pool.map(cpu_prepare,tasks,chunksize=2)
        completed=0
        while True:
            chunk=[]
            for _ in range(4):
                try: chunk.append(next(iterator))
                except StopIteration: break
            if not chunk:break
            pixels=np.concatenate([v[2] for v in chunk])
            appearances=dino_batch(pixels,model,device)
            offset=0
            for sid,values,pixel,info in chunk:
                values['reference'][...,:384]=appearances[offset];offset+=1
                if 'reference_rot90' in values:
                    values['reference_rot90'][...,:384]=appearances[offset];offset+=1
                path=cache_path(out,sid);path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**values)
                write(out/'audits/training_mask_diagnostics'/(sid+'.json'),info)
            completed+=len(chunk)
            if completed%128==0 or completed==len(tasks):
                print('[E32 A cache]',completed,'/',len(tasks),flush=True)
    assert all(cache_path(out,r['id']).exists() for r in records)
    write(out/'A_geometry/cache_manifest.json',dict(case_count=len(records),train_count=len(split['train']),
          input_keys=['reference','structure'],supervision_keys=['supervision_geometry','supervision_interior','pair_weight'],
          input_structure_shape=[7,128,96],reference_shape=[16,12,395],target_rgb_is_input=False,
          pair_weight='clip(.25+.75*(.5 color histogram compatibility+.5 readable orientation compatibility),.25,1)',
          all_eligible_training_samples_retained=True,heldout_duplicate_target_ids=[r['id'] for r in split.get('reserved_duplicate_target_rows',[])],
          rotation='true centered90 rotation in fixed384x512 canvas, crop/pad without stretch; same target sketch/mask',
          note='GT-derived masks and geometry only in supervision keys; geometry confidence is shared E26 confidence'))


class FieldDataset(Dataset):
    def __init__(self,out,records):
        self.out,self.records=out,records

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        row=self.records[index]
        with np.load(cache_path(self.out,row['id'])) as f:
            # 明确键名白名单；任何target特征都不能拼进reference或structure。
            value={key:torch.from_numpy(np.asarray(f[key],np.float32)) for key in
                   ('reference','structure','supervision_geometry','supervision_interior','pair_weight',
                    'supervision_foreground','supervision_unmasked_geometry')}
        value['id']=row['id']
        return value
