"""完整target组；RGB仅进入原冻结bbox自参考，GT只返回监督张量。"""
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from data.e32_field_dataset import cache_path
from data.e32_target_pseudogt import image_at,masks,structure_input,unit
from data.e33_interventions import nuisance,central_valid_box
from models.local_pattern_field import patch_geometry
from tools.e33r_common import DATASET,E32,cf_support,audit_seed

def source_inputs(rgb,support):
    lo,hi=central_valid_box(support)
    geometry=patch_geometry(Image.fromarray(rgb[lo:hi,lo:hi]))
    theta=np.deg2rad(2*geometry['orientation'])
    explicit=np.broadcast_to(np.float32([np.cos(theta),np.sin(theta),geometry['confidence']]),(16,16,3))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)/255
    mean=cv2.GaussianBlur(lab,(0,0),8)
    std=np.sqrt(np.maximum(cv2.GaussianBlur(lab*lab,(0,0),8)-mean*mean,0))
    color=cv2.resize(np.concatenate([mean,std],-1),(16,16))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    detail=cv2.resize(abs(gray-cv2.GaussianBlur(gray,(0,0),8)),(16,16))[...,None]
    aux=np.concatenate([explicit,color,detail],-1).astype(np.float32)
    return cv2.resize(rgb,(448,448),interpolation=cv2.INTER_AREA),aux

def topology(mask):
    m=mask.astype(np.uint8)
    components=cv2.connectedComponents(m,8)[0]-1
    holes=cv2.connectedComponents(1-np.pad(m,1),8)[0]-2
    return components,holes

def jitter_structure(row,dataset=DATASET):
    seed=int(hashlib.sha256(('E33R/jitter/'+row['id']).encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed);rgb=np.asarray(image_at(dataset/row['sketch']))
    dx,dy=rng.choice([-1,1],2)
    shifted=cv2.warpAffine(rgb,np.float32([[1,0,dx],[0,1,dy]]),(384,512),borderValue=(255,255,255))
    image=Image.fromarray(shifted);mask,_=masks(image);ink=cv2.cvtColor(shifted,cv2.COLOR_RGB2GRAY)<200
    internal=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
    coords=np.argwhere(ink&internal);trial=shifted.copy()
    if len(coords):
        chosen=coords[rng.choice(len(coords),max(1,int(.01*len(coords))),replace=False)]
        trial[chosen[:,0],chosen[:,1]]=255
    trial_image=Image.fromarray(trial);trial_mask,_=masks(trial_image)
    keep=topology(trial_mask)==topology(mask) and topology(cv2.cvtColor(trial,cv2.COLOR_RGB2GRAY)<200)==topology(ink)
    if keep:image,mask=trial_image,trial_mask
    field=cv2.resize(structure_input(image,mask),(96,128),interpolation=cv2.INTER_AREA).transpose(2,0,1)
    return torch.from_numpy(field.astype(np.float32)),dict(dx=int(dx),dy=int(dy),dropout_accepted=keep)

class GroupDataset(Dataset):
    def __init__(self,records,training=False,dataset=DATASET,e32=E32):
        self.records,self.training,self.dataset,self.e32=records,training,dataset,e32
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        cv2.setNumThreads(1);row=self.records[index]
        with np.load(cache_path(self.e32,row['id'])) as f:
            structure=np.asarray(f['structure'],np.float32)
            gt=np.asarray(f['supervision_geometry'],np.float32)
            interior=np.asarray(f['supervision_interior'],np.float32)
        rgb=np.asarray(image_at(self.dataset/row['target']).crop(tuple(row['box'])))
        pixels=[];aux=[]
        seeds=[int(np.random.randint(0,2**32,dtype=np.uint32))] if self.training else [audit_seed(row['id'],j) for j in range(2)]
        # 0..2 clean R0/R90/R180；3..5 paired nuisance；eval追加6..8第二次nuisance。
        for seed in [None]+seeds:
            for k in range(3):
                image=np.rot90(rgb,k).copy();support=np.ones(image.shape[:2],bool)
                if seed is not None:image,support=nuisance(image,support,seed)
                p,a=source_inputs(image,support);pixels.append(p);aux.append(a)
        support=cf_support(gt,interior,row['box'])
        return dict(id=row['id'],pixels=torch.from_numpy(np.stack(pixels)),aux=torch.from_numpy(np.stack(aux)),
            structure=torch.from_numpy(structure),gt=torch.from_numpy(gt),support=torch.from_numpy(support),
            strict=row['strict'])

@torch.no_grad()
def encode_reference(pixels,aux,dino,variant='full',device='cuda'):
    """源图编码；不接收target structure/GT/bbox或干预标签。"""
    b,arms=pixels.shape[:2];aux=aux.to(device)
    if variant=='G_geometry_only':dense=aux.new_zeros(b*arms,16,16,384)
    else:
        dense=[]
        for chunk in pixels.flatten(0,1).split(2):
            t=chunk.to(device).permute(0,3,1,2).float()/255
            t=(t-t.new_tensor([.485,.456,.406])[None,:,None,None])/t.new_tensor([.229,.224,.225])[None,:,None,None]
            with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
                feature=torch.stack(dino.get_intermediate_layers(t,n=4,reshape=True)).mean(0)
            feature=torch.nn.functional.interpolate(feature.float(),(16,16),mode='bilinear',align_corners=False).permute(0,2,3,1)
            dense.append(torch.nn.functional.normalize(feature,dim=-1))
        dense=torch.cat(dense)
    return torch.cat([dense.reshape(b,arms,16,16,384),aux],-1)
