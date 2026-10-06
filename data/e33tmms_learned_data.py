"""复用原E33 controlled/real身份；target RGB与box仅在监督/既有自参考构建中使用。"""
import cv2
import numpy as np
import torch
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from torch.utils.data import Dataset
from data.e32_target_pseudogt import image_at,masks
from data.e32_field_dataset import cache_path
from data.e33r_group_dataset import source_inputs
from data.e33rf_real_rotation_dataset import reference_group,rotate
from tools.e33r_common import cf_support
from tools.e33rf_common import E32,DATASET

def tensor(image):return torch.from_numpy(np.asarray(image,dtype=np.float32).copy().transpose(2,0,1)/255)

class CounterfactualRGBDataset(Dataset):
    def __init__(self,rows,controlled):self.rows,self.controlled=rows,controlled
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        cv2.setNumThreads(1);row=self.rows[index];sid=row['id']
        with np.load(cache_path(E32,sid)) as z:
            structure=z['structure'].astype(np.float32);gt=z['supervision_geometry'].astype(np.float32)
            inner=z['supervision_interior'].copy()
        mask=masks(image_at(DATASET/row['sketch']))[0].astype(np.float32)
        target=image_at(DATASET/row['target']);target_rgb=np.asarray(target).copy()
        if self.controlled:
            box=row['box'];crop=np.asarray(target.crop(tuple(box)));pixels=[];aux=[];refs=[];targets=[]
            for k in range(3):
                source=np.rot90(crop,k).copy();p,a=source_inputs(source,np.ones(source.shape[:2],bool))
                pixels.append(p);aux.append(a);refs.append(tensor(Image.fromarray(source).resize((384,512),Image.Resampling.BICUBIC)))
                supervision=target_rgb.copy();x,y,x1,y1=box;supervision[y:y1,x:x1]=source
                targets.append(tensor(Image.fromarray(supervision)))
            support=cf_support(gt,inner,box)
            features=np.zeros((3,16,16,394),np.float32)
            rec_arms=np.ones(3,np.float32)
        else:
            ref=image_at(DATASET/row['reference']);refs=[tensor(v) for v in [ref,rotate(ref,90),rotate(ref,180)]]
            features=reference_group(row)['reference'].numpy()
            pixels=np.zeros((3,448,448,3),np.uint8);aux=np.zeros((3,16,16,10),np.float32)
            targets=[tensor(target)]*3;support=(inner>=.95)&(gt[3]>=.25)
            # 真实reference干预没有旋转target真值，只有R0参与像素重建。
            rec_arms=np.array([1,0,0],np.float32)
        source_masks=[]
        for ref_tensor in refs:
            ref_image=Image.fromarray(np.uint8(np.round(ref_tensor.numpy().transpose(1,2,0)*255)))
            source_masks.append(torch.from_numpy((np.asarray(estimate_cloth_foreground_mask(ref_image,*ref_image.size)[0])>127).astype(np.float32))[None])
        return dict(id=sid,controlled=self.controlled,references=torch.stack(refs),source_masks=torch.stack(source_masks),targets=torch.stack(targets),
            pixels=torch.from_numpy(np.stack(pixels)),aux=torch.from_numpy(np.stack(aux)),
            reference_features=torch.from_numpy(features),structure=torch.from_numpy(structure),
            mask=torch.from_numpy(mask)[None],support=torch.from_numpy(support),gt=torch.from_numpy(gt),
            rec_arms=torch.from_numpy(rec_arms))
