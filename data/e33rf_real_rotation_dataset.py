"""沿用E32原matched特征和监督；新缓存只包含reference旋转特征。"""
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset
from data.e32_target_pseudogt import image_at,FrozenFeatures
from data.e32_field_dataset import cache_path,dino_batch
from data.e33rc_real_pair_dataset import cached_real,orientation_reference
from tools.e33rf_common import OUT,E32,DATASET

def rotate(image,degrees):
    m=cv2.getRotationMatrix2D((191.5,255.5),degrees,1.)
    return Image.fromarray(cv2.warpAffine(np.asarray(image),m,image.size,flags=cv2.INTER_LINEAR,borderValue=(255,255,255)))
def rotation_path(sid):return OUT/'rotation_cache'/(sid+'.npz')
def cpu_rotation(row):
    cv2.setNumThreads(1);torch.set_num_threads(1)
    image=image_at(DATASET/row['reference']);extractor=FrozenFeatures(None,'cpu')
    refs=[];pixels=[]
    with np.load(cache_path(E32,row['id'])) as z:has90='reference_rot90' in z
    angles=[180] if has90 else [90,180]
    for angle in angles:
        rgb=rotate(image,angle);refs.append(extractor.extract(rgb,perceptual=False)['reference'])
        pixels.append(cv2.resize(np.asarray(rgb),(448,448),interpolation=cv2.INTER_AREA))
    return row['id'],angles,refs,np.stack(pixels)
def write_rotations(result,dino):
    sid,angles,refs,pixels=result
    # 与原E32缓存一致：冻结DINO末4层bf16，插值后归一化，float16落盘。
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):appearance=dino_batch(pixels,dino,'cuda')
    values={}
    for angle,ref,dense in zip(angles,refs,appearance):
        ref[...,:384]=dense;values['reference_rot%d'%angle]=orientation_reference(ref).astype(np.float16)
    p=rotation_path(sid);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix('.tmp.npz');np.savez_compressed(temporary,**values);temporary.replace(p)
def reference_group(row,phase='RF2',wrong=None):
    value=cached_real(row,False);refs=[value['reference']]
    if phase!='RF1':
        with np.load(rotation_path(row['id'])) as rot,np.load(cache_path(E32,row['id'])) as old:
            r90=orientation_reference(old['reference_rot90']) if 'reference_rot90' in old else np.asarray(rot['reference_rot90'],np.float32)
            refs += [r90,np.asarray(rot['reference_rot180'],np.float32)]
    if phase=='RF3':
        for arm in ['color_near','random']:
            with np.load(cache_path(E32,wrong[row['id']][arm])) as z:refs.append(orientation_reference(z['reference']))
    gt=value['supervision_geometry'];support=(value['supervision_interior']>=.95)&(gt[3]>=.25)
    return dict(id=row['id'],reference=torch.from_numpy(np.stack(refs)),structure=torch.from_numpy(value['structure']),
                gt=torch.from_numpy(gt),support=torch.from_numpy(support))
class RealRotationDataset(Dataset):
    def __init__(self,records,phase,wrong):self.records,self.phase,self.wrong=records,phase,wrong
    def __len__(self):return len(self.records)
    def __getitem__(self,i):return reference_group(self.records[i],self.phase,self.wrong)
