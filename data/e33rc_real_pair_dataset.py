"""E32离线输入白名单；GT仅参与loss，rot90缓存只允许评测访问。"""
import numpy as np
import torch
from torch.utils.data import Dataset
from data.e32_field_dataset import cache_path
from tools.e33rc_common import E32

def orientation_reference(reference):
    assert reference.shape[-1]==395
    # 384:386是轴向方向，386是log-frequency，387是confidence。
    return np.concatenate([reference[...,:386],reference[...,387:]],-1).astype(np.float32)

def cached_real(row,evaluation=False):
    with np.load(cache_path(E32,row['id'])) as z:
        value={k:np.asarray(z[k],np.float32) for k in
            ['structure','supervision_geometry','supervision_interior','reference_histogram']}
        value['reference']=orientation_reference(z['reference'])
        if evaluation:value['reference_rot90']=orientation_reference(z['reference_rot90'])
    return value

class RealPairDataset(Dataset):
    def __init__(self,records,wrong):self.records,self.wrong=records,wrong
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index];case=cached_real(row)
        refs=[case['reference']]
        for key in ('color_near','random'):
            with np.load(cache_path(E32,self.wrong[row['id']][key])) as z:
                refs.append(orientation_reference(z['reference']))
        gt=case['supervision_geometry'];mask=(case['supervision_interior']>=.95)&(gt[3]>=.25)
        return dict(id=row['id'],reference=torch.from_numpy(np.stack(refs)),
            structure=torch.from_numpy(case['structure']),gt=torch.from_numpy(gt),support=torch.from_numpy(mask))
