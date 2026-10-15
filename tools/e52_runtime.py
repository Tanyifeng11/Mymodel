"""原 E44 生成器的最小适配：E5 原参考固定，Writer 输入单独设置。"""
import gc
from types import MethodType
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e52_protocol import *

def runtime(namespace):
    from tools import e47_runtime as m
    old=m.OUT;m.OUT=OUT
    try:r=m.runtime('B',True,namespace)
    finally:m.OUT=old
    r.load(B1);r.processor.eval().requires_grad_(False)
    r.writer_override=None;r.original_set_reference=r.set_reference
    def separate(self,pixels,mask,mask_info,allow=True):
        if self.writer_override is not None:
            image=Image.open(self.writer_override).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            pixels=to_tensor(image)[None].to(self.pipe.device)*2-1
        self.original_set_reference(pixels,mask,mask_info,allow)
    r.set_reference=MethodType(separate,r)
    return r

def release(r):
    from tools.e47_runtime import close
    close(r);gc.collect();torch.cuda.empty_cache()

def case(r,row,arm,stage,writer=None,teacher_path=None,mapping=None,diagnose=True,patch=False):
    from models.e52_diagnostics import Diagnostic
    path=image_path(stage,arm,row['id']);done=path.with_name(path.stem+'_trace.json')
    if done.exists():
        value=read(done);assert sha(path)==value['generation']['png_sha256']
        assert value['base_reference_sha256']==sha(row['reference'])
        assert value['writer_reference_sha256']==sha(writer or row['reference'])
        return value
    r.active=arm not in ['E5','E5_ALT'];r.writer_override=writer
    teacher=torch.load(teacher_path,map_location='cpu',weights_only=False) if teacher_path else None
    ctl=Diagnostic(r,row,arm,teacher,mapping,patch) if diagnose else None
    if ctl:ctl.install()
    try:record,_,_=r.generate(row,path)
    finally:
        if ctl:ctl.close()
        r.writer_override=None;r.active=True
    record.update(base_reference_sha256=sha(row['reference']),writer_reference_sha256=sha(writer or row['reference']),
        e5_sha256=E5_SHA,b1_sha256=B1_SHA,code_commit=commit())
    write(path.with_suffix('.json'),record)
    if ctl:torch.save(dict(latents=torch.stack(ctl.latents),steps=ctl.steps,attention_records=ctl.attention_records),path.with_name(path.stem+'_states.pt'))
    value=dict(id=row['id'],arm=arm,stage=stage,generation=record,
        base_reference_sha256=sha(row['reference']),writer_reference_sha256=sha(writer or row['reference']),
        teacher_path=str(teacher_path) if teacher_path else None,
        steps=ctl.steps if ctl else [],topology=ctl.topology if ctl else [],commit=commit())
    write(done,value);print('E52 IMAGE',stage,arm,row['id'],round(record['seconds'],2),flush=True)
    return value
