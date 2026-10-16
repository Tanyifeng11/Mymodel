"""冻结原E44/E5实例，只分离Writer输入并缓存同参考同t的原始特征。"""
from types import MethodType
import gc
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e53_protocol import *

def runtime(namespace):
    from tools import e47_runtime as m
    old=m.OUT;m.OUT=OUT
    try:r=m.runtime('B',True,namespace)
    finally:m.OUT=old
    r.load(B1);r.processor.eval().requires_grad_(False);r.feature_cache={};r.latent_cache={};r.e53_row=None;r.writer_key=('N',0)
    r.original_set_reference=r.set_reference;r.original_write=r.writer.write
    def features_for(self,geometry,texture,timestep):
        key=(geometry,texture);t=int(timestep);cachekey=(key,t)
        if cachekey not in self.feature_cache:
            if key not in self.latent_cache:
                row=self.e53_row;path=row['writer_refs']['G%sT%d'%key] if row.get('writer_refs') else row['reference']
                image=Image.open(path).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
                pixels=to_tensor(image)[None].to(self.pipe.device)*2-1
                self.writer.set_reference(pixels,self.empty_text);self.latent_cache[key]=self.writer.latent
            self.writer.latent=self.latent_cache[key];self.writer.text=self.empty_text
            self.feature_cache[cachekey]=self.original_write(timestep).detach()
        return self.feature_cache[cachekey]
    r.features_for=MethodType(features_for,r)
    r.writer.write=lambda t:r.features_for(*r.writer_key,t)
    return r

def case(r,row,arm,stage,spec,energy_targets=None,full=False):
    from models.e53_oracle import OracleDiagnostic
    path=image_path(stage,arm,row['id']);done=path.with_name(path.stem+'_trace.json')
    if done.exists():
        data=read(done);assert sha(path)==data['generation']['png_sha256'];return data
    raw=OUT/'traces'/stage/arm/row['id']
    if raw.exists():
        # 无完成标记的张量来自中断条件；原样归档后从同一固定seed重跑。
        from datetime import datetime,timezone
        archived=OUT/'failures'/'partial_traces'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')/stage/arm/row['id']
        archived.parent.mkdir(parents=True,exist_ok=True);raw.rename(archived)
        write(archived.parent/(row['id']+'_archive.json'),dict(source=str(raw),archive=str(archived),reason='condition had no completed trace; preserve failed partial tensors before deterministic rerun',commit=commit()))
    r.e53_row=row;r.writer_key=(spec['geometry'],spec['texture']);r.active=arm!='E5'
    ctl=OracleDiagnostic(r,row,arm,dict(spec,stage=stage),energy_targets,full);ctl.install()
    try:record,_,_=r.generate(row,path)
    finally:ctl.close()
    if arm!='E5' and record['mask']['valid']:assert all(x['energy']['relative_error']<1e-4 for x in ctl.steps)
    record.update(e5_sha256=E5_SHA,b1_sha256=B1_SHA,base_reference_sha256=sha(row['reference']),writer_key=list(r.writer_key),code_commit=commit())
    torch.save(dict(latents_before=torch.stack(ctl.latents)),path.with_name(path.stem+'_latents.pt'))
    data=dict(id=row['id'],arm=arm,stage=stage,spec=spec,energy_targets=energy_targets,steps=ctl.steps,generation=record,total_injection_energy=sum(s['energy']['after']**2 for s in ctl.steps),commit=commit())
    write(path.with_suffix('.json'),record);write(done,data);print('E53 IMAGE',stage,arm,row['id'],round(record['seconds'],2),flush=True)
    return data

def clear(r):r.feature_cache.clear();r.latent_cache.clear();gc.collect();torch.cuda.empty_cache()
def release(r):
    from tools.e47_runtime import close
    close(r);gc.collect();torch.cuda.empty_cache()
