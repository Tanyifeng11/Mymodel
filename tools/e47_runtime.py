"""沿用 E44 算子和 Writer；只显式选择 A/E5，避免把 E5 教师接到 C。"""
import argparse
from pathlib import Path
from types import MethodType

import torch
from tools.e47_protocol import *


def load_base(base):
    from tools.e15_d5_generation import build_inference_args,load_inference_module
    root=Path.cwd();checkpoint=E2B if base=='A' else E5
    args=argparse.Namespace(checkpoint=str(root/checkpoint),texture_ckpt=str(root/checkpoint),
        base_model_path=str(root/'models/stable-diffusion-v1-5'),
        vae_model_path=str(root/'models/stable-diffusion-v1-5/vae'),
        clip_model=str(root/'models/clip'),device='cuda',seed=42,steps=50)
    ns=build_inference_args(args);ns.use_tcpm_lite=int(base!='A')
    pipe,_=load_inference_module().prepare(ns)
    modules={k:getattr(pipe,k) for k in ['unet','reference_unet','bf_texture_conditioner',
        'tcpm_lite','vae','text_encoder','image_encoder'] if getattr(pipe,k,None) is not None}
    for module in modules.values():module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe,modules,(ns.width,ns.height),ns


def runtime(base='B',b1=False,namespace='p0/runtime'):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    original_loader=loader.load_e5
    original_out=module.OUT
    module.OUT=OUT/namespace
    loader.load_e5=lambda:load_base(base)
    try:r=module.Runtime('B1' if b1 else 'B0')
    finally:loader.load_e5=original_loader
    # close() 也会写 shape/frozen 清单；同进程只运行一个实例。
    r.e47_base=base;r.e47_original_out=original_out
    r.pipe.use_tcpm_lite=base!='A'
    write(OUT/namespace/'base_selection.json',dict(base=base,checkpoint=str(E2B if base=='A' else E5),
        tcpm_enabled=r.pipe.use_tcpm_lite,b1=b1,
        checkpoint_sha256=sha(E2B if base=='A' else E5)))
    return r


class TrainingInputs:
    def __init__(self,r):
        from tools import e44_train as original
        old=original.OUT;original.OUT=OUT
        try:self.inputs=original.TrainingInputs(r,'train1024')
        finally:original.OUT=old
        self.r=r

    def next(self,disable_dropout=False):
        p=self.r.pipe
        old=p.tcpm_lite.forward
        if not p.use_tcpm_lite:
            p.tcpm_lite.forward=MethodType(lambda _self,tokens,text:tokens,p.tcpm_lite)
        try:args,kw,noise,mask,record=self.inputs.next(disable_dropout)
        finally:p.tcpm_lite.forward=old
        if not p.use_tcpm_lite:kw['cross_attention_kwargs'].pop('tcpm_garment_mask',None)
        return args,kw,noise,mask,record


def exact_trace(new,old):
    a=torch.load(new,map_location='cpu',weights_only=False)
    b=torch.load(old,map_location='cpu',weights_only=False)
    keys=['initial_latent','final_latent','conditional_epsilon','unconditional_epsilon']
    result={k:dict(exact=bool(torch.equal(a[k],b[k])),
        max_abs_delta=float((a[k].float()-b[k].float()).abs().max())) for k in keys}
    return dict(fields=result,bit_exact=all(v['exact'] for v in result.values()))


def close(r):
    from tools import e44_runtime as module
    r.close();module.OUT=r.e47_original_out
