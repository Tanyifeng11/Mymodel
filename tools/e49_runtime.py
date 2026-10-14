"""独立 CLIP 纹理编码路径：不加载 SD Writer，不改原 E5 CLIP forward。"""
from types import MethodType
import torch
from torch.nn import functional as F
from models.e49_style_encoder import FrozenStyleEncoder, StyleReferenceProcessor
from tools.e44_runtime import Runtime
from tools.e49_protocol import *


class StyleRuntime(Runtime):
    def __init__(self, arm):
        super().__init__('B0')
        assert arm in TRAIN_ARMS and self.width==640
        self.variant = arm
        self.style_encoder = FrozenStyleEncoder(self.pipe.image_encoder, arm)
        self.style_features = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.processor = StyleReferenceProcessor(self.original, LAYER, self.width,
                self.pipe.image_encoder.config.hidden_size).to(self.pipe.device)
        dict(self.pipe.unet.named_modules())[LAYER[:-10]].set_processor(self.processor)
        write(OUT/'manifests'/('model_'+arm+'.json'), dict(arm=arm, clip_config=self.pipe.image_encoder.config.to_dict(),
            trainable_parameters=sum(p.numel() for p in self.parameters()),
            frozen_modules=self.before, config=CONFIG, commit=commit()))

    @torch.no_grad()
    def set_reference(self, pixels, mask, mask_info, allow=True):
        self.pixel_reference=pixels; self.mask=mask; self.mask_info=mask_info; self.allow_reference=allow
        self.style_features = self.style_encoder(pixels) if allow and mask_info['valid'] else None

    def before_unet(self, module, args, kwargs):
        if self.processor is None: return
        p=self.processor
        p.branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
        p.enabled=bool(self.active and p.branch=='conditional' and self.allow_reference and self.mask_info['valid'])
        if not p.enabled: return
        latent=args[0] if args else kwargs['sample']
        p.mask=F.interpolate(self.mask, size=(latent.shape[-2]//2,latent.shape[-1]//2), mode='area')
        # 冻结编码缓存可以 detach；StyleEmb 必须在 student forward 内保留计算图。
        p.features=p.style_embed(self.style_features.float())

    def save(self, path, step, optimizer=None, scaler=None):
        path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(variant=self.variant, step=step, config=CONFIG, state_dict=self.processor.trainable_state(),
            optimizer=optimizer.state_dict() if optimizer else None, scaler=scaler.state_dict() if scaler else None), path)
        write(path.with_suffix('.json'), dict(step=step, sha256=sha(path), commit=commit()))

    def load(self, path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['config']==CONFIG and state['variant']==self.variant
        result=self.processor.load_state_dict(state['state_dict'],strict=False)
        assert not result.unexpected_keys and all(n.startswith('original.') for n in result.missing_keys)
        return state


def make_runtime(arm, namespace=None):
    from tools import e44_runtime as module
    from tools.e47_runtime import load_base
    from models import e33tm_generation_wrapper as loader
    original_loader, original_out=loader.load_e5,module.OUT
    module.OUT=OUT/(namespace or ('training/'+arm+'/runtime'))
    loader.load_e5=lambda:load_base('B')
    try: r=StyleRuntime(arm)
    finally: loader.load_e5=original_loader
    r.e47_original_out=original_out
    r.pipe.use_tcpm_lite=True
    return r


def close(r):
    from tools.e47_runtime import close as release
    release(r)
