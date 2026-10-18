"""新训练臂独立保存强度参数，原M54/M10检查点保持冻结。"""
from pathlib import Path
import torch
from models.m54_m10_gain import M10Gain
from tools.m54_runtime import M54Runtime
from tools.m54_protocol import OUT, LAYER, read, write, sha, commit

RUN = OUT/'m10_gain_r1'
ARM = 'M10_GAIN'


class GainRuntime(M54Runtime):
    def __init__(self):
        super().__init__('M10')
        self.arm=self.recipe=self.variant=ARM
        self.processor.mode=ARM
        self.gain=M10Gain(self.width).to(self.pipe.device)
        self.processor.m10_gain=self.gain
        assert self.width==640 and sum(p.numel() for p in self.parameters())==1280

    def parameters(self):
        return list(self.gain.parameters()) if hasattr(self,'gain') else []

    def save(self,path,step,optimizer=None,scaler=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(arm=ARM,step=step,protocol=read(RUN/'protocol.json'),
            state_dict=self.gain.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,
            scaler=scaler.state_dict() if scaler else None),path)
        write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))

    def load(self,path):
        path=Path(path);assert sha(path)==read(path.with_suffix('.json'))['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['arm']==ARM and state['protocol']==read(RUN/'protocol.json')
        self.gain.load_state_dict(state['state_dict']);return state


def make_runtime(namespace):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    from tools.e47_runtime import load_base
    old_loader,old_out,old_layer=loader.load_e5,module.OUT,module.LAYER
    module.OUT=RUN/namespace;module.LAYER=LAYER;loader.load_e5=lambda:load_base('B')
    try:r=GainRuntime()
    finally:loader.load_e5=old_loader;module.OUT=old_out;module.LAYER=old_layer
    r.m54_runtime_out=RUN/namespace;r.m54_layer=LAYER;r.pipe.use_tcpm_lite=True
    return r
