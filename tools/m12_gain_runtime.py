"""只读加载 step2400；诊断与冻结检查沿用 M12。"""
import torch
from models.m54_m10_gain import M10Gain
from models.m12_gain_dual_path import GainDualPathProcessor
from tools.m12_runtime import DualRuntime
from tools.m54_runtime import M54Runtime
from tools.m54_protocol import LAYER, read, sha


def load_gain(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert sha(path) == read(path.with_suffix('.json'))['sha256']
    assert state['arm'] == 'M10_GAIN' and state['step'] == 2400
    from tools.m54_m10_gain_runtime import RUN as GAIN_RUN
    assert state['protocol'] == read(GAIN_RUN/'protocol.json')
    gain = M10Gain().to(device)
    gain.load_state_dict(state['state_dict'], strict=True)
    assert sum(p.numel() for p in gain.parameters()) == 1280
    assert all(torch.isfinite(p).all() for p in gain.parameters())
    return gain.eval().requires_grad_(False)


class GainDualRuntime(DualRuntime):
    def __init__(self, checkpoint, alpha_b=.25, alpha_m=.5):
        M54Runtime.__init__(self, 'B1')
        attention = dict(self.pipe.unet.named_modules())[LAYER[:-10]]
        gain = load_gain(checkpoint, self.pipe.device)
        self.processor = GainDualPathProcessor(self.processor, attention, alpha_b, alpha_m, gain)
        attention.set_processor(self.processor)
        self.arm = self.recipe = self.variant = 'M12_GAIN'
        self.effects_enabled = False
        self.source_hashes = self.path_hashes()


def make_runtime(namespace, alpha_b=.25, alpha_m=.5, native=None):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    from tools.e47_runtime import load_base
    from tools.m12_gain_experiment import RUN, GAIN_CHECKPOINT
    previous = loader.load_e5, module.OUT, module.LAYER
    loader.load_e5 = lambda: load_base('B')
    module.OUT, module.LAYER = RUN/namespace, LAYER
    try:
        if native == 'M10_GAIN':
            from tools.m54_m10_gain_runtime import GainRuntime
            runtime = GainRuntime()
            runtime.load(GAIN_CHECKPOINT)
            runtime.gain.eval().requires_grad_(False)
        elif native:
            runtime = M54Runtime(native)
        else:
            runtime = GainDualRuntime(GAIN_CHECKPOINT, alpha_b, alpha_m)
    finally:
        loader.load_e5, module.OUT, module.LAYER = previous
    runtime.effects_enabled = False
    runtime.m54_runtime_out, runtime.m54_layer = RUN/namespace, LAYER
    runtime.pipe.use_tcpm_lite = True
    return runtime
