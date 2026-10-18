"""M12 推理及真实投影后、加回前残差审计。"""
import csv
from pathlib import Path
import torch
from models.m12_fixed_dual_path import FixedDualPathProcessor
from tools.m54_runtime import M54Runtime
from tools.m54_protocol import LAYER, read, write
from tools.e44_runtime import digest


class DualRuntime(M54Runtime):
    def __init__(self, arm, alpha_b, alpha_m):
        super().__init__('B1')
        attention = dict(self.pipe.unet.named_modules())[LAYER[:-10]]
        self.processor = FixedDualPathProcessor(self.processor, attention, alpha_b, alpha_m)
        attention.set_processor(self.processor)
        self.arm = self.recipe = self.variant = arm
        self.effects_enabled = False
        self.source_hashes = self.path_hashes()

    def path_hashes(self):
        return {name: digest(value) for name, value in self.processor.state_dict().items() if not name.startswith('original.')}

    def parameters(self):
        return []

    def generate(self, row, path, **kwargs):
        self.processor.reset()
        latent_steps = []
        original_step = self.pipe.scheduler.step

        def capture(*args, **options):
            result = original_step(*args, **options)
            latent_steps.append(result[0].detach().cpu())
            return result

        self.pipe.scheduler.step = capture
        try:
            metadata, contexts, outputs = super().generate(row, path, **kwargs)
        finally:
            self.pipe.scheduler.step = original_step
        records = [dict(id=row['id'], arm=self.arm, **record) for record in self.processor.records]
        assert len(records) == 50 * 2 * 4 and len(latent_steps) == 50
        assert {branch: self.processor.steps[branch] for branch in ['conditional', 'unconditional']} == dict(conditional=50, unconditional=50)
        assert all(record['fusion_energy'] == 0 for record in records if record['branch'] == 'unconditional')
        path = Path(path)
        tensor_path = path.with_suffix('.pt')
        trace = torch.load(tensor_path, map_location='cpu', weights_only=False)
        trace['step_latents'] = torch.stack(latent_steps)
        torch.save(trace, tensor_path)
        from tools.m54_protocol import sha
        metadata.update(diagnostics_sha256=sha(tensor_path), alpha_b=self.processor.alpha_b,
            alpha_m=self.processor.alpha_m, delta_rms=sum(record['fusion_rms'] for record in records
                if record['branch'] == 'conditional' and record['region'] == 'all') / 50,
            source_branch_hashes=self.source_hashes, simultaneous_same_state=True)
        write(path.with_suffix('.json'), metadata)
        effects = [dict(step=record['step'], rms=record['fusion_rms']) for record in records
            if record['branch'] == 'conditional' and record['region'] == 'all']
        write(path.with_name(path.stem + '_effects.json'), effects)
        with path.with_name(path.stem + '_rms.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        assert self.path_hashes() == self.source_hashes
        return metadata, contexts, outputs


def make_runtime(arm, alpha_b, alpha_m, namespace):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    from tools.e47_runtime import load_base
    from tools.m12_experiment import RUN
    old_loader, old_out, old_layer = loader.load_e5, module.OUT, module.LAYER
    module.OUT = RUN / namespace
    module.LAYER = LAYER
    loader.load_e5 = lambda: load_base('B')
    try:
        runtime = DualRuntime(arm, alpha_b, alpha_m)
    finally:
        loader.load_e5, module.OUT, module.LAYER = old_loader, old_out, old_layer
    runtime.m54_runtime_out = RUN / namespace
    runtime.m54_layer = LAYER
    runtime.pipe.use_tcpm_lite = True
    return runtime
