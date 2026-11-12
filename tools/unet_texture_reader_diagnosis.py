"""零更新检查已训练纹理读取的选择性、幅度与共同状态去噪响应。"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.e2_compat import load_trainable_strict
from models.e2_texture_diagnostics import TrajectoryTap, STEPS
from models.unet_texture_reader import ATTENTION, attach_reader, encode_reference
from tools.e2_texture_bottleneck import image, forward_at
from tools.e2_fft_experiment import setup, read, write, sha, environment
from tools.m1_evaluation import prepare_inference
from tools.m1_residual_followup import bundle
from tools import unet_spatial_texture as experiment


def rms(value):
    return float(value.float().square().mean().sqrt())


def relative(value, baseline):
    return rms(value.float() - baseline.float()) / max(rms(baseline), 1e-12)


def attention_stats(reader, hidden, original, grid):
    key, value = reader.keys_and_values(grid)
    query = reader.query(hidden)

    def heads(value):
        return value.float().unflatten(-1, (reader.heads, -1)).transpose(1, 2)

    q, k, v = map(heads, (query, key, value))
    logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    weights = logits.softmax(-1)
    context = (weights @ v).transpose(1, 2).flatten(2)
    residual = reader.output(context.to(hidden.dtype))
    pooled = reader.output(value.mean(1, keepdim=True)).expand_as(residual)
    return dict(q_rms=rms(query), k_rms=rms(key), v_rms=rms(value),
        centered_logit_std=float(logits.std(-1, unbiased=False).mean()),
        entropy_fraction=float((-(weights * weights.clamp_min(1e-20).log()).sum(-1) / math.log(key.shape[1])).mean()),
        top1_multiple_uniform=float(weights.amax(-1).mean() * key.shape[1]),
        residual_rms=rms(residual), base_attention_rms=rms(original),
        residual_to_attention=rms(residual) / max(rms(original), 1e-12),
        residual_difference_from_uniform=relative(residual, pooled),
        residual_spatial_variation=rms(residual - residual.mean(1, keepdim=True)) / max(rms(residual), 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=experiment.probe.PROJECT)
    parser.add_argument('--previous', type=Path, default=experiment.probe.PREVIOUS)
    parser.add_argument('--data', type=Path, default=experiment.probe.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--out', type=Path, default=experiment.RUN)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    core, out = setup(args)
    folder = out / 'diagnosis_20261112'
    checkpoint = out / 'training/Spatial/latest.pt'
    checkpoint_sha = sha(checkpoint)
    assert checkpoint_sha == read(out / 'training/Spatial/complete.json')['checkpoint_sha256']
    pipe, network = experiment.load_base(args, core)
    prepare_inference(args, pipe, network)
    reader = attach_reader(network, pipe.vae)
    load_trainable_strict(network, torch.load(checkpoint, map_location='cpu', weights_only=False)['network'])
    network.eval().requires_grad_(False)
    control_ids = list(read(out / 'protocol.json')['controls'])[:2]
    pairs = [p for p in read(out / 'strict_pairs.json')['pairs'] if p['case_id'] in control_ids]
    records, roundtrips = [], []
    with torch.inference_mode():
        for pair in pairs:
            references = {name: core.tensor(experiment.probe.control_path(out, pair, name)).to(args.device).half()
                          for name in experiment.CONTROLS}
            grids = {name: encode_reference(pipe.vae, reference) for name, reference in references.items()}
            for name, grid in grids.items():
                decoded = pipe.vae.decode(grid / pipe.vae.config.scaling_factor).sample
                pixels = ((decoded[0].float() + 1) * 127.5).clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()
                path = folder / 'vae_roundtrip' / (pair['case_id'] + '_' + name + '.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
                roundtrips.append(dict(case_id=pair['case_id'], reference=name,
                    bands=experiment.probe.patch_bands(pixels[:64, :64])))
            sketch = core.tensor(args.data / pair['sketch']).to(references['v16'])
            features = experiment.probe.projected(network, sketch, references['v16'])
            with TrajectoryTap(pipe, network.adapter) as trajectory:
                image(args, core, pipe, network, pair, features)
            assert trajectory.condition_calls == 50 and set(trajectory.states) == set(STEPS)
            for step, state in trajectory.states.items():
                captured = []

                def capture(_module, inputs, kwargs, output):
                    if network.adapter.conditional:
                        captured.append((inputs[0] if inputs else kwargs['hidden_states'], output))

                hook = network.unet.get_submodule(ATTENTION).register_forward_hook(capture, with_kwargs=True)
                baseline = forward_at(pipe, network, state, features)
                hook.remove()
                assert torch.equal(baseline, state['conditional']['epsilon']) and len(captured) == 1
                hidden, original = captured[0]
                predictions = {}
                for name, grid in grids.items():
                    stats = attention_stats(reader, hidden, original, grid)
                    with reader.inference(grid):
                        predicted = forward_at(pipe, network, state, features)
                        predictions[name] = predicted
                        key, value = reader.cached
                        reader.cached = (key.mean(1, keepdim=True), value.mean(1, keepdim=True))
                        uniform = forward_at(pipe, network, state, features)
                    records.append(dict(case_id=pair['case_id'], step=step, timestep=state['timestep'],
                        reference=name, **stats, epsilon_change_from_reader=relative(predicted, baseline),
                        epsilon_difference_from_uniform=relative(predicted, uniform)))
                records.append(dict(case_id=pair['case_id'], step=step, timestep=state['timestep'],
                    intervention='reader_only_reference_swap_bfm_fixed',
                    h16_epsilon_delta=rms(predictions['h16'] - predictions['v16']) / max(rms(baseline), 1e-12),
                    v32_epsilon_delta=rms(predictions['v32'] - predictions['v16']) / max(rms(baseline), 1e-12)))
            print('READER_DIAG_CASE', pair['case_id'], flush=True)
    assert sha(checkpoint) == checkpoint_sha
    split = read(out / 'split.json')
    write(folder / 'result.json', dict(checkpoint_sha256=checkpoint_sha, generator_updates=0,
        cases=control_ids, training_samples=len(split['train_indices']), global_sample_presentations=600 * 8,
        records=records, vae_roundtrips=roundtrips, environment=environment()))
    bundle(out, 'reader_diagnosis_results.zip', (folder,))
    print('READER_DIAG_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
