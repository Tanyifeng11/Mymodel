"""已批准C-R筛选：固定C1/C2仅推理；C3/C4/C5/C6独立同预算续训。"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from models.e2_compat import bare_network, compute_losses, cpu_snapshot, compare_snapshots, load_trainable_strict
from models.e2_region_router import attach_router, RegionRouter, VERSION, LEARNED
from tools.e2_fft_experiment import SOURCE, setup, load, read, write, sha, environment
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash, optimizer_hash, frozen_hash
from tools.e2_a_minimal import mask_snapshot


def c_environment():
    value = environment()
    for name in ['models/e2_structure.py', 'models/e2_region_router.py', 'tools/e2_c_screen.py',
                 'tools/e2_c_evaluation.py', 'scripts/e2_c_screen.sbatch', 'tests/test_e2_region_router.py',
                 'tests/test_e2_c_metrics.py']:
        value['source'][name] = sha(SOURCE / name)
    return value


def config(args, arm):
    cfg = dict(read(Path(args.baseline) / 'comparison/B0.json'))
    cfg.update(arm=arm, protocol='C_R_screen_v1', code_commit=environment()['commit'],
        A=False, B=False, C=True, c_mode=arm, router_width=16, router_warmup=100,
        router_kappa=.5, distance_w=4., distance_tau=1., router_version=VERSION,
        fft=False, fft_weight=0., updates=2000, effective_batch=8, world_size=1,
        split=str(Path(args.out) / 'split.json'), split_sha256=sha(Path(args.out) / 'split.json'))
    cfg.pop('auxiliary_weights', None)
    return cfg


def prepare(args):
    from tools.e2_b_screen_training import audit_training
    core, out = setup(args); baseline = Path(args.baseline)
    complete = read(baseline / 'training/B0/complete.json')
    assert complete['status'] == 'completed' and complete['updates'] == 2000
    assert sha(baseline / 'training/B0/latest.pt') == complete['checkpoint_sha256']
    assert read(baseline / 'audit/training_inputs_final.json')['passed']
    audit_training(args); mask_snapshot(args, core)
    assert sha(out / 'audit/training_inputs.json') == sha(baseline / 'audit/training_inputs.json')
    for name in ['strict_pairs.json', 'inputs/registry.json', 'audit/input_version_comparison.json']:
        dest = out / name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert dest.read_bytes() == (baseline / name).read_bytes()
        else: shutil.copy2(baseline / name, dest)
    provenance = []
    for pair in read(out / 'strict_pairs.json')['pairs']:
        for key in ['sketch', 'gt', 'original_reference', 'swapped_reference']:
            src = Path(args.data) / pair[key]
            assert sha(src) == pair['hashes'][key]
            dest = out / 'inputs/frozen' / pair[key]; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        for condition in ['old', 'new']:
            src = baseline / 'outputs/B0' / pair['case_id'] / (condition + '.png')
            meta = read(src.with_suffix('.json'))
            assert sha(src) == meta['output_sha256'] and meta['checkpoint_sha256'] == complete['checkpoint_sha256']
            assert meta['manifest_sha256'] == sha(out / 'strict_pairs.json')
            dest = out / 'outputs/C0' / pair['case_id'] / src.name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            meta.update(arm='C0', reused_from_arm='B0', original_metadata_sha256=sha(src.with_suffix('.json')))
            write(dest.with_suffix('.json'), meta)
            provenance.append(dict(file=str(src), output_sha256=sha(src), metadata_sha256=meta['original_metadata_sha256']))
    folder = out / 'training/C0'; folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline / 'training/B0/training.jsonl', folder / 'training.jsonl')
    write(folder / 'complete.json', dict(complete, alias='C0', reused_from=str(baseline / 'training/B0')))
    write(out / 'audit/baseline_reuse.json', dict(checkpoint=str(baseline / 'training/B0/latest.pt'),
        checkpoint_sha256=complete['checkpoint_sha256'], records=provenance,
        required_gate='alloff/eta0 exact forward+image and first4update B0 replay', prepared=True))
    parameters = sum(p.numel() for p in RegionRouter().parameters())
    for arm in LEARNED: write(out / 'comparison' / (arm + '.json'), config(args, arm))
    write(out / 'protocol.json', dict(version='C_R_screen_v1', router_version=VERSION,
        probe_arms=['C0', 'C1', 'C2'], learned_arms=['C0'] + list(LEARNED),
        probe='same completed B0 weights; no new training; not a trained C1/C2 ablation',
        training='common matureE2 initialization,continuation2000/effectivebatch8/world1/seed42; C0 audited B0 reuse',
        A=False, B=False, region_loss=False, time_gate=False, added_parameters=parameters,
        settings=dict(width=16, eta='min(completed_optimizer_updates/100,1)', kappa=.5, w=4., tau=1.,
            eps_g=.15, kappa_s=.5, b0=.2),
        structure_source='old sketch convolution weights,independent target-sketch forward without reference FiLM; no A checkpoint',
        geometry='fixed input-sketch maps; EDT pixel center +/-1,clip32; old9x9 body unchanged; noGT conditions',
        primary='v3 gray_strict_response_change,C3/C4/C5/C6 minus C0,Holm family4',
        protections=['old_boundary_f1_r2','old_iou','old_lpips','new_boundary_f1_r2','new_iou',
                     'new_leak','new_background_mae','new_band_edge_f1'],
        protection_holm_family=32, noninferiority_claim=False,
        limitation='32 texture-focused targets,23stripes/3checks/6dots,single training seed; internal candidates not semantic seams',
        environment=c_environment()))
    print('C_PREPARE_PASS', parameters, flush=True)


def load_c(args, core):
    pipe, network, checked = load(args, core)
    attach_router(network, mode=args.arm)
    return pipe, network, checked


def regression(args):
    from PIL import Image
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args); pipe, network, _ = load(args, core)
    state = torch.load(Path(args.baseline) / 'training/B0/latest.pt', map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network']); del state
    network.eval(); style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    data = dataset(pipe, core); choices = read(out / 'split.json')['smoke']; before = []
    for item in choices:
        batch = {k: v[None].to(args.device) for k, v in data[item['index']].items()}
        rng = torch.Generator(device=args.device).manual_seed(42000 + item['index'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            total, raw, token, captured, _ = compute_losses(pipe, network, batch, rng, style, core, audit_mode=True)
        before.append(cpu_snapshot(raw, token, captured, total, rng)); network.adapter.clear()
        del total, raw, captured
    adapter = attach_router(network, 'C6'); network.eval(); comparisons = []
    for enabled, eta in [(False, 1.), (True, 0.)]:
        adapter.enabled, adapter.eta = enabled, eta
        for item, expected in zip(choices, before):
            batch = {k: v[None].to(args.device) for k, v in data[item['index']].items()}
            rng = torch.Generator(device=args.device).manual_seed(42000 + item['index'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                total, raw, token, captured, _ = compute_losses(pipe, network, batch, rng, style, core, audit_mode=True)
            errors = compare_snapshots(expected, cpu_snapshot(raw, token, captured, total, rng), atol=0, rtol=0)
            comparisons.append(dict(enabled=enabled, eta=eta, index=item['index'], errors=errors)); adapter.clear()
            del total, raw, captured
    write(out / 'regression/forward_result.json', dict(passed=True, comparisons=comparisons))
    del style, data, before
    pipe.unet.half(); pipe.vae.half(); adapter.half()
    pipe.image_encoder.to(args.device); pipe.bf_texture_conditioner.to(args.device)
    records = []
    with torch.inference_mode():
        for enabled, eta in [(False, 1.), (True, 0.)]:
            adapter.enabled, adapter.eta = enabled, eta
            for pair in read(out / 'strict_pairs.json')['pairs'][:4]:
                adapter.condition(core.tensor(Path(args.data) / pair['sketch']).to(args.device).half(),
                                  core.tensor(Path(args.data) / pair['original_reference']).to(args.device).half())
                latent = torch.randn((1, 4, 64, 48), device=args.device, dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                image = pipe(ref_image=core.tensor(Path(args.data) / pair['sketch']).to(args.device).half(),
                    prompt=pair['caption'], texture_clip_image=None, texture_embeds=None, null_prompt='',
                    negative_prompt=' worst quality, low quality', width=384, height=512, num_images_per_prompt=1,
                    guidance_scale=7., sketch_scale=.6, ipa_scale=0., latents=latent,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']), num_inference_steps=50)[0]
                expected = Image.open(out / 'outputs/C0' / pair['case_id'] / 'old.png').convert('RGB')
                error = int(np.abs(np.asarray(image).astype(int) - np.asarray(expected).astype(int)).max())
                dest = out / 'regression' / ('alloff' if not enabled else 'eta0') / (pair['case_id'] + '.png')
                dest.parent.mkdir(parents=True, exist_ok=True); image.save(dest)
                records.append(dict(enabled=enabled, eta=eta, case_id=pair['case_id'], max_pixel_difference=error))
                assert error == 0; adapter.clear()
    write(out / 'regression/image_result.json', dict(passed=True, records=records, environment=c_environment()))
    print('C_REGRESSION_PASS', flush=True)


class CTrainingRun(TrainingRun):
    def update(self):
        adapter = bare_network(self.network).adapter
        adapter.eta = min(self.step / self.config['router_warmup'], 1.)
        adapter.records = []
        row = super().update()
        row['routing'], row['eta'] = adapter.records, adapter.eta
        assert len(row['routing']) == 8
        return row


def smoke(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    assert read(out / 'regression/forward_result.json')['passed'] and read(out / 'regression/image_result.json')['passed']
    assert read(out / 'diagnostics/fixed_complete.json')['passed']
    pipe, network, _ = load(args, core); initial = network.trainable_state()
    adapter = attach_router(network, 'C6', enabled=False); adapter.router.requires_grad_(False)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False); data = dataset(pipe, core)
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8); assert acc.num_processes == 1
    cfg = read(Path(args.baseline) / 'training/B0/complete.json')['config']
    old = [json.loads(x) for x in (out / 'training/C0/training.jsonl').read_text().splitlines()]
    run = TrainingRun(args, core, pipe, network, data, style, cfg, read(out / 'split.json')['train_indices'], acc)
    records = []
    for expected in old[:4]:
        actual = run.update()
        assert actual['tokens'] == expected['tokens'] and actual['losses'] == expected['losses'] and actual['grad_norm'] == expected['grad_norm']
        records.append(dict(step=actual['step'], tokens_equal=True, losses_equal=True, gradient_norm_equal=True))
    write(out / 'audit/baseline_replay.json', dict(passed=True, records=records))
    network = acc.unwrap_model(run.network, keep_fp32_wrapper=False); del run; acc.free_memory()
    results = {}
    for arm in LEARNED:
        adapter = attach_router(network, arm)
        state = network.trainable_state(); state.update(initial); load_trainable_strict(network, state); del state
        before = network.trainable_state(); frozen = frozen_hash(pipe, network)
        cfg = dict(read(out / 'comparison' / (arm + '.json')), updates=4, warmup=0, workers=0, diagnostics=True)
        run = CTrainingRun(args, core, pipe, network, data, style, cfg,
            [x['index'] for x in read(out / 'split.json')['smoke']], acc)
        logs = [run.update(), run.update()]; folder = out / 'smoke' / arm; folder.mkdir(parents=True, exist_ok=True)
        run.save(folder / 'resume.pt'); expected = run.update()
        expected_state = acc.unwrap_model(run.network).trainable_state(); expected_optimizer = optimizer_hash(run.optimizer)
        run.restore(folder / 'resume.pt'); actual = run.update()
        assert tensor_hash(expected_state.items()) == tensor_hash(acc.unwrap_model(run.network).trainable_state().items())
        assert expected_optimizer == optimizer_hash(run.optimizer)
        assert actual['tokens'] == expected['tokens'] and actual['losses'] == expected['losses'] and actual['routing'] == expected['routing']
        logs.extend([actual, run.update()]); network = acc.unwrap_model(run.network)
        changes = {}
        for name, value in network.trainable_state().items():
            group = 'old_E2' if not name.startswith('adapter.router.') else name.split('.')[2]
            changes[group] = changes.get(group, 0.) + float((value.float() - before[name].float()).square().sum())
        assert all(np.isfinite(v) and v > 0 for v in changes.values()) and frozen_hash(pipe, network) == frozen
        batch = {k: v[None].to(args.device) for k, v in data[read(out / 'split.json')['smoke'][0]['index']].items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, raw, _, captured, _ = compute_losses(pipe, network, batch,
                torch.Generator(device=args.device).manual_seed(42000), style, core, audit_mode=True)
            image_loss = raw['gram'] + .05 * raw['edge'] + .1 * raw['color'] + .1 * raw['body'] + .03816244801608925 * raw['rest']
        router = network.adapter.router
        targets = [captured['epsilon'], captured['vae_input'], router.structure[0].weight,
                   router.gates[0][-1].weight, router.reduce_s[0].weight, router.reduce_u[0].weight]
        gradients = torch.autograd.grad(image_loss, targets, allow_unused=True)
        norms = {name: float(g.float().norm()) if g is not None else 0. for name, g in zip(
            ['epsilon', 'vae_input', 'C_structure', 'C_gate_head', 'C_reduce_s', 'C_reduce_u'], gradients)}
        assert all(np.isfinite(v) and v > 0 for v in norms.values())
        results[arm] = dict(passed=True, updates=4, executed_update_calls=5, resume_exact=True,
            frozen_unchanged=True, changes_squared_l2=changes, image_loss_gradient_norms=norms, logs=logs)
        write(folder / 'result.json', results[arm])
        network = acc.unwrap_model(run.network, keep_fp32_wrapper=False)
        del run, expected_state, before, captured, raw, image_loss, gradients, targets, batch; acc.free_memory()
        print('C_SMOKE_PASS', arm, flush=True)
    write(out / 'smoke/result.json', dict(passed=True, arms=results, environment=c_environment()))


def train(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    from tools.e2_b_screen_training import assert_snapshot
    core, out = setup(args)
    assert args.arm in LEARNED and read(out / 'smoke/result.json')['passed']
    assert read(out / 'audit/baseline_replay.json')['passed'] and read(out / 'diagnostics/fixed_complete.json')['passed']
    cfg = read(out / 'comparison' / (args.arm + '.json'))
    assert cfg['code_commit'] == environment()['commit'] and assert_snapshot(args) == cfg['training_input_map_sha256']
    assert sha(out / 'split.json') == cfg['split_sha256']
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8); assert acc.num_processes == 1
    pipe, network, _ = load_c(args, core); data = dataset(pipe, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    folder = out / 'training' / args.arm; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        complete = read(folder / 'complete.json')
        assert complete['config'] == cfg and complete['checkpoint_sha256'] == sha(folder / 'latest.pt')
        print('C_ALREADY_COMPLETED', args.arm); return
    run = CTrainingRun(args, core, pipe, network, data, style, cfg, read(out / 'split.json')['train_indices'], acc)
    if (folder / 'latest.pt').exists(): run.restore(folder / 'latest.pt')
    log = folder / 'training.jsonl'
    if log.exists():
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        log.write_text(''.join(json.dumps(x) + '\n' for x in rows if x['step'] <= run.step))
    baseline = [json.loads(x) for x in (out / 'training/C0/training.jsonl').read_text().splitlines()]
    while run.step < cfg['updates']:
        row = run.update(); assert row['tokens'] == baseline[row['step'] - 1]['tokens']
        with log.open('a', encoding='utf8') as stream: stream.write(json.dumps(row, allow_nan=False) + '\n')
        if run.step <= 4 or run.step % 25 == 0: print('C_TRAIN', args.arm, run.step, row['losses'], row['eta'], flush=True)
        if run.step % 100 == 0 or run.step == cfg['updates']: run.save(folder / 'latest.pt')
    write(folder / 'complete.json', dict(status='completed', updates=run.step, config=cfg,
        environment=c_environment(), checkpoint_sha256=sha(folder / 'latest.pt')))
    print('C_TRAIN_COMPLETED', args.arm, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'regression', 'smoke', 'train', 'audit-final', 'generate', 'evaluate', 'bundle'])
    p.add_argument('--project', default='/share/home/u2515283058/Mymodel')
    p.add_argument('--baseline', default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--out', default='/share/home/u2515283058/Mymodel/output_eval/e2_c_screen_20261109')
    p.add_argument('--data', default='/share/home/u2515283058/datasets/BF')
    p.add_argument('--candidates', default=str(SOURCE / 'configs/e2_b_candidates.json'))
    p.add_argument('--device', default='cuda:0'); p.add_argument('--arm', default='C6', choices=['C0','C1','C2'] + list(LEARNED))
    p.add_argument('--phase', choices=['fixed', 'learned'], default='learned')
    args = p.parse_args(); args.parent = args.baseline
    if args.action in ['generate', 'evaluate', 'bundle']:
        from tools import e2_c_evaluation
        getattr(e2_c_evaluation, args.action)(args)
    elif args.action == 'audit-final':
        from tools.e2_b_screen_training import audit_training
        audit_training(args, final=True); core, _ = setup(args); mask_snapshot(args, core, final=True)
    else: globals()[args.action](args)


if __name__ == '__main__': main()
