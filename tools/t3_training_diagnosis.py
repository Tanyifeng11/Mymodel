"""T3 历史日志与实际留存权重的训练诊断；不启动优化器更新。"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
NAME = 't3_training_diagnosis_20261030'
REST_WEIGHT = .03816244801608925


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def sup_root(project):
    root = project/'output_eval/e2a_sup_comp_01_20261021'
    return root/'review_results_20261026' if (root/'review_results_20261026').exists() else root


def logs(project):
    root = project/'output_eval'/NAME
    training = sup_root(project)/'training/formal/T3'
    config = read(training/'config.json')
    first, second = csv_read(training/'training_log.csv'), csv_read(training/'auxiliary_log.csv')
    steps = [int(r['update']) for r in first]
    assert steps == list(range(1, config['updates']+1))
    assert steps == [int(r['update']) for r in second]
    records = []
    for a, b in zip(first, second):
        assert a['job'] == b['job']
        base = float(a['denoise'])+float(a['gram'])+.05*float(a['edge'])+.1*float(a['rgb_global'])
        body, rest = float(b['body_rgb']), float(b['rest'])
        rec = dict(update=int(a['update']), base=base, body_rgb=body, rest=rest,
                   weighted_body=.1*body, weighted_rest=REST_WEIGHT*rest,
                   weighted_aux=float(b['weighted_aux']), total=float(a['total']),
                   lr=float(a['lr']), grad_norm=float(a['grad_norm']), job=a['job'])
        assert all(np.isfinite(v) for v in rec.values() if isinstance(v, (int, float)))
        records.append(rec)
    total_error = max(abs(r['total']-r['base']-r['weighted_aux']) for r in records)
    aux_error = max(abs(r['weighted_aux']-r['weighted_body']-r['weighted_rest']) for r in records)
    assert total_error < 1e-6 and aux_error < 1e-7
    csv_write(root/'training/aligned_losses.csv', records)
    provenance = []
    for p in list(training.glob('*.csv'))+list(training.glob('*.json'))+list((training/'pauses').glob('*.json')):
        target = root/'history/formal_T3'/p.relative_to(training)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(target))
        provenance.append(dict(source=str(p), saved=str(target.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p)))
    csv_write(root/'audit/history_files.csv', provenance)
    bins = []
    for start in range(0, len(records), 500):
        part = records[start:start+500]
        row = dict(first_update=part[0]['update'], last_update=part[-1]['update'], n=len(part))
        for k in ['base', 'body_rgb', 'rest', 'weighted_body', 'weighted_rest', 'total', 'lr', 'grad_norm']:
            values = [r[k] for r in part]
            row[k+'_mean'], row[k+'_median'] = float(np.mean(values)), float(np.median(values))
        bins.append(row)
    csv_write(root/'training/loss_bins500.csv', bins)
    sections = []
    for name, start, end in [('first1000', 1, 1000), ('first20pct', 1, 5673),
                             ('middle20pct', 11347, 17019), ('last20pct', 22693, 28365),
                             ('last1000', 27366, 28365)]:
        part = records[start-1:end]
        sections.append(dict(window=name, first_update=start, last_update=end,
                             **{k: float(np.mean([r[k] for r in part])) for k in ['base', 'body_rgb', 'rest', 'total', 'lr']}))
    csv_write(root/'training/phase_loss_summary.csv', sections)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    jobs = [r['update'] for i, r in enumerate(records) if i and r['job'] != records[i-1]['job']]
    for ax, key, title in zip(axes[:3], ['base', 'body_rgb', 'rest'],
                              ['基础损失 L_base', '衣内参考RGB统计损失 L_bodyRGB（原始值）', '其余区域GT重建损失 L_rest（原始值）']):
        ax.plot(steps, [r[key] for r in records], color='#9ab9cf', alpha=.16, lw=.5, label='逐更新原始值')
        ax.plot([(r['first_update']+r['last_update'])/2 for r in bins], [r[key+'_mean'] for r in bins],
                color='#005b96', lw=1.8, label='固定500更新分箱均值')
        for v in jobs:
            ax.axvline(v, color='gray', ls=':', lw=.8)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
    axes[3].plot(steps, [r['lr'] for r in records], color='#c06014')
    axes[3].set_title('学习率；虚线为Slurm作业切换位置', fontsize=11)
    axes[3].set_xlabel('成功优化器更新步数')
    for v in jobs:
        axes[3].axvline(v, color='gray', ls=':', lw=.8)
    fig.suptitle('正式T3历史训练：同一连续轨迹，28,365次成功更新')
    fig.tight_layout()
    (root/'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(root/'figures/three_training_losses.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in [('base', 'L_base'), ('weighted_body', '0.1 × L_bodyRGB'), ('weighted_rest', '0.038162448 × L_rest')]:
        axes[0].plot([r['last_update'] for r in bins], [r[key+'_mean'] for r in bins], label=label)
    for key, label in [('base', 'L_base'), ('body_rgb', 'L_bodyRGB'), ('rest', 'L_rest')]:
        initial = bins[0][key+'_mean']
        axes[1].plot([r['last_update'] for r in bins], [r[key+'_mean']/initial for r in bins], label=label)
    axes[0].set_title('各项加权标量贡献（非梯度贡献）')
    axes[1].set_title('相对首500更新均值；首箱=1')
    for ax in axes:
        ax.set_xlabel('成功优化器更新步数')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'figures/weighted_loss_contributions.png', dpi=150)
    plt.close(fig)
    pilot = sup_root(project)/'training/pilot/T3'
    pilot_rows = csv_read(pilot/'training_log.csv')
    same = all(all(a[k] == b[k] for k in ['total', 'denoise', 'lr', 'rng_rank0']) for a, b in zip(pilot_rows, first))
    audit = dict(complete=True, successful_updates=len(records), missing_or_duplicate_updates=0,
                 total_reconstruction_max_error=total_error, auxiliary_reconstruction_max_error=aux_error,
                 main_body_field_is_placeholder=all(float(r['body_rgb']) == 0 for r in first),
                 auxiliary_body_field_used=True, pilot_updates=len(pilot_rows), pilot_is_exact_formal_prefix=same,
                 jobs={j: sum(r['job'] == j for r in records) for j in sorted({r['job'] for r in records})},
                 three_losses=['base', 'body_rgb', 'rest'], coefficients=dict(body_rgb=.1, rest=REST_WEIGHT),
                 schedule_warning='LR goes to zero by design; a flat tail is not proof of global convergence',
                 historical_validation_losses='not recorded; fixed-sample reevaluation is a separate diagnostic')
    write(root/'audit/log_integrity.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def inventory(project):
    root = project/'output_eval'/NAME
    training = sup_root(project)/'training/formal/T3'
    rows = []
    for parent in [training, sup_root(project)/'training/pilot/T3', project/'output/phase1_e1_grouped_e5']:
        for path in sorted(parent.rglob('*')):
            if path.is_file() and (path.suffix in ['.pt', '.bin'] or path.name == 'progress.json'):
                rows.append(dict(path=str(path), resolved=str(path.resolve()), bytes=path.stat().st_size,
                                 sha256=sha(path) if path.suffix in ['.pt', '.bin'] and 'optimizer' not in path.name else None))
    csv_write(root/'audit/server_checkpoint_files.csv', rows)
    write(root/'audit/server_checkpoint_inventory.json', dict(files=rows,
        latest=read(training/'state-latest/progress.json'), verification=read(training/'checkpoint_verification.json'),
        formal_config=read(training/'config.json'),
        note='aliases resolving to one file are one checkpoint; pilot is not formal trajectory'))
    print('INVENTORY_COMPLETE', len(rows), flush=True)


def prepare(project):
    root = project/'output_eval'/NAME
    root.mkdir(parents=True, exist_ok=True)
    training = sup_root(project)/'training/formal/T3'
    latest = read(training/'state-latest/progress.json')
    assert latest['updates'] == 28048
    final = training/'checkpoint-final/joint_model.pt'
    expected = read(training/'checkpoint_verification.json')['sha256']
    assert sha(final) == expected
    e1 = project/'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt'
    assert sha(e1) == 'e51d71dbf17c424fb639293ab99cfa83b9658451e127e6a61e8263e15bd5e39a'
    states = []
    for i, key in enumerate(['unet', 'ref_unet', 'bf_texture_conditioner', 'spatial_texture_encoder', 'spatial_injection']):
        p = training/'state-latest'/('pytorch_model'+('_%d'%i if i else '')+'.bin')
        states.append(dict(key=key, path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    registry = [dict(stage='E1_init', update=0, kind='joint', path=str(e1), sha256=sha(e1),
                     role='T3 initialization control, not a trained early checkpoint'),
                dict(stage='T3_u28048', update=28048, kind='accelerate_state', path=str(training/'state-latest'),
                     initialization_template=str(final), components=states,
                     role='retained late resume snapshot, 98.8824% of formal updates'),
                dict(stage='T3_final', update=28365, kind='joint', path=str(final), sha256=expected,
                     role='formal final checkpoint')]
    actual_dirs = sorted(str(p) for p in training.glob('checkpoint-*') if p.is_dir())
    write(root/'protocol/checkpoints.json', registry)
    write(root/'audit/checkpoint_coverage.json', dict(formal_checkpoint_directories=actual_dirs,
        saved_resume_update=latest['updates'], formal_final_update=28365,
        requested_progress=[.2, .6, 1.], requested_early_update=5673, requested_middle_update=17019,
        early_and_middle_available=False, pilot_is_formal_prefix=False,
        limitation='Early/middle formal weights not retained; E1 is initialization, pilot is a separate run. No training or replay performed.'))
    rows = read(project/'output_eval/t3_p0_p1_20261028/protocol/input_manifest.json')
    assert len(rows) == 128 and sum(r['condition'] == 'original' for r in rows) == 96
    def asset(path):
        path = Path(path)
        digest = sha(path)
        target = root/'inputs'/(digest+path.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(str(path), str(target))
        assert sha(target) == digest
        return str(target), digest
    approved = []
    for row in rows:
        row = dict(row)
        for key in ['sketch', 'reference', 'gt']:
            path, digest = asset(row[key])
            assert digest == row[key+'_sha256']
            row[key] = path
        approved.append(row)
    write(root/'protocol/input_manifest.json', approved)
    original = [r for r in approved if r['condition'] == 'original']
    train = read(project/'data/train_bf_texture.json')
    train_ids = {Path(r['cloth']).stem for r in train}
    assert not train_ids & {r['id'] for r in original}
    order = lambda text: hashlib.sha256(('T3_DIAG_301042/'+text).encode()).hexdigest()
    train_indices = sorted(range(len(train)), key=lambda i: order(Path(train[i]['cloth']).stem))[:32]
    development = sorted(original, key=lambda r: order(r['id']))[:32]
    probes = []
    trainbase = project.parent/'datasets/BF/training'
    for index in train_indices:
        r = train[index]
        paths = {k: asset(trainbase/r[src])[0] for k, src in [('cloth', 'cloth'), ('sketch', 'sketch'), ('texture', 'texture')]}
        mask = project/'output_eval/e2a_attr_01_20261019/masks/train'/('%03d'%(index//1000))/('%05d_mask.png'%index)
        probes.append(dict(id=Path(r['cloth']).stem, split='train_probe32', caption=r['caption'],
                           original_training_index=index, mask=asset(mask)[0], **paths))
    for r in development:
        probes.append(dict(id=r['id'], split='dev_probe32', caption=r['caption'],
                           original_training_index=None, cloth=r['gt'], sketch=r['sketch'], texture=r['reference']))
    write(root/'protocol/loss_probes.json', probes)
    config = dict(protocol='T3-TRAINING-DIAGNOSIS-20261030', new_training=0, independent_confirmation=False,
        log_components=['L_base', 'L_bodyRGB', 'L_rest'], body_weight=.1, rest_weight=REST_WEIGHT,
        inference=dict(resolution=[384, 512], sampler='DDIM', steps=50, cfg=7., sketch_scale=.6,
                       ipa_scale=1., texture_scale=1., seed='historical per-identity seed and explicit latent SHA'),
        data=dict(original_n=96, swapped_n=32, source='frozen P2b96 and reference-swap32, already viewed development data'),
        loss_probe=dict(train_n=32, development_n=32, timesteps=[64, 256, 512, 768], seed=301042,
            VAE='posterior mean', dropout=False, mask='historical training mask for train; same historical sketch algorithm for development',
            precision='fp16 model parameters; float32 loss computation; all checkpoints identical settings',
            comparison='fixed-input diagnostic; absolute levels not comparable to historical stochastic/dropout training log'),
        metric_scope='legacy structure/background proxies + RGB/gray body-patch DISTS + Lab histogram; no valid periodicity primary endpoint',
        analysis='identity paired descriptive effects and bootstrap CI; no new confirmatory success claim',
        checkpoint_limitation='only initialization and two late snapshots; cannot establish early/middle validation trajectory',
        code_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(root/'protocol/config.json', config)
    for p in [Path(__file__), SOURCE/'scripts/t3_training_diagnosis.sbatch']:
        target = root/'code'/p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(target))
    print('PREPARED', len(approved), 'generation inputs;', len(probes), 'loss probes;', 'stages', [r['stage'] for r in registry], flush=True)


def load_pipe(project, stage):
    import gc
    import importlib.util
    from types import SimpleNamespace
    import torch
    root = project/'output_eval'/NAME
    entry = next(r for r in read(root/'protocol/checkpoints.json') if r['stage'] == stage)
    path = Path(entry.get('initialization_template', entry['path']))
    if entry['kind'] == 'joint':
        assert sha(path) == entry['sha256']
    spec = importlib.util.spec_from_file_location('t3_diag_inference', SOURCE/'inference_IMAGGarment-1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sd = project/'models/stable-diffusion-v1-5'
    args = SimpleNamespace(GAM_model_ckpt=str(path), texture_ckpt=str(path), base_model_path=str(sd),
        vae_model_path=str(sd/'vae'), image_encoder_path=str(project/'models/clip'), device='cuda:0',
        width=384, height=512, texture_num_tokens=16, force_texture_num_tokens_override=False,
        texture_condition_mode='token', alpha1=2., alpha2=2., alpha3=1.5, alpha4=1.)
    pipe, _ = module.prepare(args)
    del _
    if entry['kind'] == 'accelerate_state':
        models = dict(unet=pipe.unet, ref_unet=pipe.reference_unet, bf_texture_conditioner=pipe.bf_texture_conditioner)
        verification = []
        for record in entry['components']:
            assert sha(record['path']) == record['sha256']
            if record['key'] in models:
                state = torch.load(record['path'], map_location='cpu', weights_only=False)
                model = models[record['key']]
                result = model.load_state_dict(state, strict=True)
                # 比较实际设备上的值与保存值经相同精度转换后的值，确保没有借用末期权重。
                assert all(torch.equal(v, state[k].to(device=v.device, dtype=v.dtype)) for k, v in model.state_dict().items())
                verification.append(dict(component=record['key'], tensors=len(state), strict=True, device_values_equal=True))
                del state
        write(root/'audit'/('loaded_state_'+stage+'_'+os.environ.get('SLURM_JOB_ID', 'local')+'.json'), verification)
    for model in [pipe.unet, pipe.reference_unet, pipe.vae, pipe.text_encoder, pipe.image_encoder, pipe.bf_texture_conditioner]:
        model.eval().requires_grad_(False)
    pipe.set_layer_group_enabled(True)
    pipe._setup_layer_groups()
    pipe.set_progress_bar_config(disable=True)
    gc.collect()
    torch.cuda.empty_cache()
    return pipe


def gpu_setup():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate(project, stage, shard=0, shards=1):
    import time
    import torch
    from PIL import Image
    gpu_setup()
    root = project/'output_eval'/NAME
    rows = [r for i, r in enumerate(read(root/'protocol/input_manifest.json')) if i % shards == shard]
    digest = lambda t: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
    with torch.inference_mode():
        pipe = load_pipe(project, stage)
        captured = {}
        prepare_latents = pipe.prepare_latents
        def audited(*args, **kwargs):
            value = prepare_latents(*args, **kwargs)
            captured['latent'] = digest(value)
            return value
        pipe.prepare_latents = audited
        for row in rows:
            path = root/'outputs'/stage/row['condition']/(row['id']+'.png')
            if path.exists():
                saved = read(path.with_suffix('.json'))
                assert saved['output_sha256'] == sha(path)
                assert saved['initial_latent_sha256'] == row['historical_initial_latent_sha256']
                continue
            begin = time.perf_counter()
            torch.manual_seed(row['seed'])
            gen = torch.Generator(device='cuda').manual_seed(row['seed'])
            latent = torch.randn((1, 4, 64, 48), generator=gen, device='cuda', dtype=torch.float16)
            assert digest(latent) == row['historical_initial_latent_sha256']
            sketch = Image.open(row['sketch']).convert('RGB').resize((384, 512), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(sketch, np.float32).transpose(2, 0, 1)/127.5-1)[None]
            image = pipe(ref_image=tensor, prompt=row['caption'], texture_clip_image=Image.open(row['reference']).convert('RGB'),
                texture_embeds=None, null_prompt='', negative_prompt=' worst quality, low quality', width=384, height=512,
                num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=1., generator=gen,
                latents=latent, num_inference_steps=50, texture_mode='patch_resampled', texture_num_tokens=16,
                texture_scale=1., texture_condition_mode='token', fusion_type='minimal', texture_preprocess_mode='plain_resize',
                spatial_mask=None, debug_spatial=False, force_texture_num_tokens_override=False)[0]
            assert captured['latent'] == digest(latent)
            assert image.size == (384, 512)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            write(path.with_suffix('.json'), dict(stage=stage, id=row['id'], condition=row['condition'],
                seed=row['seed'], initial_latent_sha256=captured['latent'], output_sha256=sha(path),
                input_sketch_sha256=row['sketch_sha256'], input_reference_sha256=row['reference_sha256'],
                GT_used_in_generation=False, seconds=time.perf_counter()-begin))
            print('GENERATED', stage, row['id'], row['condition'], flush=True)
    write(root/'audit'/('generation_'+stage+'_shard%d.json'%shard), dict(complete=True, n=len(rows), shard=shard, shards=shards))


def loss_probe(project, stage, shard=0, shards=1):
    import importlib.util
    import torch
    import torch.nn.functional as F
    gpu_setup()
    root = project/'output_eval'/NAME
    history = sup_root(project)
    source = history/'source/train_GAM_texture_joint.py'
    assert sha(source) == read(history/'training/formal/T3/config.json')['base_source_sha256']
    spec = importlib.util.spec_from_file_location('t3_diag_training_source', source)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    probes = read(root/'protocol/loss_probes.json')
    with torch.inference_mode():
        pipe = load_pipe(project, stage)
        # 单步监督前向使用训练默认scale=1；生成时的sketch_scale=.6只用于完整采样。
        pipe.set_scale(1.)
        pipe.set_ipa_scale(1.)
        dataset = training.JointTextureDataset(str(root/'protocol/loss_probes.json'), pipe.tokenizer,
            image_root='', width=384, height=512, texture_preprocess_mode='plain_resize')
        style_model = training.VGGGramStyleLoss().cuda().eval().requires_grad_(False)
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule='scaled_linear',
                                 num_train_timesteps=1000, prediction_type='epsilon')
        results = []
        for index, row in enumerate(probes):
            if index % shards != shard:
                continue
            target = root/'loss_probe'/stage/(row['split']+'_'+row['id']+'.json')
            if target.exists():
                results.extend(read(target))
                continue
            batch = {k: v[None].cuda() for k, v in dataset[index].items()}
            latent = pipe.vae.encode(batch['vae_cloth'].half()).latent_dist.mean*pipe.vae.config.scaling_factor
            ref_latent = pipe.vae.encode(batch['vae_sketch'].half()).latent_dist.mean*pipe.vae.config.scaling_factor
            text = pipe.text_encoder(batch['input_ids'])[0]
            clip = pipe.image_encoder(batch['clip_texture'].half(), output_hidden_states=True)
            tokens, _ = pipe.bf_texture_conditioner(clip_image_embeds=clip.image_embeds,
                texture_images=batch['texture_image'].half(), clip_vision_tokens=clip.hidden_states[-1][:, 1:, :],
                texture_mode='patch_resampled')
            enc_h = torch.cat([text, tokens], dim=1)
            body, _, _ = training.build_region_masks(batch['garment_mask'].float(), 9)
            values = []
            for timestep in [64, 256, 512, 768]:
                key = row['split']+'/'+row['id']+'/'+str(timestep)
                seed = 301042+int(hashlib.sha256(key.encode()).hexdigest()[:7], 16)
                noise = torch.randn(latent.shape, generator=torch.Generator(device='cuda').manual_seed(seed),
                                    device='cuda', dtype=latent.dtype)
                t = torch.tensor([timestep], device='cuda', dtype=torch.long)
                noisy = scheduler.add_noise(latent, noise, t)
                pipe.reference_unet(ref_latent, torch.zeros_like(t), None, return_dict=False)
                sa = {n: p.cache['hidden_states'] for n, p in pipe.reference_unet.attn_processors.items()
                      if 'attn1' in n and hasattr(p, 'cache')}
                predicted = pipe.unet(noisy, t, encoder_hidden_states=enc_h,
                                     cross_attention_kwargs={'sa_hidden_states': sa}).sample
                x0 = training.reconstruct_x0(noisy, predicted, t, scheduler)
                decoded = pipe.vae.decode(x0/pipe.vae.config.scaling_factor).sample.float().contiguous()
                mask = batch['garment_mask'].float().contiguous()
                gt = batch['vae_cloth'].float().contiguous()
                reference = batch['texture_image'].float().contiguous()
                denoise = F.mse_loss(predicted.float(), noise.float())
                gram = style_model(decoded, gt, mask=mask)
                edge = training.masked_edge_l1(decoded, batch['vae_sketch'].float().contiguous(), mask=mask)
                color = training.texture_color_stat_loss(decoded, reference, garment_mask=mask)
                body_loss = training.texture_color_stat_loss(decoded, reference, garment_mask=body)
                rest = training.masked_l1_loss(decoded, gt, 1-body)
                base = denoise+gram+.05*edge+.1*color
                total = base+.1*body_loss+REST_WEIGHT*rest
                rec = dict(stage=stage, id=row['id'], split=row['split'], timestep=timestep, seed=seed,
                    base=float(base), body_rgb=float(body_loss), rest=float(rest), total=float(total),
                    denoise=float(denoise), gram=float(gram), edge=float(edge), color=float(color),
                    noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    latent_sha256=hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest(),
                    reference_latent_sha256=hashlib.sha256(ref_latent.cpu().numpy().tobytes()).hexdigest())
                assert all(np.isfinite(v) for v in rec.values() if isinstance(v, float))
                values.append(rec)
            write(target, values)
            results.extend(values)
            print('LOSS_PROBE', stage, row['split'], row['id'], flush=True)
    csv_write(root/'loss_probe'/('combined_'+stage+'_shard%d.csv'%shard), results)
    write(root/'audit'/('loss_probe_'+stage+'_shard%d.json'%shard), dict(complete=True, rows=len(results)))


def evaluate(project, stage):
    import cv2
    import torch
    from PIL import Image
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_leakage
    from tools import t3_response_run as response
    from tools.pattern_fidelity_run import patches, reference_images
    from tools.pattern_fidelity_core import gray_rgb, js
    from tools.pattern_fidelity_evaluate import lab_hist
    gpu_setup()
    root = project/'output_eval'/NAME
    rows = read(root/'protocol/input_manifest.json')
    old = {r['id']: r for r in rows if r['condition'] == 'original'}
    new = {r['id']: r for r in rows if r['condition'] == 'swapped'}
    response.ROOT = root/'runtime_assets'/stage
    model = response.load_dists('cuda')
    sanity = response.sanity(model)
    write(root/'audit'/('DISTS_sanity_'+stage+'.json'), sanity)
    cache = {}
    def feature(parts, view):
        data = [gray_rgb(x) for x in parts] if view == 'gray' else parts
        value = torch.from_numpy(np.stack(data).transpose(0, 3, 1, 2).copy()).float().cuda()/255
        return model.forward_once(value)
    def reference(path):
        if path not in cache:
            part = reference_images(path)
            cache[path] = dict(rgb=feature(part, 'rgb'), gray=feature(part, 'gray'), color=lab_hist(part))
        return cache[path]
    results = []
    with torch.inference_mode():
        for row in rows:
            target = root/'metrics'/stage/row['condition']/(row['id']+'.json')
            path = root/'outputs'/stage/row['condition']/(row['id']+'.png')
            meta = read(path.with_suffix('.json'))
            assert sha(path) == meta['output_sha256']
            if target.exists():
                rec = read(target)
                assert rec['output_sha256'] == sha(path)
                results.append(rec)
                continue
            bundle = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
            _, coords, candidates = patches(bundle['garment'])
            image = np.asarray(Image.open(path).convert('RGB'))
            kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=bundle)
            structure = compute_structure_preservation(**kwargs)
            leak = compute_texture_leakage(**kwargs)
            rec = dict(stage=stage, id=row['id'], condition=row['condition'], output_sha256=sha(path),
                iou=structure.get('sketch_iou'), edge_f1=structure.get('edge_f1'),
                edge_l1=structure.get('edge_l1'), leak=leak.get('leak_colored_frac'),
                patch_n=len(coords), candidates=candidates, dists_rgb=None, dists_gray=None, color_js=None,
                dists_rgb_margin=None, dists_gray_margin=None, color_margin=None)
            if len(coords) >= 4:
                part = [image[y:y+64, x:x+64] for x, y in coords]
                f = {view: feature(part, view) for view in ['rgb', 'gray']}
                histogram = lab_hist(part)
                current = reference(row['reference'])
                for view in ['rgb', 'gray']:
                    rec['dists_'+view] = response.patch_distance(model, f[view], current[view])
                rec['color_js'] = js(histogram, current['color'])
                if row['id'] in new:
                    prior, alternative = reference(old[row['id']]['reference']), reference(new[row['id']]['reference'])
                    for view in ['rgb', 'gray']:
                        rec['dists_'+view+'_margin'] = response.patch_distance(model, f[view], prior[view])-response.patch_distance(model, f[view], alternative[view])
                    rec['color_margin'] = js(histogram, prior['color'])-js(histogram, alternative['color'])
            else:
                rec['missing_reason'] = 'fewer than4 strict interior patches; retained NA'
            write(target, rec)
            results.append(rec)
            print('METRICS', stage, row['id'], row['condition'], flush=True)
    csv_write(root/'metrics'/('combined_'+stage+'.csv'), results)
    write(root/'audit'/('evaluation_'+stage+'.json'), dict(complete=True, n=len(results)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['logs', 'inventory', 'prepare', 'generate', 'loss_probe', 'evaluate'])
    p.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    p.add_argument('--stage', default='T3_final')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    if a.action in ['generate', 'loss_probe']:
        globals()[a.action](a.project, a.stage, a.shard, a.shards)
    elif a.action == 'evaluate':
        evaluate(a.project, a.stage)
    else:
        globals()[a.action](a.project)
