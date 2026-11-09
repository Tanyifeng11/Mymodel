"""A0复用已验收B0，A5独立续训2000更新；所有新开关关闭时保留E2。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import numpy as np
import torch

from models.e2_compat import compute_losses, cpu_snapshot, compare_snapshots, load_trainable_strict
from models.e2_structure import attach_structure, StructureResidual, VERSION
from tools.e2_fft_experiment import SOURCE, setup, load, read, write, sha, environment
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash, optimizer_hash, frozen_hash


def a_environment():
    env = environment()
    for name in ['models/e2_structure.py', 'tools/e2_a_minimal.py', 'tools/e2_a_evaluation.py',
                 'tests/test_e2_structure.py', 'tests/test_e2_a_metrics.py', 'scripts/e2_a_minimal.sbatch']:
        env['source'][name] = sha(SOURCE / name)
    return env


def config(args):
    value = dict(read(Path(args.baseline) / 'comparison/B0.json'))
    value.update(arm='A5', code_commit=environment()['commit'], structure_mode='A5',
        structure_width=32, structure_version=VERSION, protocol='A_minimal_v1',
        a_pre='original_sketch', a_encoder=True, a_support=False, B=False, C=False,
        fft=False, fft_weight=0., updates=2000, effective_batch=8, world_size=1,
        split=str(Path(args.out) / 'split.json'), split_sha256=sha(Path(args.out) / 'split.json'))
    value.pop('auxiliary_weights', None)
    return value


def mask_snapshot(args, core, final=False):
    rows = read(core.ROOT / 'training_manifest.json')
    before_baseline = (Path(args.baseline) / 'comparison/B0.json').stat().st_mtime_ns
    def inspect(index):
        name = '%03d/%05d_mask.png' % (index // 1000, index)
        p = core.MASKS / name
        stat = p.stat()
        assert stat.st_mtime_ns <= before_baseline, '旧mask在B0配置冻结后被改动，不能直接复用B0: ' + name
        return name, dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(p))
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(inspect, range(len(rows))))
    path = Path(args.out) / 'audit/mask_inputs.json'
    if final or path.exists():
        assert read(path) == result
    else:
        write(path, result)
    write(Path(args.out) / ('audit/mask_inputs_final.json' if final else 'audit/mask_input_status.json'),
          dict(passed=True, files=len(result), snapshot_sha256=sha(path),
               historical_mask_sha_available=False, evidence='current hashes plus all mtimes predate B0 config; exact four-update baseline replay required'))


def prepare(args):
    from tools.e2_b_screen_training import audit_training
    core, out = setup(args)
    baseline = Path(args.baseline)
    complete = read(baseline / 'training/B0/complete.json')
    assert complete['status'] == 'completed' and complete['updates'] == 2000
    assert sha(baseline / 'training/B0/latest.pt') == complete['checkpoint_sha256']
    assert read(baseline / 'audit/training_inputs_final.json')['passed']
    audit_training(args)
    assert sha(out / 'audit/training_inputs.json') == sha(baseline / 'audit/training_inputs.json')
    mask_snapshot(args, core)
    for name in ['strict_pairs.json', 'inputs/registry.json', 'audit/input_version_comparison.json']:
        target = out / name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == (baseline / name).read_bytes()
        else:
            shutil.copy2(baseline / name, target)
    spec = read(out / 'strict_pairs.json')
    assert len(spec['pairs']) == 32
    provenance = []
    for pair in spec['pairs']:
        for key in ['sketch','gt','original_reference','swapped_reference']:
            source_input=Path(args.data)/pair[key]
            assert sha(source_input)==pair['hashes'][key]
            archived_input=out/'inputs/frozen'/pair[key]
            archived_input.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source_input,archived_input)
        for condition in ['old', 'new']:
            source = baseline / 'outputs/B0' / pair['case_id'] / (condition + '.png')
            meta = read(source.with_suffix('.json'))
            assert sha(source) == meta['output_sha256']
            assert meta['checkpoint_sha256'] == complete['checkpoint_sha256']
            assert meta['manifest_sha256'] == sha(out / 'strict_pairs.json')
            target = out / 'outputs/A0' / pair['case_id'] / source.name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            provenance.append(dict(file=str(source), metadata_sha256=sha(source.with_suffix('.json')), output_sha256=sha(source)))
            meta.update(arm='A0', reused_from_arm='B0', original_metadata_sha256=sha(source.with_suffix('.json')))
            write(target.with_suffix('.json'), meta)
    folder = out / 'training/A0'; folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline / 'training/B0/training.jsonl', folder / 'training.jsonl')
    write(folder / 'complete.json', dict(complete, alias='A0', reused_from=str(baseline / 'training/B0')))
    write(out / 'audit/baseline_reuse.json', dict(prepared=True, authorized_reuse_not_new_training=True,
        checkpoint=str(baseline / 'training/B0/latest.pt'), checkpoint_sha256=complete['checkpoint_sha256'],
        records=provenance, required_gate='exact alloff/zeroon forward+image and four-update legacy replay',
        original_code_commit=complete['environment']['commit']))
    variants = {m:dict(mode=m, parameters=sum(p.numel() for p in StructureResidual(mode=m).parameters()),
                       status='configuration_only_not_trained') for m in ['U1', 'A3', 'A4', 'A5']}
    write(out / 'comparison/A5.json', config(args))
    write(out / 'comparison/variants.json', dict(variants=variants,
        note='U1 is parameter-matched unlayered two-lane single-scale control, not the specification single-branch A1; only A5 authorized for this minimal run'))
    write(out / 'protocol.json', dict(version='e2_A_minimal_v1', arms=['A0', 'A5'],
        training='mature_E2 continuation2000/effectivebatch8/single_seed42; A0 reuses verified B0',
        primary='old_boundary_f1_r2', secondary=['old_edge_f1', 'old_iou', 'old_inner_support_proxy'],
        tolerance_pixels=[1, 2, 4], secondary_holm_family=3,
        texture_guard='same fixed v3 cross-patch DISTS; one gray strict comparison and three secondary comparisons',
        baseline_input_manifest_sha256=sha(out / 'strict_pairs.json'), structure_version=VERSION,
        evaluation_limit='32 texture-focused identities,23stripes/3checks/6dots; no human semantic internal-line labels; not a balanced formal A validation',
        a_support=False, B=False, C=False, environment=a_environment()))
    print('A_PREPARE_PASS', flush=True)


def load_a(args, core):
    pipe, network, checked = load(args, core)
    attach_structure(network, mode='A5', width=32)
    return pipe, network, checked


def regression(args):
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    pipe, network, _ = load(args, core)
    state = torch.load(Path(args.baseline) / 'training/B0/latest.pt', map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network']); del state
    network.eval()
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    data = dataset(pipe, core)
    choices = read(out / 'split.json')['smoke']
    before = []
    for item in choices:
        batch = {k:v[None].to(args.device) for k,v in data[item['index']].items()}
        rng = torch.Generator(device=args.device).manual_seed(42000 + item['index'])
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            total, raw, token, captured, _ = compute_losses(pipe, network, batch, rng, style, core, audit_mode=True)
        before.append(cpu_snapshot(raw, token, captured, total, rng)); network.adapter.clear()
        del total, raw, captured
    adapter = attach_structure(network, enabled=False)
    network.eval()
    comparisons = []
    for enabled in [False, True]:
        adapter.structure_enabled = enabled
        for item, expected in zip(choices, before):
            batch = {k:v[None].to(args.device) for k,v in data[item['index']].items()}
            rng = torch.Generator(device=args.device).manual_seed(42000 + item['index'])
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                total, raw, token, captured, _ = compute_losses(pipe, network, batch, rng, style, core, audit_mode=True)
            errors = compare_snapshots(expected, cpu_snapshot(raw, token, captured, total, rng), atol=0, rtol=0)
            comparisons.append(dict(enabled=enabled, index=item['index'], errors=errors)); adapter.clear()
            del total, raw, captured
    write(out / 'regression/forward_result.json', dict(passed=True, comparisons=comparisons))
    del style, data, before
    pipe.unet.half(); pipe.vae.half(); adapter.half()
    pipe.image_encoder.to(args.device); pipe.bf_texture_conditioner.to(args.device)
    images = []
    with torch.inference_mode():
        for enabled in [False, True]:
            adapter.structure_enabled = enabled
            for pair in read(out / 'strict_pairs.json')['pairs'][:4]:
                adapter.condition(core.tensor(Path(args.data)/pair['sketch']).to(args.device).half(),
                                  core.tensor(Path(args.data)/pair['original_reference']).to(args.device).half())
                latent = torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                image = pipe(ref_image=core.tensor(Path(args.data)/pair['sketch']).to(args.device).half(),
                    prompt=pair['caption'],texture_clip_image=None,texture_embeds=None,null_prompt='',
                    negative_prompt=' worst quality, low quality',width=384,height=512,num_images_per_prompt=1,
                    guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,latents=latent,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']),num_inference_steps=50)[0]
                from PIL import Image
                reference = Image.open(out / 'outputs/A0' / pair['case_id'] / 'old.png').convert('RGB')
                error = int(np.abs(np.asarray(image).astype(int)-np.asarray(reference).astype(int)).max())
                path=out/'regression'/('zeroon' if enabled else 'alloff')/(pair['case_id']+'.png')
                path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                images.append(dict(case_id=pair['case_id'],enabled=enabled,max_pixel_difference=error))
                assert error == 0
                adapter.clear()
    write(out / 'regression/image_result.json',dict(passed=True,records=images,environment=a_environment()))
    print('A_REGRESSION_PASS',flush=True)


class ATrainingRun(TrainingRun):
    def update(self):
        from models.e2_compat import bare_network
        adapter = bare_network(self.network).adapter
        adapter.structure_records = []
        row = super().update()
        row['structure'] = adapter.structure_records
        return row


def smoke(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    assert read(out/'regression/forward_result.json')['passed'] and read(out/'regression/image_result.json')['passed']
    pipe, network, _ = load(args, core)
    legacy_initial = network.trainable_state()
    adapter = attach_structure(network, enabled=False)
    adapter.structure.requires_grad_(False)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    data = dataset(pipe, core)
    acc = Accelerator(mixed_precision='bf16',gradient_accumulation_steps=8)
    assert acc.num_processes == 1
    baseline_config = read(Path(args.baseline)/'training/B0/complete.json')['config']
    old_rows = [__import__('json').loads(t) for t in (Path(args.baseline)/'training/B0/training.jsonl').read_text().splitlines()][:4]
    run = TrainingRun(args,core,pipe,network,data,style,baseline_config,read(out/'split.json')['train_indices'],acc)
    replay = []
    for expected in old_rows:
        got = run.update()
        assert got['tokens'] == expected['tokens']
        assert got['losses'] == expected['losses'] and got['grad_norm'] == expected['grad_norm'], 'B0前四更新不逐位重现，不可直接复用'
        replay.append(dict(step=got['step'],tokens_equal=True,losses_equal=True,gradient_norm_equal=True))
    write(out/'audit/baseline_replay.json',dict(passed=True,records=replay))
    network = acc.unwrap_model(run.network,keep_fp32_wrapper=False)
    del run
    acc.free_memory()
    load_trainable_strict(network,legacy_initial)
    adapter = network.adapter; adapter.structure_enabled = True; adapter.structure.requires_grad_(True)
    before = network.trainable_state(); frozen_before = frozen_hash(pipe,network)
    cfg=dict(read(out/'comparison/A5.json'),updates=4,warmup=0,workers=0,diagnostics=True)
    run=ATrainingRun(args,core,pipe,network,data,style,cfg,[x['index'] for x in read(out/'split.json')['smoke']],acc)
    logs=[run.update(),run.update()]
    folder=out/'smoke';folder.mkdir(exist_ok=True)
    run.save(folder/'resume.pt')
    expected=run.update(); expected_state=acc.unwrap_model(run.network).trainable_state()
    optimizer_expected=optimizer_hash(run.optimizer)
    run.restore(folder/'resume.pt');actual=run.update()
    assert tensor_hash(expected_state.items())==tensor_hash(acc.unwrap_model(run.network).trainable_state().items())
    assert optimizer_hash(run.optimizer)==optimizer_expected
    assert expected['tokens']==actual['tokens'] and expected['losses']==actual['losses']
    logs.extend([actual,run.update()])
    network=acc.unwrap_model(run.network)
    changes={}
    for name,p in network.trainable_state().items():
        group='A_outer' if name.startswith('adapter.structure.outer') else 'A_inner' if name.startswith('adapter.structure.inner') else 'A_heads' if name.startswith('adapter.structure.heads') else 'old_E2'
        changes[group]=changes.get(group,0.)+float((p.float()-before[name].float()).square().sum())
    assert all(np.isfinite(x) and x>0 for x in changes.values()) and len(changes)==4
    assert frozen_hash(pipe,network)==frozen_before
    batch={k:v[None].to(args.device) for k,v in data[read(out/'split.json')['smoke'][0]['index']].items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _,raw,_,captured,_=compute_losses(pipe,network,batch,torch.Generator(device=args.device).manual_seed(42000),style,core,audit_mode=True)
        image_loss=raw['gram']+.05*raw['edge']+.1*raw['color']+.1*raw['body']+.03816244801608925*raw['rest']
    targets=[captured['epsilon'],captured['vae_input'],adapter.structure.heads[0].weight,adapter.structure.outer[0].weight,adapter.structure.inner[0].weight]
    grads=torch.autograd.grad(image_loss,targets,allow_unused=True)
    norms={k:float(g.float().norm()) if g is not None else 0. for k,g in zip(['epsilon','vae_input','A_head0','A_outer0','A_inner0'],grads)}
    assert all(np.isfinite(x) and x>0 for x in norms.values())
    write(folder/'result.json',dict(passed=True,updates=4,executed_update_calls=5,resume_exact=True,
        changes_squared_l2=changes,frozen_unchanged=True,image_loss_gradient_norms=norms,
        added_trainable_parameters=sum(p.numel() for p in adapter.structure.parameters()),logs=logs,environment=a_environment()))
    print('A_SMOKE_PASS',flush=True)


def train(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    from tools.e2_b_screen_training import assert_snapshot
    core,out=setup(args)
    assert read(out/'smoke/result.json')['passed'] and read(out/'audit/baseline_replay.json')['passed']
    cfg=read(out/'comparison/A5.json')
    assert cfg['code_commit']==environment()['commit'] and assert_snapshot(args)==cfg['training_input_map_sha256']
    assert sha(out/'split.json')==cfg['split_sha256']
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=8)
    assert acc.num_processes==1
    pipe,network,_=load_a(args,core);data=dataset(pipe,core)
    style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    folder=out/'training/A5';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        value=read(folder/'complete.json')
        assert value['config']==cfg and sha(folder/'latest.pt')==value['checkpoint_sha256']
        print('A_ALREADY_COMPLETED');return
    run=ATrainingRun(args,core,pipe,network,data,style,cfg,read(out/'split.json')['train_indices'],acc)
    if (folder/'latest.pt').exists():run.restore(folder/'latest.pt')
    log=folder/'training.jsonl'
    if log.exists():
        rows=[__import__('json').loads(t) for t in log.read_text().splitlines()]
        log.write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows if r['step']<=run.step))
    baseline_rows=[__import__('json').loads(t) for t in (out/'training/A0/training.jsonl').read_text().splitlines()]
    while run.step<cfg['updates']:
        row=run.update()
        assert row['tokens']==baseline_rows[row['step']-1]['tokens'],'A5旧随机流不同于A0'
        with log.open('a',encoding='utf-8') as stream:stream.write(__import__('json').dumps(row,allow_nan=False)+'\n')
        if run.step<=4 or run.step%25==0:print('A_TRAIN',run.step,row['losses'],row['structure'][-1],flush=True)
        if run.step%100==0 or run.step==cfg['updates']:run.save(folder/'latest.pt')
    write(folder/'complete.json',dict(status='completed',updates=run.step,config=cfg,
        environment=a_environment(),checkpoint_sha256=sha(folder/'latest.pt')))
    print('A_TRAIN_COMPLETED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','regression','smoke','train','audit-final','generate','evaluate','bundle'])
    p.add_argument('--project',default='/share/home/u2515283058/Mymodel')
    p.add_argument('--baseline',default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--out',default='/share/home/u2515283058/Mymodel/output_eval/e2_a_minimal_20261109')
    p.add_argument('--data',default='/share/home/u2515283058/datasets/BF')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',default='A5',choices=['A5'])
    p.add_argument('--candidates',default=str(SOURCE/'configs/e2_b_candidates.json'))
    args=p.parse_args();args.parent=args.baseline
    if args.action in ['generate','evaluate','bundle']:
        from tools import e2_a_evaluation
        getattr(e2_a_evaluation,args.action)(args)
    elif args.action=='audit-final':
        from tools.e2_b_screen_training import audit_training
        audit_training(args,final=True);core,_=setup(args);mask_snapshot(args,core,final=True)
    else:globals()[args.action](args)


if __name__=='__main__':main()
