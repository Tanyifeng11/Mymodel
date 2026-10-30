"""一次有预算上限的T3权重续训；复用冻结训练损失与固定评价协议。"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tools import t3_training_diagnosis as diag

PROJECT = Path('/share/home/u2515283058/Mymodel')
NAME = 't3_controlled_continue_20261030'
ROOT = PROJECT/'output_eval'/NAME
PREVIOUS = PROJECT/'output_eval/t3_training_diagnosis_20261030'
RUN = ROOT/'training'
STAGES = ['C0250', 'C0500', 'C1000']
BASE_SHA = '25f95844696ba5c07eb9223e3bdd4d68c78ba4f1a36d561569c35cc90186d259'


def frozen(path, value):
    if path.exists():
        assert diag.read(path) == value, ('FROZEN_CHANGED', str(path))
    else:
        diag.write(path, value)


def prepare():
    assert not (RUN/'training_complete.json').exists()
    base = next(r for r in diag.read(PREVIOUS/'protocol/checkpoints.json') if r['stage'] == 'T3_final')
    assert diag.sha(base['path']) == base['sha256'] == BASE_SHA
    registry = [base]+[dict(stage=s, update=n, kind='joint', path=str(RUN/('checkpoint-%d'%n)/'joint_model.pt'))
                      for s, n in zip(STAGES, [250, 500, 1000])]
    # 完全复用旧输入与基线数值；路径改为本轮拷贝，文件内容哈希不变。
    shutil.copytree(PREVIOUS/'inputs', ROOT/'inputs', dirs_exist_ok=True)
    for name in ['input_manifest.json', 'loss_probes.json']:
        value = diag.read(PREVIOUS/'protocol'/name)
        for row in value:
            for key in ['sketch', 'reference', 'gt', 'cloth', 'texture', 'mask']:
                if key in row:
                    old = Path(row[key]); new = ROOT/'inputs'/old.name
                    assert diag.sha(old) == diag.sha(new)
                    row[key] = str(new)
        frozen(ROOT/'protocol'/name, value)
    for part in ['outputs', 'metrics', 'loss_probe']:
        shutil.copytree(PREVIOUS/part/'T3_final', ROOT/part/'T3_final', dirs_exist_ok=True)
    protocol = dict(name=NAME, initialization=base, optimizer='fresh AdamW; not uninterrupted full-state resume',
        lr=5e-6, scheduler='constant, nonzero from the first optimizer update', warmup=0,
        maximum_successful_updates=1000, checkpoints=[0,250,500,1000], effective_batch=8,
        world_size=2, microbatch=1, accumulation=4, seed=42, data_start_epoch=5,
        rng='fresh seed42+rank; subsequent restarts restore this run optimizer/scheduler/RNG',
        original_training_objective=True, body_weight=.1, rest_weight=diag.REST_WEIGHT,
        fixed_evaluation=diag.read(PREVIOUS/'protocol/config.json'),
        decision=dict(dev_total_reduction_at1000_at_least=.005, dev_total_CI_high_below=0,
            dev_total_delta_at500_below=0, guardrail_mean_absolute_degradation_max=.005,
            guardrail_metrics=['iou','edge_f1','leak'], guardrail_conditions=['original','swapped'],
            texture_additional_requirement='swapped gray DISTS delta<0 at500 and CI_high<0 at1000',
            interpretation='pragmatic exploratory thresholds; no universal or confirmatory claim'),
        gpu_seconds_hard_cap=9000, new_independent_test=False,
        prior_protocol_sha256=diag.sha(PREVIOUS/'protocol/config.json'))
    frozen(ROOT/'protocol/controlled_protocol.json', protocol)
    if not (ROOT/'protocol/checkpoints.json').exists(): diag.write(ROOT/'protocol/checkpoints.json', registry)
    files = [Path(__file__), diag.SOURCE/'scripts/t3_controlled_continue.sbatch',
             diag.SOURCE/'tools/t3_training_diagnosis.py', diag.SOURCE/'tools/e2a_sup_train.py',
             diag.SOURCE/'tools/e2a_sup_losses.py', diag.SOURCE/'tools/e2a_attr_runtime.py']
    for p in files:
        target=ROOT/'code'/p.name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    diag.write(ROOT/'audit/prepared.json', dict(complete=True, commit=subprocess.check_output(
        ['git','-C',str(diag.SOURCE),'rev-parse','HEAD'],text=True).strip(),
        protocol_sha256=diag.sha(ROOT/'protocol/controlled_protocol.json'),
        baseline_reused=True, baseline_files={p.name:diag.sha(p) for p in (PREVIOUS/'protocol').glob('*.json')}))
    print('PREPARED_CONTROLLED_CONTINUATION', flush=True)


def train():
    import torch
    # 显式按启动器LOCAL_RANK绑定设备，再创建Accelerator及任何CUDA上下文。
    # Slurm批处理环境中的默认设备不能被两个进程共同沿用。
    local_rank=int(os.environ['LOCAL_RANK'])
    previous_device=os.environ.get('ACCELERATE_TORCH_DEVICE')
    print('LAUNCH_ALLOCATION',dict(host=os.environ.get('HOSTNAME'),nodes=os.environ.get('SLURM_JOB_NUM_NODES'),
        visible=os.environ.get('CUDA_VISIBLE_DEVICES'),local_rank=local_rank,
        cuda_count=torch.cuda.device_count()),flush=True)
    assert torch.cuda.device_count()==2 and local_rank in [0,1]
    os.environ['ACCELERATE_TORCH_DEVICE']='cuda:%d'%local_rank
    torch.cuda.set_device(local_rank)
    print('RANK_DEVICE_BINDING',dict(rank=os.environ['RANK'],local_rank=local_rank,
        current_device=torch.cuda.current_device(),visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        previous_accelerate_device=previous_device),flush=True)
    from tools.e2a_sup_train import Hook
    from tools.e2a_attr_runtime import AuditRuntime as OriginalRuntime
    from tools.e2a_sup_common import canonical_environment
    assert diag.read(ROOT/'audit/prepared.json')['complete']
    assert not (RUN/'training_complete.json').exists(), 'This run is already complete'
    protocol=diag.read(ROOT/'protocol/controlled_protocol.json')
    assert diag.sha(protocol['initialization']['path']) == BASE_SHA
    historical=PROJECT/'output_eval/e2a_sup_comp_01_20261021'
    config=diag.read(historical/'training/formal/T3/config.json')
    source=historical/'source/train_GAM_texture_joint.py'
    assert diag.sha(source)==config['base_source_sha256']
    RUN.mkdir(parents=True,exist_ok=True)

    class Runtime(OriginalRuntime):
        def restore(self, initial):
            if not (self.folder/'state-latest/progress.json').exists():
                assert initial==0
                return 0, 5, 0
            result=super().restore(initial)
            if self.accelerator.is_main_process:
                p=self.folder/'auxiliary_log.csv'
                if p.exists(): diag.csv_write(p,[r for r in diag.csv_read(p) if int(r['update'])<=result[0]])
            self.accelerator.wait_for_everyone()
            return result

        def save_state(self, step, epoch, offset, force=False):
            if step not in [250,500,1000] and not force: return
            folder=self.folder/'state-latest'
            self.accelerator.save_state(str(folder),safe_serialization=False)
            diag.write(folder/('stream_rank%d.json'%self.rank),dict(rng_chain=self.rng_chain))
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                diag.write(folder/'progress.json',dict(updates=step,epoch=epoch,offset=offset,seed=self.args.seed,
                    world_size=self.accelerator.num_processes,microbatches=self.microbatches))
            self.accelerator.wait_for_everyone()

    # 旧Hook提供相同T3辅助项和更新统计；环境仅规范无序默认字段列表。
    import tools.e2a_sup_train as old_hook
    old_hook.freeze=lambda path,value: frozen(Path(path),canonical_environment(value))
    hook=Hook('T3','controlled',RUN)
    replacements={
        'from tools.e2a_attr_runtime import AuditRuntime, EpochSampler':'from tools.e2a_attr_runtime import EpochSampler',
        '    runtime = AuditRuntime(args, accelerator)':'    runtime = AuditRuntime(args, accelerator)\n    SUP_HOOK.setup(locals())',
        '            with accelerator.accumulate(unet):':'            with accelerator.accumulate(unet), SUP_HOOK.context():',
        '                if not torch.isfinite(loss.detach()).all():':
            '                loss = SUP_HOOK(locals(), globals())\n                if not torch.isfinite(loss.detach()).all():',
        '        "cosine",':'        "constant",',
        '                    and global_step % checkpoint_interval_steps == 0':'                    and global_step in [250, 500]',
    }
    modified=source.read_text(encoding='utf-8')
    for before,after in replacements.items():
        assert modified.count(before)==1, ('ANCHOR_CHANGED',before)
        modified=modified.replace(before,after)
    argv=list(config['argv'])+['--output_dir',str(RUN),'--attr_audit',str(RUN),
        '--gam_init_ckpt',protocol['initialization']['path'],'--resume_from_checkpoint','',
        '--start_global_step','0','--max_train_steps','1000','--learning_rate','5e-6','--num_warmup_steps','0']
    rank=int(os.environ.get('RANK','0'))
    if rank==0:
        frozen(ROOT/'protocol/training_config.json',dict(argv=argv,hooks=replacements,base_source_sha256=diag.sha(source),
            transformed_sha256=hashlib.sha256(modified.encode()).hexdigest(),protocol=protocol))
    diag.write(ROOT/'audit'/('training_job_%s_rank%d.json'%(os.environ.get('SLURM_JOB_ID'),rank)),
        dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),new_optimizer=True,lr=5e-6,
             runner_sha256=diag.sha(__file__),job=os.environ.get('SLURM_JOB_ID')))
    namespace=dict(__name__='__main__',__file__=str(source),SUP_HOOK=hook,AuditRuntime=Runtime)
    sys.argv=[str(source)]+argv
    exec(compile(modified,str(source),'exec'),namespace)
    if rank==0:
        count=diag.read(RUN/'optimizer_update_count.json')
        assert count['actual_successful_updates']==1000 and count['skipped_updates']==0
        rows=diag.csv_read(RUN/'training_log.csv')
        assert [int(r['update']) for r in rows]==list(range(1,1001))
        assert all(abs(float(r['lr'])-5e-6)<1e-15 for r in rows)
        assert diag.read(RUN/'state-latest/progress.json')['updates']==1000
        print('CONTROLLED_TRAINING_COMPLETE',flush=True)


def register():
    import torch
    assert diag.read(RUN/'training_complete.json')['updates']==1000
    registry=diag.read(ROOT/'protocol/checkpoints.json')
    for r in registry:
        r['sha256']=diag.sha(r['path'])
        if r['stage']!='T3_final':
            weights=torch.load(r['path'],map_location='cpu',mmap=True,weights_only=False)
            tensors=[v for group in weights.values() if isinstance(group,dict) for v in group.values() if torch.is_tensor(v)]
            assert all(torch.isfinite(t).all().item() for t in tensors)
            del weights,tensors
    diag.write(ROOT/'protocol/checkpoints.json',registry)
    diag.write(ROOT/'audit/checkpoints_registered.json',dict(complete=True,registry=registry))
    print('REGISTERED_FINITE_CHECKPOINTS',flush=True)


def worker(shard,shards):
    import gc
    import torch
    diag.NAME=NAME
    assert diag.read(ROOT/'audit/checkpoints_registered.json')['complete']
    diag.write(ROOT/'audit'/('worker_%s_shard%d.json'%(os.environ.get('SLURM_JOB_ID'),shard)),
        dict(shard=shard,shards=shards,runner_sha256=diag.sha(__file__),torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    for stage in STAGES:
        pipe=diag.load_pipe(PROJECT,stage)
        diag.generate(PROJECT,stage,shard,shards,pipe=pipe)
        diag.loss_probe(PROJECT,stage,shard,shards,pipe=pipe)
        del pipe
        gc.collect(); torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','register','worker','evaluate'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--stage',choices=STAGES)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='train':train()
    elif a.action=='register':register()
    elif a.action=='worker':worker(a.shard,a.shards)
    else:
        assert a.stage
        diag.NAME=NAME
        diag.evaluate(PROJECT,a.stage)
