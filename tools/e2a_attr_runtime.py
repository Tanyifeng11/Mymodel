"""训练审计：独立采样器、共同随机流、真实更新日志及可续跑状态。"""
import csv
import hashlib
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Sampler
from tools.e2a_attr_common import read, write, sha


class EpochSampler(Sampler):
    def __init__(self, dataset, seed): self.dataset, self.seed, self.epoch = dataset, int(seed), 0
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.dataset)
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


class AuditRuntime:
    def __init__(self, args, accelerator):
        self.enabled = bool(args.attr_audit)
        if not self.enabled: return
        self.args, self.accelerator = args, accelerator
        self.folder = Path(args.attr_audit); self.folder.mkdir(parents=True, exist_ok=True)
        self.rank = accelerator.process_index
        self.losses = []
        self.rng_chain = '0'*64
        self.microbatches = 0
        self.started = time.time()
        self.paused = False
        soft_seconds=int(os.environ.get('E2A_JOB_SOFT_SECONDS','0'))
        self.soft_end=int(os.environ.get('E2A_JOB_START',self.started))+soft_seconds if soft_seconds else None

    def restore(self, initial):
        path = self.folder/'state-latest/progress.json'
        if not path.exists(): return initial, 0, 0
        progress = read(path)
        assert progress['world_size'] == self.accelerator.num_processes
        assert progress['seed'] == self.args.seed
        self.accelerator.load_state(str(path.parent))
        self.microbatches = progress['microbatches']
        self.rng_chain = read(path.parent/('stream_rank%d.json'%self.rank))['rng_chain']
        if self.accelerator.is_main_process:
            log = self.folder/'training_log.csv'
            if log.exists():
                with log.open(newline='') as f: rows=list(csv.DictReader(f))
                rows=[r for r in rows if int(r['update'])<=progress['updates']]
                # 保留被回滚日志，恢复点之后的记录不能混入正式统计。
                write(self.folder/('recovery_discarded_%s.json'%os.environ.get('SLURM_JOB_ID','local')),
                      dict(resume_update=progress['updates'], old_log_sha256=sha(log)))
                if rows:
                    from tools.e2a_attr_common import csv_write
                    csv_write(log, rows)
        self.accelerator.wait_for_everyone()
        return progress['updates'], progress['epoch'], progress['offset']

    def record_rng(self, indices, noise, timesteps):
        if not self.enabled: return
        digest = hashlib.sha256(bytes.fromhex(self.rng_chain))
        for tensor in (indices, timesteps, noise):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(repr(random.getstate()).encode())
        self.rng_chain = digest.hexdigest(); self.microbatches += 1

    def record_losses(self, *values):
        if self.enabled: self.losses.append(torch.stack([v.detach().float() for v in values]))

    def check_update(self, optimizer, step):
        if not self.enabled: return
        skipped = int(self.accelerator.optimizer_step_was_skipped)
        skipped = self.accelerator.reduce(torch.tensor(skipped, device=self.accelerator.device), reduction='sum')
        if skipped.item():
            if self.accelerator.is_main_process:
                write(self.folder/'overflow_failure.json', dict(at_update=step+1, completed_updates=step,
                    skipped_ranks=int(skipped.item()), policy='run rejected; restart all paired arms with one common reviewed setting'))
            raise RuntimeError('AMP_SKIPPED_UPDATE: paired budget/RNG audit requires a clean run')

    def log_update(self, step, optimizer, grad_norm):
        if not self.enabled: return
        pause_flag=float(self.soft_end is not None and time.time()>=self.soft_end and step<self.args.max_train_steps)
        vector=torch.cat([torch.stack(self.losses).mean(0),torch.tensor([pause_flag],device=self.accelerator.device)])
        values = self.accelerator.reduce(vector, reduction='mean').cpu().tolist()
        self.paused = values.pop()>0
        self.losses.clear()
        if self.accelerator.is_main_process:
            names = ['total','denoise','gram','edge','rgb_global','body_rgb','boundary','outside']
            row = dict(update=step, **dict(zip(names,values)), lr=optimizer.param_groups[0]['lr'],
                grad_norm=float(grad_norm), microbatches_per_rank=self.microbatches, rng_rank0=self.rng_chain,
                elapsed_seconds=time.time()-self.started, job=os.environ.get('SLURM_JOB_ID'))
            path=self.folder/'training_log.csv'; exists=path.exists()
            with path.open('a', newline='', encoding='utf-8') as handle:
                writer=csv.DictWriter(handle, fieldnames=list(row))
                if not exists: writer.writeheader()
                writer.writerow(row)
            if step%100==0 or step==1:
                write(self.folder/'optimizer_update_count.json', dict(actual_successful_updates=step,
                    skipped_updates=0, attempted_updates=step, target_updates=self.args.max_train_steps,
                    elapsed_seconds=time.time()-self.started, world_size=self.accelerator.num_processes))

    def save_state(self, step, epoch, offset, force=False):
        if not self.enabled or (step%1000 and not force) or step==self.args.max_train_steps: return
        folder=self.folder/'state-latest'
        self.accelerator.save_state(str(folder), safe_serialization=False)
        write(folder/('stream_rank%d.json'%self.rank), dict(rng_chain=self.rng_chain))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            write(folder/'progress.json', dict(updates=step, epoch=epoch, offset=offset,
                seed=self.args.seed, world_size=self.accelerator.num_processes, microbatches=self.microbatches))
        self.accelerator.wait_for_everyone()

    def finish_pause(self, step):
        if self.accelerator.is_main_process:
            write(self.folder/'pauses'/('job_%s.json'%os.environ.get('SLURM_JOB_ID','local')),
                dict(clean_pause=True,retained_successful_updates=step,resume_state=str(self.folder/'state-latest'),
                     replayed_or_discarded_updates=0,reason='yield before Slurm walltime; resume complete optimizer/scheduler/RNG state'))
        self.accelerator.wait_for_everyone()

    def finish(self, step):
        if not self.enabled: return
        write(self.folder/('seed_and_rng_rank%d.json'%self.rank), dict(seed=self.args.seed,
            process_seed=self.args.seed+self.rank, rank=self.rank, rng_chain=self.rng_chain,
            microbatches=self.microbatches, actual_updates=step))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            streams=[read(self.folder/('seed_and_rng_rank%d.json'%r)) for r in range(self.accelerator.num_processes)]
            write(self.folder/'seed_and_rng_manifest.json', dict(streams=streams,
                order='torch.randperm(seed+epoch), independent of model RNG', dropout='historical Python random',
                augmentation='plain_resize; no random augmentation'))
            write(self.folder/'optimizer_update_count.json', dict(actual_successful_updates=step,
                attempted_updates=step, skipped_updates=0, target_updates=self.args.max_train_steps,
                world_size=self.accelerator.num_processes, complete=step==self.args.max_train_steps))
            write(self.folder/'training_complete.json', dict(complete=True, updates=step, seed=self.args.seed))
