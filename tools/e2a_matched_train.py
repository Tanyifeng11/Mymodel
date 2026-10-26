"""复用冻结训练 hook；只将两域系数选取改为匹配 C 的区域梯度。"""
import argparse
import csv
import numpy as np
import torch
from tools import e2a_sup_train as engine
from tools.e2a_matched_common import *

BaseHook=engine.Hook


class MatchedSupervision:
    def __init__(self,arm,helpers):
        assert arm=='T3M'
        self.h={k:helpers[k] for k in ['build_region_masks','texture_color_stat_loss','masked_l1_loss']}
        self.network=None

    def raw(self,pred,gt,reference,mask,timestep,weight):
        body,boundary,outside=self.h['build_region_masks'](mask,9)
        assert torch.allclose(body+boundary+outside,torch.ones_like(body),atol=1e-6,rtol=0)
        body_loss=pred.sum()*0
        if weight.sum().item()>0:
            body_loss=self.h['texture_color_stat_loss'](pred,reference,body,sample_weight=weight)
        return dict(body_rgb=body_loss,rest=self.h['masked_l1_loss'](pred,gt,1-body),
            # 仅校准/日志诊断；训练权重表不包含此项，不能叠加 C 的区域损失。
            c_region=.05*self.h['masked_l1_loss'](pred,gt,boundary)+.1*self.h['masked_l1_loss'](pred,gt,outside))


def select_gamma(rank_rows):
    assert len(rank_rows)==2 and all(len(x)==256 for x in rank_rows)
    batches=[]
    for i in range(64):
        batch=[row for rows in rank_rows for row in rows[4*i:4*i+4]]
        c=np.sqrt(np.mean([float(r['c_region_rms'])**2 for r in batch]))
        rest=np.sqrt(np.mean([float(r['rest_rms'])**2 for r in batch]))
        assert np.isfinite(c) and c>0 and np.isfinite(rest) and rest>0
        batches.append(dict(effective_batch=i+1,samples=8,C_weighted_region_rms=float(c),
            unweighted_rest_rms=float(rest),gamma_ratio=float(c/rest)))
    gamma=float(np.median([r['gamma_ratio'] for r in batches]));assert np.isfinite(gamma) and gamma>0
    for r in batches:
        r['matched_ratio']=gamma/r['gamma_ratio']
        r['C_to_matched_rest_ratio']=r['gamma_ratio']/gamma
    return gamma,batches


class MatchedHook(BaseHook):
    def setup(self,env):
        super().setup(env)
        args=env['args']
        assert args.seed==42 and args.start_global_step==0 and not args.resume_from_checkpoint
        assert args.gam_init_ckpt==str(model_path('E1'))
        assert args.gradient_accumulation_steps==4 and args.train_batch_size==1
        assert args.texture_condition_mode=='token'
        assert args.lambda_region_texture==args.lambda_boundary==args.lambda_leak==0
        assert args.max_train_steps==(800 if self.phase=='pilot' else 28365)
        if self.phase!='calibrate':
            current=read(self.run/('environment_rank%d.json'%self.rank))
            baseline=read(folder('T3M','calibrate')/('environment_rank%d.json'%self.rank))
            current.pop('phase');baseline.pop('phase')
            assert current==baseline,'CALIBRATION_TRAINING_ENVIRONMENT_CHANGED'
        self.scheduler=env['lr_scheduler']
        self.scheduler_before=copy.deepcopy(self.scheduler.state_dict())
        self.scaler_before=copy.deepcopy(self.acc.scaler.state_dict())
        binding=dict(passed=True,implementation='MatchedHook / MatchedSupervision',
            source_sha256={p:sha(SOURCE/p) for p in TRAINING_SOURCES},
            body_exact_C=True,body_coefficient=.1,separate_C_region_terms_disabled=True,
            c_region_diagnostic_only=True,time_gate=False,
            gamma=None if self.phase=='calibrate' else self.coefficients['rest'])
        freeze(self.run/('matched_binding_rank%d.json'%self.rank),binding)

    @torch.no_grad()
    def selfcheck(self,env):
        p,g,r,m=[env[k].detach() for k in ['decoded_loss','target_loss','texture_target_loss','mask_loss']]
        t=env['timesteps'];w=torch.ones(p.shape[0],device=p.device)
        a=self.impl.raw(p,g,r,m,t,w);b=self.impl.raw(p,g,-r,m,t,w)
        assert torch.equal(a['rest'],b['rest']) and torch.equal(a['c_region'],b['c_region'])
        assert not torch.allclose(a['body_rgb'],b['body_rgb'],atol=1e-7,rtol=0)
        body,boundary,outside=self.impl.h['build_region_masks'](m,9)
        assert torch.equal(1-body,boundary+outside)
        direct=self.impl.h['texture_color_stat_loss'](p,r,body,sample_weight=w)
        assert torch.equal(a['body_rgb'],direct)
        dropped=self.impl.raw(p,g,r,m,t,torch.zeros_like(w))
        assert dropped['body_rgb'].item()==0 and torch.equal(a['rest'],dropped['rest'])
        for fill in [0,1]:
            values=self.impl.raw(p,g,r,torch.full_like(m,fill),t,w)
            assert all(torch.isfinite(v).item() for v in values.values())
        write(self.run/('loss_semantics_rank%d.json'%self.rank),dict(passed=True,
            body_exact_C_function=True,reference_swap_correct=True,body_condition_dropout_preserved=True,
            masks_disjoint_and_cover=True,rest_equals_boundary_union_outside=True,
            empty_full_masks_finite=True,c_region_diagnostic_only=True))

    def finish_calibration(self):
        assert not self.optimizer.state
        assert self.scheduler.state_dict()==self.scheduler_before
        assert self.acc.scaler.state_dict()==self.scaler_before
        assert all(torch.equal(p.detach().cpu(),b) for p,b in zip(self.probes,self.probe_before))
        assert set(self.parameter_probe)=={'body_rgb','rest','c_region'}
        csv_write(self.run/('calibration_rank%d.csv'%self.rank),self.rows)
        write(self.run/('gradient_probe_rank%d.json'%self.rank),self.parameter_probe)
        self.acc.wait_for_everyone()
        if self.acc.is_main_process:
            rows=[]
            for rank in range(2):
                with (self.run/('calibration_rank%d.csv'%rank)).open() as f:rows.append(list(csv.DictReader(f)))
                assert read(self.run/('loss_semantics_rank%d.json'%rank))['passed']
                assert set(read(self.run/('gradient_probe_rank%d.json'%rank)))=={'body_rgb','rest','c_region'}
            assert all(np.isfinite(float(v)) for rank in rows for row in rank for k,v in row.items() if k!='rng_chain')
            with (OLD/'training/seed42/F000/training_log.csv').open() as f:base=list(csv.DictReader(f))[:64]
            assert all(rows[0][4*i+3]['rng_chain']==base[i]['rng_rank0'] for i in range(64))
            gamma,batches=select_gamma(rows)
            coefficients={'body_rgb':.1,'rest':gamma}
            evidence=dict(passed=True,formula=CONFIG['gamma_selection'],gamma=gamma,
                batches=batches,samples=512,effective_batches=64,optimizer_updates=0,
                body_coefficient_fixed=.1,source_commit=commit(),used_P2_or_LOCK=False,
                actual_trainable_gradient_probes=True,
                source_sha256={p:sha(SOURCE/p) for p in TRAINING_SOURCES},
                matched_ratio_median=float(np.median([b['matched_ratio'] for b in batches])),
                C_to_matched_rest_ratio_median=float(np.median([b['C_to_matched_rest_ratio'] for b in batches])),
                texture_condition_active_rate=float(np.mean([float(r['texture_condition_active']) for rank in rows for r in rank])),
                calibration_files={('rank%d'%r):sha(self.run/('calibration_rank%d.csv'%r)) for r in range(2)})
            freeze(ROOT/'gamma_calibration.json',evidence)
            freeze(ROOT/'loss_registry.json',dict(T3M=dict(coefficients=coefficients,
                samples=512,optimizer_updates=0,source_commit=commit(),used_P2_or_LOCK=False)))
            write(self.run/'calibration_complete.json',dict(passed=True,samples=512,effective_batches=64,
                optimizer_updates=0,scheduler_updates=0,scaler_updates=0,
                coefficients=coefficients,gamma_calibration_sha256=sha(ROOT/'gamma_calibration.json')))
            write(self.run/'optimizer_update_count.json',dict(actual_successful_updates=0,
                attempted_updates=0,skipped_updates=0,scheduler_updates=0,scaler_updates=0,
                calibration_complete=True,world_size=2))
            print('MATCHED_GAMMA_PASSED',gamma,flush=True)
        self.acc.wait_for_everyone()
        raise engine.CalibrationDone()


def train(phase):
    prep=read(ROOT/'audit/preparation_complete.json');assert prep['passed']
    assert all(sha(SOURCE/p)==h for p,h in prep['source_sha256'].items())
    # 只在新进程内绑定新根和新 loss；原文件字节、旧输出、主随机流不变。
    engine.ROOT=ROOT;engine.CONFIG=CONFIG;engine.folder=folder;engine.freeze=freeze
    engine.Supervision=MatchedSupervision;engine.Hook=MatchedHook
    engine.train('T3M',phase)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['T3M'],default='T3M')
    parser.add_argument('--phase',choices=['calibrate','pilot','formal'],required=True)
    train(parser.parse_args().phase)
