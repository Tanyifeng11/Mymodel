"""单卡真实计算图工程测试：16样本、零更新，不选择γ或替代双卡配对校准。"""
import copy
import math
import os
from tools import e2a_sup_train as engine
from tools.e2a_matched_train import MatchedSupervision,MatchedHook
from tools.e2a_matched_common import *


class SingleGPUHook(engine.Hook):
    selfcheck=MatchedHook.selfcheck

    def setup(self,env):
        self.acc=env['accelerator'];self.runtime=env['runtime'];self.rank=self.acc.process_index
        self.models=[env[k] for k in ['unet','ref_unet','bf','spatial_texture_encoder','spatial_injection']]
        self.optimizer=env['optimizer'];self.scheduler=env['lr_scheduler']
        assert self.acc.num_processes==1 and engine.torch.cuda.device_count()==1
        # 单进程 Accelerate 可返回 cuda（index=None）；继承的 fork_rng 需要显式编号。
        assert self.acc.device.type=='cuda'
        self.acc.state.device=engine.torch.device('cuda',engine.torch.cuda.current_device())
        assert self.acc.device.index is not None
        assert env['args'].gam_init_ckpt==str(model_path('E1')) and env['args'].start_global_step==0
        assert not env['args'].resume_from_checkpoint
        assert env['noise_scheduler'].config.prediction_type=='epsilon'
        assert not any(p.requires_grad for p in env['vae'].parameters())
        params=env['trainable_params']
        self.probes=[params[i] for i in sorted({0,len(params)//2,len(params)-1})]
        self.probe_before=[p.detach().cpu().clone() for p in self.probes]
        self.scheduler_before=copy.deepcopy(self.scheduler.state_dict())
        self.scaler_before=copy.deepcopy(self.acc.scaler.state_dict())
        freeze(self.run/'single_gpu_environment.json',dict(world_size=1,visible_GPUs=1,
            resolved_cuda_index=self.acc.device.index,
            torch=engine.torch.__version__,seed=env['args'].seed,batch_size=env['args'].train_batch_size,
            gradient_accumulation=env['args'].gradient_accumulation_steps,
            optimizer=type(self.optimizer.optimizer).__name__,no_optimizer_updates=True,
            inference_or_formal_weights_produced=False,paired_two_rank_RNG_claimed=False))

    def __call__(self,env,namespace):
        result=super().__call__(env,namespace)
        if len(self.rows)==16:self.finish_calibration()
        return result

    def finish_calibration(self):
        assert len(self.rows)==16 and not self.optimizer.state
        assert self.scheduler.state_dict()==self.scheduler_before
        assert self.acc.scaler.state_dict()==self.scaler_before
        assert all(engine.torch.equal(p.detach().cpu(),b) for p,b in zip(self.probes,self.probe_before))
        assert set(self.parameter_probe)=={'body_rgb','rest','c_region'}
        assert all(math.isfinite(float(v)) for r in self.rows for k,v in r.items() if k!='rng_chain')
        assert read(self.run/'loss_semantics_rank0.json')['passed']
        csv_write(self.run/'gradient_trace.csv',self.rows)
        write(self.run/'gradient_probe.json',self.parameter_probe)
        evidence=dict(passed=True,job=os.environ['SLURM_JOB_ID'],source_commit=commit(),
            run=str(self.run),samples=16,physical_GPUs=1,world_size=1,
            forward_backward_graph_verified=True,actual_trainable_gradient_probes=True,
            optimizer_updates=0,scheduler_updates=0,scaler_updates=0,
            peak_allocated_bytes=engine.torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=engine.torch.cuda.max_memory_reserved(),
            gamma_selected=False,weights_used_for_formal=False,paired_two_rank_RNG_claimed=False,
            formal_budget_changed=False,trace_sha256=sha(self.run/'gradient_trace.csv'),
            semantics_sha256=sha(self.run/'loss_semantics_rank0.json'),
            source_sha256={p:sha(SOURCE/p) for p in TRAINING_SOURCES+['tools/e2a_matched_smoke.py']})
        freeze(ROOT/'audit/single_gpu_test_complete.json',evidence)
        print('SINGLE_GPU_GRAPH_TEST_PASSED',16,'ZERO_UPDATES',flush=True)
        raise engine.CalibrationDone()


def run():
    request=read(ROOT/'audit/single_gpu_test_request.json')
    assert request['passed']
    if request['source_commit']!=commit():
        job=next(j for j in read(ROOT/'jobs.json') if str(j['id'])==os.environ['SLURM_JOB_ID'])
        review=read(ROOT/'audit'/('retry_release_%d.json'%job['retry_of']))
        assert review['passed'] and review['source_commit']==commit()
        assert review['key']=='single_gpu_graph_test' and review['failed_job']==job['retry_of']
        assert review['request_sha256']==sha(ROOT/'audit/single_gpu_test_request.json')
    prep=read(ROOT/'audit/preparation_complete.json');assert prep['passed']
    assert all(sha(SOURCE/p)==h for p,h in prep['source_sha256'].items())
    assert not (ROOT/'audit/single_gpu_test_complete.json').exists()
    run=ROOT/'engineering/single_gpu_test'/('job_'+os.environ['SLURM_JOB_ID'])
    def test_freeze(path,value):
        if Path(path).name=='config.json':
            value=dict(value,world_size=1,effective_batch=4,updates=0,
                engineering_test_only=True,samples=16,gamma_selection=False,
                formal_or_paired_calibration=False)
        return freeze(path,value)
    engine.ROOT=ROOT;engine.CONFIG=CONFIG;engine.folder=lambda arm,phase:run
    engine.freeze=test_freeze;engine.Hook=SingleGPUHook;engine.Supervision=MatchedSupervision
    engine.train('T3M','calibrate')


if __name__=='__main__':run()
