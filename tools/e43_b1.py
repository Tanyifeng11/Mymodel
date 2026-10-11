"""只读复用E36已有B1；独立进程和旁表，不影响E43三条主分支。"""
import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e43_protocol import *
from tools.e43_runtime import Runner
from tools.e43_metrics import metric_record
from tools.e36_runtime import SkipBridge


class B1Runner(Runner):
    def call(self,x,t,kw):
        # Runner直接调用forward，需在真实UNet调用前重置原E36 hook的计数。
        self.bridge.index=0;self.bridge.timestep=int(t.flatten()[0]) if torch.is_tensor(t) else int(t)
        self.bridge.branch='conditional' if 'sa_hidden_states' in (kw.get('cross_attention_kwargs') or {}) else 'unconditional'
        return super().call(x,t,kw)


def collect(phase,shard,shards):
    from tools.e36_protocol import CONFIG as B1_CONFIG
    from models.e36_guided_kernel_filter import GuidedKernelFilter
    from tools.e33tmoc_appearance_eval import frozen_lpips,model_hash
    path=Path('output_eval/e36_dagf_guided_filter_20261010/checkpoints/B1_CONV/step0800.pt')
    if not path.exists():
        write(OUT/phase/('b1_shard%d.json'%shard),dict(available=False,reason='existing checkpoint missing; no retraining'))
        return
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert checkpoint['step']==800 and checkpoint['arm']=='B1_CONV' and checkpoint['config']==B1_CONFIG
    sd=checkpoint['state_dict'];adapter=GuidedKernelFilter(sd['norm.weight'].numel(),sd['time.weight'].shape[1],'B1_CONV').cuda()
    adapter.load_state_dict(sd,strict=True);adapter.eval().requires_grad_(False);before=model_hash(adapter)
    run=B1Runner();run.bridge=SkipBridge(run.p.unet);metric,info=frozen_lpips();metric.cuda()
    rows=read(OUT/'split_sources.json')[phase];replay=None
    try:
        for i,row in enumerate(rows):
            if i%shards!=shard:continue
            for seed in CONFIG['seeds']:
                folder=OUT/phase/'samples'/row['id']/('seed%d'%seed);original=read(folder/'complete.json')['outputs'][0]
                if (folder/'b1_complete.json').exists():continue
                if replay is None:
                    run.bridge.adapter=None
                    image,r=run.baseline(row,seed,instrument=False)
                    assert np.array_equal(np.asarray(image),np.asarray(Image.open(original['output_png'])))
                    assert r['trajectory_hashes']==original['trajectory_hashes']
                    replay=dict(id=row['id'],seed=seed,adapter_off_50step_and_png_equal=True)
                run.bridge.reset();run.bridge.adapter=adapter
                sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
                run.bridge.sketch=(to_tensor(sketch)[None]*2-1).to(run.p.device,run.p.unet.dtype)
                image,r=run.baseline(row,seed,instrument=False)
                assert r['initial_hash']==original['initial_hash'] and r['unet_calls']==100
                trace=run.bridge.report();assert any(v['applied'] for v in trace['trace'])
                image.save(folder/'E36_B1.png');png=folder/'E36_B1.png'
                output=dict(r,arm='E36_B1',output_png=str(png),output_png_sha256=sha(png),metrics=metric_record(png,row,metric),
                    id=row['id'],group_id=row['group_id'],seed=seed,alpha=None,effective_alpha=None)
                write(folder/'b1_complete.json',dict(output=output,checkpoint_sha256=sha(path),adapter_hash=before,
                    adapter_training_steps_this_experiment=0,hook_trace=trace,slurm_job_id=os.environ.get('SLURM_JOB_ID')))
                print('E43 B1',phase,i,row['id'],seed,flush=True)
        assert model_hash(adapter)==before
        write(OUT/phase/('b1_shard%d.json'%shard),dict(available=True,replay=replay,checkpoint_sha256=sha(path),
            adapter_hash_unchanged=True,adapter_training_steps_this_experiment=0,frozen=run.frozen(),**verify_inputs()))
    finally:run.bridge.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','holdout'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(2);collect(a.phase,a.shard,a.shards)
