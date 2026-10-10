"""真实 E5 上的零初始化回放、两支 trace、干预和梯度核验。"""
import time
import torch
from torch.nn import functional as F
from tools.e36_protocol import *
from tools.e36_infer import Generator
from tools.e36_train import TrainingInputs,build_adapters,gradients


def run():
    from tools.e34_sarr_protocol import seed_all
    seed_all(42);prepare();g=Generator();checks=[];intervention={};gradient_report={}
    try:
        rows=read(OUT/'splits/dev32.json')[:8]
        for i,row in enumerate(rows):
            folder=OUT/'p0/replay'/row['id'];g.bridge.reset()
            off,args,outputs=g.generate(row,'A0_E5_OFF',folder/'OFF1.png',capture_first=i==0)
            if i==0:
                write(OUT/'p0/hook_trace.json',g.bridge.report())
                g.adapters,counts=build_adapters(g);write(OUT/'p0/parameter_match.json',counts)
            again,_,_=g.generate(row,'A0_E5_OFF',folder/'OFF2.png')
            old=PREVIOUS/'s0_e5_dev'/row['id']/'I0.png'
            assert off['png_sha256']==again['png_sha256']==sha(old)
            assert off['final_latent_sha256']==again['final_latent_sha256']
            records={}
            for arm in ARMS:
                g.bridge.reset();record,_,_=g.generate(row,arm,folder/(arm+'_ZERO.png'))
                assert record['png_sha256']==off['png_sha256']
                assert record['final_latent_sha256']==off['final_latent_sha256']
                trace=g.bridge.report();assert trace['counts']['conditional/0']==50 and trace['counts']['unconditional/0']==50
                records[arm]=dict(exact=True,trace=trace)
            restored,_,_=g.generate(row,'A0_E5_OFF',folder/'OFF_AFTER.png')
            assert restored['png_sha256']==off['png_sha256'] and restored['final_latent_sha256']==off['final_latent_sha256']
            checks.append(dict(id=row['id'],replay_exact=True,zero_exact=records,restored_exact=True))
            write(OUT/'p0/replay_checks.json',checks);print('P0 REPLAY',i+1,'/8',flush=True)
            if i==0:
                with torch.no_grad():
                    for arm,adapter in g.adapters.items():
                        torch.nn.init.normal_(adapter.head.weight,std=.02);adapter.collect=True
                        g.bridge.adapter=adapter
                        for branch in ('conditional','unconditional'):
                            a,kw=args[branch];torch.cuda.synchronize();start=time.perf_counter()
                            prediction=g.pipe.unet(*a,**kw)[0]
                            torch.cuda.synchronize();delta=float((prediction.float()-outputs[branch].float()).abs().mean())
                            assert torch.isfinite(prediction).all()
                            assert delta>0 if branch=='conditional' else delta==0
                            intervention[arm+'/'+branch]=dict(eps_mae=delta,seconds=time.perf_counter()-start,**adapter.stats)
                        adapter.head.weight.zero_();adapter.head.bias.zero_()
                    g.bridge.adapter=None
                write(OUT/'p0/intervention.json',intervention)
        data=TrainingInputs(g);seed_all(42)
        # 真正的端到端 epsilon MSE，而非直接对 adapter 输出伪造梯度。
        for arm,adapter in g.adapters.items():
            optimizer=torch.optim.AdamW(adapter.parameters(),lr=.0001)
            scaler=torch.amp.GradScaler('cuda');reports=[];g.bridge.adapter=adapter
            for step in range(1,6):
                a,kw,noise,sketch,_=data.next(batch_size=2 if step==1 else 1)
                g.bridge.sketch=sketch;optimizer.zero_grad(set_to_none=True);adapter.collect=True
                with torch.autocast('cuda',dtype=torch.float16):prediction=g.pipe.unet(*a,**kw).sample
                loss=F.mse_loss(prediction.float(),noise.float());scaler.scale(loss).backward();scaler.unscale_(optimizer)
                grads=gradients(adapter);assert grads['head.weight']>0
                if step==5:
                    assert grads['project.weight']>0 and grads['guide.0.weight']>0
                    if arm!='B1_CONV':assert grads['ks.weight']>0 and grads['kf.weight']>0 and grads['mix.weight']>0
                    if arm=='B3_CONFLICT':assert grads['edge.weight']>0
                reports.append(dict(step=step,batch=2 if step==1 else 1,loss=float(loss),gradients=grads,stats=adapter.stats,
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);scaler.step(optimizer);scaler.update()
            gradient_report[arm]=reports
            write(OUT/'p0/gradients.json',gradient_report)
        write(OUT/'p0/no_edit_threshold.json',dict(source='8 actual repeated OFF runs; RGB and latent exactly equal',
            rgb_mae_floor=0.,latent_mae_floor=0.,eps_mae_floor=0.,policy='zero measured floor; trained intervention must be strictly nonzero'))
        write(OUT/'p0/smoke_report.json',dict(pass_p0=True,replay_ids=8,parameter_match=counts,
            replay_pass=True,hook_trace_pass=True,grad_pass=True,cfg_adapter_policy=CONFIG['cfg_adapter_policy']))
        decision(p0_replay_pass=True,p0_hook_trace_pass=True,p0_grad_pass=True)
    finally:g.close()


if __name__=='__main__':run()
