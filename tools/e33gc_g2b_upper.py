"""仅fit01/02、固定80更新，同hook/mask/幅值的特权空间残差；不是候选方法。"""
import time,json
import torch,numpy as np,cv2
from torch import nn
from torch.nn import functional as F
from tools.e33gc_g2b_train import *
from tools.e33gc_g2b_eval import dft_pair

class FreeResidual(nn.Module):
    def __init__(self,cases):
        super().__init__();self.values=nn.ParameterList([nn.Parameter(torch.zeros(1,320,64,48,device='cuda')) for _ in range(6)])
        # prefix field张量地址只用于特权身份/arm选择；checkpoint重算时读取捕获的geometry，避免最后一臂覆盖。
        self.lookup={case['geom'][i:i+1].data_ptr():j*3+i for j,case in enumerate(cases) for i in range(3)}
    def forward(self,h,texture,geometry,falloff):
        residual=self.values[self.lookup[geometry.data_ptr()]].tanh()*.1
        mask=F.interpolate(falloff,(64,48),mode='bilinear',align_corners=False)
        return (residual*mask).flatten(2).transpose(1,2)

def run_upper():
    init();assert read(OUT/'decision_summary.json')['g2b_AI_amended_fit_pass'] is False
    smoke=read(OUT/'G2b_smoke/smoke_audit.json');assert smoke['pass_autograd'] and smoke['budget_pass']
    torch.manual_seed(42);torch.set_num_threads(2);cv2.setNumThreads(1)
    rows=read(OUT/'protocol/g2b_fit_probe_ids.json')['fit'][:2]
    assert [r['label'] for r in rows]==['fit_01','fit_02']
    pipe,modules,size,ns=load_e5();before=module_hashes(modules)
    rf,_=build(42,checkpoint=RF/'seed42/RF2/checkpoint_final.pt');rf.eval().requires_grad_(False)
    dino,_=load_dino('cuda',WEIGHTS);dino.eval().requires_grad_(False)
    injection=TracedInjection(pipe,CausalResidual().cuda())
    cases=[cache_case(row,pipe,ns,injection.adapter,injection,rf,dino) for row in rows]
    adapter=FreeResidual(cases);injection.adapter=adapter
    opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4)
    train_hours=read(OUT/'G2b_train/training_complete.json')['elapsed_seconds']/3600
    budget_left=6-train_hours-smoke['full4_identity_update_seconds']/3600*10
    began=time.monotonic();dest=OUT/'G2b_upper_bound_if_needed';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'checkpoint_step80.pt').exists()
    for step in range(1,81):
        opt.zero_grad(set_to_none=True);parts=[]
        for case in cases:
            images=render_case(case,pipe,ns,injection);loss,values=losses(images,case['targets'],case['baseline'],case['masks'])
            (loss/2).backward();parts.append({k:float(v.detach()) for k,v in values.items()})
        norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);assert torch.isfinite(norm);opt.step()
        with (dest/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,losses=parts,grad_norm=float(norm),seconds=time.monotonic()-began))+'\n')
        print('UPPER',step,'seconds',round(time.monotonic()-began,2),flush=True)
        assert (time.monotonic()-began)/3600<budget_left,'combined GPU budget exhausted'
    torch.save(dict(residual=adapter.state_dict(),privileged_identity_and_arm=True,updates=80),dest/'checkpoint_step80.pt')
    diagnostic(cases,pipe,ns,injection,80,label='G2b_upper_bound_if_needed')
    results=[]
    for case in cases:
        folder=dest/'step80'/case['row']['id']
        metrics=dft_pair({arm:Image.open(folder/(arm+'.png')).convert('RGB') for arm in ARMS},case['row']['dft_boxes'])
        results.append(dict(id=case['row']['id'],label=case['row']['label'],DFT=metrics))
    numerical=all(r['DFT']['r90_success'] and r['DFT']['r180_success'] for r in results)
    write(dest/'diagnosis.json',dict(results=results,numerical_pass=numerical,visual_pass=None,updates=80,
        elapsed_seconds=time.monotonic()-began,checkpoint_sha256=sha(dest/'checkpoint_step80.pt'),
        privileged=True,same_site=True,alpha_max=.1,same_last8=True))
    decision(g2b_free_residual_upper_numerical_pass=numerical,g2b_free_residual_upper_pass=False if not numerical else None,
        next_route='fixed_protocol_failed_stop_no_extra_training')
    assert before==module_hashes(modules);injection.close();verify_frozen();bundle('upper')

if __name__=='__main__':run_upper()
