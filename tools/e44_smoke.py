"""真实 8 例完整 DDIM50，随后检查 step1/5/20 的端到端梯度。"""
import gc
import torch
from tools.e34_sarr_protocol import seed_all
from tools.e44_protocol import *
from tools.e44_runtime import Runtime,sketch_masks
from PIL import Image
from tools.e44_train import TrainingInputs,loss_forward,gradients


def run():
    prepare();rows=read(OUT/'splits/dev32.json')[:8];replay={}
    mask_audit=[dict(id=row['id'],**sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE),'cpu')[2]) for row in rows]
    write(OUT/'p0/mask_audit.json',mask_audit)
    diagnostic=next(i for i,info in enumerate(mask_audit) if info['valid'])
    seed_all(42);r=Runtime('B0')
    try:
        for row in rows:
            path=OUT/'p0/B0'/(row['id']+'.png')
            record,_,_=r.generate(row,path)
            old=Path('output_eval/e34_sarr_20261010/s0_e5_dev')/row['id']/'I0.png'
            assert sha(old)==record['png_sha256'],'历史 E5 RGB 回放不一致'
            replay[row['id']]=record
        write(OUT/'p0/baseline_replay.json',replay)
    finally:r.close()
    del r;gc.collect();torch.cuda.empty_cache()
    checks={};grads={}
    for arm in ['B2','B3','B1']:
        seed_all(42);r=Runtime(arm);checks[arm]=[]
        initial={k:v.clone() for k,v in r.processor.trainable_state().items()}
        try:
            for i,row in enumerate(rows):
                r.active=False
                off,contexts,outputs=r.generate(row,OUT/'p0'/arm/(row['id']+'_off.png'),capture=i==diagnostic)
                r.active=True
                zero,_,_=r.generate(row,OUT/'p0'/arm/(row['id']+'_zero.png'))
                assert off['png_sha256']==zero['png_sha256']==replay[row['id']]['png_sha256']
                assert off['final_sha256']==zero['final_sha256']==replay[row['id']]['final_sha256']
                checks[arm].append(dict(id=row['id'],off_exact=True,zero_exact=True,mask=zero['mask']))
                if i==diagnostic:
                    assert r.processor.trace, '首例必须实际通过新增分支'
                    assert r.processor.trace[0]['target_shape']==[1,768,640],r.processor.trace[0]
                    write(OUT/'p0'/('shapes_'+arm+'.json'),r.processor.trace)
                    f=r.processor.features.clone()
                    assert torch.isfinite(f).all() and f.shape[1]==256
                    # 非零调试只用于验证真实条件影响，不作为训练结果。
                    with torch.no_grad():
                        if arm=='B1':r.processor.gate.fill_(.1)
                        else:torch.nn.init.normal_(r.processor.output.weight,std=.02)
                        interventions={}
                        for branch in ['conditional','unconditional']:
                            args,kw=contexts[branch]
                            pred=r.pipe.unet(*args,**kw)[0]
                            delta=float((pred.float()-outputs[branch].float()).abs().mean())
                            assert delta>0 if branch=='conditional' else delta==0
                            interventions[branch]=dict(eps_mae=delta)
                        write(OUT/'p0'/('cfg_'+arm+'.json'),interventions)
                    r.processor.load_state_dict(initial,strict=False)
                    # 冻结特征必须随 Reference 改变；下一例回放也核验逐步提取。
                    donor=Image.open(rows[(i+1)%len(rows)]['reference']).convert('RGB')
                    r.generate(row,OUT/'p0'/arm/'feature_change.png',reference=donor)
                    assert not torch.equal(f,r.processor.features)
                    write(OUT/'p0'/('feature_'+arm+'.json'),dict(finite=True,tokens=256,
                        reference_changed=True,diagnostic_id=row['id'],writer_calls=r.writer.calls if r.writer else 0))
                write(OUT/'p0/replay_checks.json',checks)
                print('E44 P0',arm,i+1,'/8',flush=True)
            data=TrainingInputs(r);seed_all(42)
            optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda');reports=[]
            for step in range(1,21):
                # mask失效的输入保留在训练池；梯度诊断选首个有效输入并记录跳过数量。
                skipped=0
                while True:
                    args,kw,noise,mask,record=data.next(disable_dropout=True)
                    if r.mask_info['valid']:break
                    skipped+=1
                optimizer.zero_grad(set_to_none=True)
                loss,stats=loss_forward(r,args,kw,noise,mask)
                assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer)
                g=gradients(r);first='gate' if arm=='B1' else 'output.weight'
                assert g[first] is not None and g[first]>0
                if step>=5:assert g['to_k_ref.weight']>0 and g['to_v_ref.weight']>0
                if step in [1,5,20]:reports.append(dict(step=step,gradients=g,**stats,input=record,skipped_invalid=skipped))
                torch.nn.utils.clip_grad_norm_(r.parameters(),1.);scaler.step(optimizer);scaler.update()
            grads[arm]=reports;write(OUT/'p0/gradient_audit.json',grads)
            r.processor.load_state_dict(initial,strict=False)
            r.save(OUT/'p0'/arm/'step0000.pt',0)
        finally:r.close()
        del r;gc.collect();torch.cuda.empty_cache()
    verify_inputs()
    write(OUT/'p0/smoke_report.json',dict(pass_p0=True,replay_n=8,off_exact=True,zero_exact=True,
        gradient_valid=True,cfg_unconditional_exact=True,keep_weight=.1,
        note='BF-local and architecture-matched frozen Writer; not original MimicBrush weights'))
    decision(P0='pass')


if __name__=='__main__':run()
