"""e48 回放验收、固定无训练矩阵和能量校准。"""
import argparse
import gc
import platform
import shutil
import torch
from tools.e48_protocol import *
from models.e48_interventions import Intervention


def make_runtime(namespace):
    from tools import e47_runtime as module
    previous=module.OUT;module.OUT=OUT
    try:r=module.runtime('B',True,namespace)
    finally:module.OUT=previous
    r.load(B1);r.processor.eval().requires_grad_(False)
    return r


def release(r):
    from tools.e47_runtime import close
    close(r);del r;gc.collect();torch.cuda.empty_cache()


def run_case(r,row,arm,stage,audit=False,diagnose=False,compare=True):
    from tools.e47_runtime import exact_trace
    path=image_path(arm,row['id'],stage)
    complete=path.with_name(path.stem+'_intervention.json')
    if complete.exists():
        saved=read(complete)
        assert saved['arm']==arm and saved['diagnose']==diagnose
        assert sha(path)==saved['generation']['png_sha256']
        return saved
    ctl=Intervention(r,row,arm,audit=audit,diagnose=diagnose)
    ctl.install()
    try:record,_,_=r.generate(row,path)
    finally:ctl.close()
    assert len(ctl.steps)==len(ctl.scheduler_steps)==ctl.unconditional_calls==50
    assert record['seed']==42 and not record['gt_used_in_inference']
    saved=dict(arm=arm,diagnose=diagnose,generation=record,steps=ctl.steps,scheduler_steps=ctl.scheduler_steps,
        calls=dict(teacher=ctl.teacher_module_calls,student=ctl.student_module_calls,
            unconditional=ctl.unconditional_calls),source_commit=commit())
    if compare and arm in ['B','D','SAFE']:
        historical=history(arm,row['id'])
        trace=exact_trace(path.with_suffix('.pt'),historical.with_suffix('.pt'))
        assert sha(path)==sha(historical) and trace['bit_exact'],(arm,row['id'],trace)
        saved['historical_exact']=True;saved['trace_check']=trace
    torch.save(torch.stack(ctl.trajectory),path.with_name(path.stem+'_latents.pt'))
    write(complete,saved)
    print('E48 RGB',stage,arm,row['id'],round(record['seconds'],3),flush=True)
    return saved


def audit():
    from tools.e22_4_generation import module_hashes
    import diffusers
    prepare()
    rows=read(OUT/'splits/probe.json');calibration=read(OUT/'splits/calibration.json')
    assert not (OUT/'audit/complete.json').exists(),'已完成的验收不可重复选择结果'
    r=make_runtime('audit/runtime');before=module_hashes({'b1':r.processor})
    try:
        calibration_results=[run_case(r,row,'D','calibration',compare=False) for row in calibration]
        all_energy=sum(s['raw_energy'] for v in calibration_results for s in v['steps'])
        late=sum(s['raw_energy'] for v in calibration_results for s in v['steps'] if s['index']>=35)
        assert all_energy>0
        write(OUT/'audit/calibration.json',dict(ids=[r['id'] for r in calibration],
            raw_energy=all_energy,late_energy=late,late_lambda=(late/all_energy)**.5,
            frozen_before_probe_metrics=True,new_training=0))
        values=[run_case(r,row,'D','audit',audit=True,diagnose=True) for row in rows]
        for row in [rows[0],rows[4]]:
            for arm in ['B','SAFE']:
                run_case(r,row,arm,'audit',audit=True)
        probe=rows[0]
        for arm in ['EPS_LOCK','ENERGY_EPS','LATENT_LOCK','LATE','ENERGY_LATE']:
            run_case(r,probe,arm,'audit',audit=True,compare=False)
        assert before==module_hashes({'b1':r.processor})
        write(OUT/'audit/environment.json',dict(python=platform.python_version(),torch=torch.__version__,
            diffusers=diffusers.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0)))
        write(OUT/'audit/complete.json',dict(passed=True,exact_D=16,exact_B=2,exact_SAFE=2,
            decomposition_valid=True,conditional_isolation=True,frozen_b1=True,commit=commit()))
    finally:release(r)
    decision(audit='pass')


def import_history(rows,arm,stage):
    for row in rows:
        source=history(arm,row['id']);meta=read(source.with_suffix('.json'))
        assert sha(source)==meta['png_sha256'] and sha(source.with_suffix('.pt'))==meta['diagnostics_sha256']
        target=image_path(arm,row['id'],stage);target.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ['.png','.json','.pt']:
            if target.with_suffix(suffix).exists():assert sha(target.with_suffix(suffix))==sha(source.with_suffix(suffix))
            else:shutil.copy2(source.with_suffix(suffix),target.with_suffix(suffix))
    write(OUT/stage/(arm+'_complete.json'),dict(arm=arm,n=len(rows),reused=True,commit=commit()))


def generate(arm,stage):
    prepare();assert read(OUT/'audit/complete.json')['passed']
    rows=read(OUT/'splits'/('probe.json' if stage=='probe' else 'dev32.json'))
    if arm in COMPONENT_ARMS:
        assert stage=='probe'
        write(OUT/'component_protocol.json',dict(arms=COMPONENT_ARMS,scope='exploratory full RGB diagnosis',
            fixed_weights=B1_SHA,formula='project(gate*p_ref*reference) or project(-gate*p_ref*self)',
            identities=[r['id'] for r in rows],freeze_before_component_images=True,
            motivation='same-state reference component produced larger boundary epsilon effect; verify final image'))
    if arm in ['B','D','SAFE']:
        import_history(rows,arm,stage);return
    if stage=='dev32':
        assert read(OUT/'probe/evaluation_complete.json')['complete']
    r=make_runtime(stage+'/'+arm+'/runtime')
    try:
        for row in rows:run_case(r,row,arm,stage,compare=False)
    finally:release(r)
    write(OUT/stage/(arm+'_complete.json'),dict(arm=arm,n=len(rows),commit=commit()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit','generate','batch','evaluate'])
    parser.add_argument('--arm',choices=ARMS+COMPONENT_ARMS);parser.add_argument('--arms',nargs='+',choices=ARMS+COMPONENT_ARMS)
    parser.add_argument('--stage',choices=['probe','dev32'],default='probe');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.action=='prepare':prepare()
    elif args.action=='audit':audit()
    elif args.action=='generate':generate(args.arm,args.stage)
    elif args.action=='batch':
        for arm in args.arms:generate(arm,args.stage)
    else:
        from tools.e48_eval import evaluate
        evaluate(args.stage)


if __name__=='__main__':main()
