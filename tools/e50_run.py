"""e50 先验收再生成；既有基线不重新选择，生成可断点续跑。"""
import argparse
import gc
import torch
from tools.e50_protocol import *
from models.e50_structure_transport import StructureTransport


def runtime(namespace):
    from tools import e47_runtime as module
    old=module.OUT;module.OUT=OUT
    try:r=module.runtime('B',True,namespace)
    finally:module.OUT=old
    r.load(B1);r.processor.eval().requires_grad_(False)
    return r


def release(r):
    from tools.e47_runtime import close
    close(r);del r;gc.collect();torch.cuda.empty_cache()


def case(r,row,arm,stage='probe',disabled=False,audit=False):
    from tools.e47_runtime import exact_trace
    source_stage='probe' if row['id'] in {x['id'] for x in read(SOURCE/'splits/probe.json')} else 'dev32'
    source=SOURCE/source_stage/'images/B'/(row['id']+'.pt')
    trace=torch.load(source,map_location='cpu',weights_only=False)
    path=image_path(arm,row['id'],stage)
    done=path.with_name(path.stem+'_intervention.json')
    if done.exists():
        value=read(done)
        assert value['arm']==arm and value['disabled']==disabled and sha(path)==value['generation']['png_sha256']
        return value
    ctl=StructureTransport(r,row,arm,trace,disabled=disabled,audit=audit)
    ctl.install()
    try:record,_,_=r.generate(row,path)
    finally:ctl.close()
    assert len(ctl.steps)==ctl.unconditional==50
    assert all(s['independent_epsilon_exact'] for s in ctl.steps)
    check=None
    if disabled:
        historical=SOURCE/source_stage/'images'/('D' if ctl.kind=='STRUCT' else 'B')/(row['id']+'.png')
        check=exact_trace(path.with_suffix('.pt'),historical.with_suffix('.pt'))
        assert check['bit_exact'] and sha(path)==sha(historical)
    if ctl.maps:torch.save(ctl.maps,path.with_name(path.stem+'_matching.pt'))
    result=dict(arm=arm,disabled=disabled,steps=ctl.steps,generation=record,trace_check=check,
        independent_source=str(source),independent_source_sha256=sha(source),source_commit=commit())
    write(done,result)
    print('E50 RGB',stage,arm,row['id'],round(record['seconds'],2),flush=True)
    return result


def audit():
    from tools.e22_4_generation import module_hashes
    rows=prepare();r=runtime('audit/runtime')
    before=module_hashes({'processor':r.processor})
    try:
        invalid=[row for row in read(SOURCE/'splits/dev32.json')
            if not read(SOURCE/'dev32/images/B'/(row['id']+'.json'))['mask']['valid']]
        assert len(invalid)==4
        for i,row in enumerate(rows[:4]+invalid):
            case(r,row,'STRUCT' if i%2==0 else 'MATCH','audit_disabled',disabled=True,audit=True)
        for row in [rows[0],rows[-1]]:
            for arm in ['STRUCT','MATCH','STRUCT_ENERGY','MATCH_ENERGY']:
                case(r,row,arm,'audit_live',audit=True)
        assert before==module_hashes({'processor':r.processor})
        write(OUT/'audit/complete.json',dict(passed=True,disabled_full_exact=8,live_full=8,
            independent_E5_exact_steps_per_case=50,unconditional_isolation=True,
            frozen_processor=True,commit=commit()))
    finally:release(r)
    decision(audit='pass')


def generate(arm):
    rows=prepare();assert read(OUT/'audit/complete.json')['passed']
    r=runtime('probe/'+arm+'/runtime')
    try:
        for row in rows:case(r,row,arm)
    finally:release(r)
    write(OUT/'probe'/(arm+'_complete.json'),dict(n=16,arm=arm,commit=commit()))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','generate','evaluate'])
    p.add_argument('--arm',choices=ARMS[2:]);args=p.parse_args();torch.set_num_threads(2)
    if args.action=='prepare':prepare()
    elif args.action=='audit':audit()
    elif args.action=='generate':generate(args.arm)
    else:
        from tools.e50_eval import evaluate
        evaluate()


if __name__=='__main__':main()
