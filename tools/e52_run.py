"""E52 分阶段入口；工程验收未通过时不运行因果确认。"""
import argparse
import torch
from tools.e52_protocol import *
from tools.e52_runtime import runtime,release,case

def audit():
    from tools.e47_runtime import exact_trace
    from tools.e52_eval import replay_metrics
    prepare();path=OUT/'audit/p0_complete.json'
    if path.exists():assert read(path)['passed'];return
    rows=read(OUT/'splits/dev32.json');r=runtime('audit/runtime');checks=[]
    try:
        for row in rows:
            base=image_path('p0','E5',row['id'])
            for arm,old in [('E5','B'),('B1','D')]:
                case(r,row,arm,'p0',teacher_path=base.with_suffix('.pt') if arm=='B1' else None)
                current=image_path('p0',arm,row['id']);source=SOURCE/'dev32/images'/old/(row['id']+'.png')
                result=exact_trace(current.with_suffix('.pt'),source.with_suffix('.pt'))
                assert result['bit_exact'] and sha(current)==sha(source),('BASELINE_REPLAY_FAIL',row['id'],arm,result)
                checks.append(dict(id=row['id'],arm=arm,trace=result,png_exact=True))
        row=next(x for x in rows if read(image_path('p0','E5',x['id']).with_suffix('.json'))['mask']['valid'])
        other=next(x for x in rows if x['id']!=row['id'])
        case(r,row,'E5_ALT','p0',writer=other['reference'])
        result=exact_trace(image_path('p0','E5_ALT',row['id']).with_suffix('.pt'),image_path('p0','E5',row['id']).with_suffix('.pt'))
        assert result['bit_exact'] and sha(image_path('p0','E5_ALT',row['id']))==sha(image_path('p0','E5',row['id'])),'BASE_REFERENCE_LEAK'
        # 普通 B1 在诊断包装器下精确回放，同时覆盖 OFF、无效 mask 和两条 CFG 支路。
        shapes=read(image_path('p0','B1',row['id']).with_name(row['id']+'_trace.json'))['steps'][0]
        assert shapes['reference_shape']==[1,256,640] and shapes['target_shape']==[1,768,640],shapes
    finally:release(r)
    metrics=replay_metrics(rows)
    write(path,dict(passed=True,full_replays=checks,writer_off_invariance=result,shapes=shapes,
        gpu_metric_replay=metrics,invalid_ids=[x['id'] for x in rows if not read(image_path('p0','E5',x['id']).with_suffix('.json'))['mask']['valid']],
        unconditional_and_latent_exact=True,commit=commit()))
    decision(P0='pass');print('E52 P0 PASS',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','provenance','generate','evaluate','freeze','finish'])
    p.add_argument('--stage',default='pilot');p.add_argument('--arm');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(2)
    if args.action=='prepare':prepare()
    elif args.action=='audit':audit()
    elif args.action=='provenance':
        from tools.e52_provenance import run
        run()
    elif args.action=='generate':
        from tools.e52_generate import run
        run(args.stage,args.arm,args.shard,args.shards)
    elif args.action=='evaluate':
        from tools.e52_eval import evaluate
        evaluate(args.stage)
    elif args.action=='freeze':
        from tools.e52_data import freeze_confirmation
        freeze_confirmation()
    else:
        from tools.e52_finish import finish
        finish()

if __name__=='__main__':main()
