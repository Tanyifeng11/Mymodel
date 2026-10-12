"""Slurm 入口，分阶段运行，任何工程错误均留下实际状态。"""
import argparse
import gc
import torch
from tools.e44_protocol import OUT,write,decision


def run():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','pilot','evaluate','main'])
    parser.add_argument('--variant',choices=['B1','B2','B3'],default='B2');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.action=='smoke':
        from tools.e44_smoke import run as smoke
        smoke()
    elif args.action in ['pilot','main']:
        from tools.e44_train import train
        from tools.e44_eval import generate
        train(args.variant,args.action);gc.collect();torch.cuda.empty_cache()
        generate('p1' if args.action=='pilot' else 'p3',args.variant)
    else:
        from tools.e44_eval import generate,evaluate,counterfactual
        from tools.e44_protocol import read,sha,ARMS
        sequences=[read(OUT/'p1'/a/'complete.json')['sequence_sha256'] for a in ARMS]
        assert len(set(sequences))==1,'三组训练序列不一致'
        generate('p1','B0');gc.collect();torch.cuda.empty_cache()
        evaluate('p1');gc.collect();torch.cuda.empty_cache()
        for arm in ['B0']+ARMS:
            counterfactual(arm);gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    try:run()
    except Exception as exc:
        decision(engineering_error=repr(exc))
        raise
