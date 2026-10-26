"""仅生成新 T3-M；复用冻结推理、六指标及已批准的开发集换参考。"""
import argparse
import torch
from tools.e2a_matched_common import *
from tools import e2a_sup_eval as backend


def evaluate(stage,phase='formal'):
    prep=read(ROOT/'audit/preparation_complete.json');assert prep['passed']
    assert all(sha(SOURCE/p)==h for p,h in read(SUP/'evaluation/metric_protocol.json')['evaluation_code'].items())
    backend.ROOT=ROOT;backend.ARMS=['T3M'];backend.CONFIG=CONFIG
    backend.model_path=model_path;backend.folder=folder
    backend.evaluate('T3M',stage,phase)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['T3M'],default='T3M')
    p.add_argument('--stage',choices=['P2a','P2b','reference_swap'],required=True)
    p.add_argument('--phase',choices=['pilot','formal'],default='formal')
    a=p.parse_args();torch.set_num_threads(2);evaluate(a.stage,a.phase)
