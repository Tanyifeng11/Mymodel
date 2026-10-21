"""复用旧冻结推理/六指标后端，新的 P2 独立写出；GT 不进入推理。"""
import argparse
import torch
from tools.e2a_sup_common import *
from tools import e2a_attr_eval as evaluator


def evaluate(arm,stage,phase='formal'):
    assert stage in ['P2a','P2b'] and arm in ['B','C','E1']+ARMS
    if phase=='pilot':
        assert arm in ARMS and stage=='P2a'
        path=folder(arm,'pilot')/'checkpoint-final/joint_model.pt'
        rows=read(ROOT/'splits/P2a.json')[:8]
        output_stage='pilot_P2a'
    else:
        path=model_path(arm);rows=read(ROOT/'splits'/(stage+'.json'));output_stage=stage
    if arm in EXPECTED:assert sha(path)==EXPECTED[arm]
    else:assert sha(path)==read(path.parent.parent/'checkpoint_verification.json')['sha256']
    evaluator.OUT=ROOT
    evaluator.model_path=lambda name:path
    evaluator.input_rows=lambda name:rows
    freeze(ROOT/'checkpoint_manifest.json',read(OLD/'checkpoint_manifest.json'))
    evaluator.generate(output_stage,arm)
    evaluator.measure(output_stage,arm)
    write(ROOT/'evaluation'/output_stage/arm/'supervision_evaluation.json',dict(checkpoint_sha256=sha(path),
        rows_sha256=sha(ROOT/'splits'/(stage+'.json')),phase=phase,
        loss_disabled_at_inference=True,metric_backend_unchanged=True,GT_only_in_metrics=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True)
    p.add_argument('--stage',choices=['P2a','P2b'],required=True)
    p.add_argument('--phase',choices=['pilot','formal'],default='formal')
    a=p.parse_args();torch.set_num_threads(2);evaluate(a.arm,a.stage,a.phase)
