"""只读检查同一作业内的 GT 归一化与零更新感知保护项。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools import e2_b_init_epoch20_ab as experiment
from tools.e2_fft_experiment import setup as setup_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    args = experiment.setup(parser.parse_args().prior)
    setup_core(args)
    base = experiment.ft.light.LightingObjective(args)
    rows = []
    with torch.no_grad():
        for pair in args.trainval['pairs']:
            image = np.asarray(experiment.tail.rgb(args.run/'fixed'/pair['case_id']/'baseline125.png'))
            rgb = experiment.guard.anchor_tensor(image,torch.empty((),device=args.device))
            target = np.asarray(experiment.tail.rgb(args.data/pair['gt']).resize(experiment.tail.SIZE,Image.Resampling.BILINEAR))
            cpu = experiment.roi.tensors([target]).to(args.device)
            gpu = experiment.guard.anchor_tensor(target,rgb)
            mask = experiment.tail.case_mask(args,pair)
            _,old = base(rgb,gpu,mask,'inside')
            _,current = base(rgb,cpu,mask,'inside')
            delta = {k:float(current[k]-old[k]) for k in ('lpips','inside_lpips')}
            rows.append(dict(case=pair['case_id'],split=pair['split'],max_target_error=float((cpu-gpu).abs().max()),
                delta=delta,zero_guard=sum(max(v,0.) for v in delta.values())))
    report = dict(protocol_sha256=args.spec_sha,optimizer_updates=0,rows=rows,
        positive=sum(r['zero_guard']>0 for r in rows),max_zero_guard=max(r['zero_guard'] for r in rows),
        code_sha256=experiment.tail.sha(Path(__file__)))
    experiment.tail.write(args.run/'normalization_probe.json',report)
    print('INIT20_NORM_PROBE',len(rows),report['positive'],report['max_zero_guard'],flush=True)


if __name__=='__main__':
    main()
