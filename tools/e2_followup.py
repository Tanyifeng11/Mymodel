"""2026-11-08已批准的诊断/固定换料/A离线比较；无训练动作。"""
import argparse
from pathlib import Path
import zipfile

from tools.e2_followup_data import read, write, sha


def bundle(args):
    root = Path(args.out)
    files = [p for p in root.rglob('*') if p.is_file()
             and p.relative_to(root).parts[0] != 'source'
             and p.suffix in ['.json', '.jsonl', '.png', '.csv', '.out', '.err', '.txt']
             and p.name != 'bundle_manifest.json']
    write(root/'bundle_manifest.json', {str(p.relative_to(root)):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    with zipfile.ZipFile(root/'review_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/'bundle_manifest.json']:z.write(p,str(p.relative_to(root)))
    print('BUNDLE', (root/'review_results.zip').stat().st_size, sha(root/'review_results.zip'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','offline','freeze-pairs','replay','probe','generate','evaluate','bundle'])
    p.add_argument('--parent',default='/share/home/u2515283058/Mymodel/output_eval/e2_fft_20261106')
    p.add_argument('--out')
    p.add_argument('--project',default='/share/home/u2515283058/Mymodel')
    p.add_argument('--data',default='/share/home/u2515283058/datasets/BF/training')
    p.add_argument('--cloth-dir',help='仅供本地数据目录显式映射；服务器默认沿用cloth字段')
    p.add_argument('--manifest',default='/share/home/u2515283058/Mymodel/output_eval/bf_fusion_t3_three_arm_20261101/training_manifest.json')
    p.add_argument('--mask-root',default='/share/home/u2515283058/Mymodel/output_eval/e2a_attr_01_20261019/masks/train')
    p.add_argument('--review',default=str(Path(__file__).resolve().parents[1]/'configs/e2_followup_review.json'))
    p.add_argument('--arm',choices=['E2','E2_FFT'])
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    args.out=args.out or str(Path(args.parent)/'followup_20261108')
    Path(args.out).mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    if args.action=='bundle':bundle(args)
    elif args.action in ['prepare','offline','freeze-pairs']:
        from tools import e2_followup_data
        getattr(e2_followup_data,args.action.replace('-','_'))(args)
    elif args.action in ['replay','probe']:
        from tools import e2_followup_diagnostics
        getattr(e2_followup_diagnostics,args.action)(args)
    else:
        from tools import e2_followup_evaluation
        getattr(e2_followup_evaluation,args.action)(args)


if __name__=='__main__':main()
