"""已批准的B0/B1/B2/B4/B5筛选入口；先输入审阅，再测试/冒烟/训练。"""
import argparse
from pathlib import Path
import zipfile

from tools.e2_fft_experiment import SOURCE, read, write, sha


def bundle(args):
    root=Path(args.out)
    prefixes=['inputs','audit','calibration','smoke','comparison','training','generation','outputs','metrics','logs','tests']
    files=[p for prefix in prefixes for p in (root/prefix).rglob('*') if p.is_file()
           and p.suffix in ['.json','.jsonl','.png','.txt','.out','.err','.csv']]
    files += [root/n for n in ['strict_pairs.json','split.json'] if (root/n).exists()]
    files=sorted(set(files))
    write(root/'bundle_manifest.json',{p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with zipfile.ZipFile(root/'review_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[root/'bundle_manifest.json']:archive.write(path,path.relative_to(root).as_posix())
    write(root/'bundle_info.json',dict(bytes=(root/'review_results.zip').stat().st_size,
        sha256=sha(root/'review_results.zip'),files=len(files)))
    print('B_BUNDLE',sha(root/'review_results.zip'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','extend-review','freeze-pairs','audit-training','audit-training-final',
        'calibrate','smoke','train','generate','evaluate','bundle'])
    p.add_argument('--project',default='/share/home/u2515283058/Mymodel')
    p.add_argument('--parent',default='/share/home/u2515283058/Mymodel/output_eval/e2_fft_20261106')
    p.add_argument('--out',default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--data',default='/share/home/u2515283058/datasets/BF')
    p.add_argument('--candidates',default=str(SOURCE/'configs/e2_b_candidates.json'))
    p.add_argument('--review',default=str(SOURCE/'configs/e2_b_input_review.json'))
    p.add_argument('--review-all',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',choices=['B0','B1','B2','B4','B5'])
    args=p.parse_args()
    Path(args.out).mkdir(parents=True,exist_ok=True)
    if args.action=='bundle':bundle(args)
    elif args.action in ['prepare','extend-review','freeze-pairs']:
        from tools import e2_b_screen_data
        getattr(e2_b_screen_data,args.action.replace('-','_'))(args)
    elif args.action in ['generate','evaluate']:
        from tools import e2_b_screen_evaluation
        getattr(e2_b_screen_evaluation,args.action)(args)
    else:
        from tools import e2_b_screen_training
        if args.action.startswith('audit-training'):
            e2_b_screen_training.audit_training(args,final=args.action.endswith('-final'))
        else:getattr(e2_b_screen_training,args.action)(args)


if __name__=='__main__':main()
