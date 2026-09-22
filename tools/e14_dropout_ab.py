"""本地核验A/B数据顺序、dropout行为并分类三组已有特征。"""
import argparse
import json
from pathlib import Path
from tools.e14_pattern_probe import write_json


def verify(root):
    root = Path(root)
    logs = [[json.loads(line) for line in (root/mode/'training_metrics.jsonl').read_text().splitlines()]
            for mode in ['legacy_pooled','zero_final_tokens']]
    a,b = logs
    if not a or len(a) != len(b):
        raise ValueError('两组训练步数不一致或日志为空')
    count = 0
    for x,y in zip(a,b):
        for key in ['step','sample_indices','drop_flags']:
            if x[key] != y[key]: raise ValueError('两组训练配对不一致：'+key)
        for flag,old,new in zip(x['drop_flags'],x['token_abs_mean'],y['token_abs_mean']):
            if flag:
                count += 1
                if new != 0 or old == 0: raise ValueError('dropout行为不符合对照预期')
    if not count: raise ValueError('没有抽到image dropout样本，不能评价修复')
    result = dict(steps=len(a),dropped_samples=count,paired_order_and_flags=True,
                  note='单seed短训；日志核验不保证所有GPU算子逐位确定；不能代替生成质量评估。')
    write_json(root/'paired_training_audit.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    print(verify(args.root))
    if not args.verify_only:
        from tools.e14_real_orientation import evaluate
        for stage in ['baseline','legacy_pooled','zero_final_tokens']:
            evaluate(Path(args.root)/stage/'representations',no_pca=True)
