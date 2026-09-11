"""阻断对照完成后检查每张图每一步的区域外差值确实归零。"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    files = sorted(root.glob('token/*/output_block_trace.json'))
    if len(files) != 100:
        raise ValueError(f'阻断轨迹缺失：{len(files)}/100')
    for path in files:
        rows = json.loads(path.read_text(encoding='utf-8'))
        if [r['step_index'] for r in rows] != list(range(50)):
            raise ValueError(f'去噪步不完整：{path}')
        if any(r['outside_elements'] <= 0 or r['outside_delta_after_max'] != 0.0 for r in rows):
            raise ValueError(f'区域外阻断未通过检查：{path}')
    (root/'output_block_check.json').write_text(json.dumps({
        'status': 'passed', 'samples': len(files), 'steps_per_sample': 50,
        'scope': '同 latent 条件噪声预测区域外与关闭旁路完全相等，不保证最终图像背景不变',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('输出端阻断检查通过：100 张 x 50 步')


if __name__ == '__main__':
    main()
