"""服务器审计历史BF权重并提取特征；本地运行统一的无PCA方向探针。"""
import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.e14_pattern_probe import read_labels, write_json


CHECKPOINTS = {
    'texture': 'texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin',
    'e0': 'phase1_e0_baseline_e5/checkpoint-28365/joint_model.pt',
    'e1': 'phase1_e1_grouped_e5/checkpoint-final/joint_model.pt',
    'e2a': 'phase1_e2a_region_e5/checkpoint-final/joint_model.pt',
    'e2b_color_safe': 'phase1_e2b_color_safe_gate_e3/checkpoint-final/joint_model.pt',
    'e5': 'phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt',
}


def module_name(key):
    parts = key.split('.')
    return '.'.join(parts[:2]) if parts[0] == 'token_source_proj' else parts[0]


def fingerprint(state):
    import torch
    groups = {}
    for key, tensor in sorted(state.items()):
        group = groups.setdefault(module_name(key), hashlib.sha256())
        group.update((key + str(tuple(tensor.shape)) + str(tensor.dtype)).encode())
        group.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    hashes = {k: h.hexdigest() for k, h in groups.items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def compare(left, right):
    result = {}
    for group in sorted({module_name(k) for k in left} | {module_name(k) for k in right}):
        a = {k for k in left if module_name(k) == group}
        b = {k for k in right if module_name(k) == group}
        compatible = a == b and all(left[k].shape == right[k].shape for k in a)
        entry = dict(compatible=compatible, missing_in_right=sorted(a-b), added_in_right=sorted(b-a))
        if compatible:
            squared, base, maximum, changed, count = 0., 0., 0., 0, 0
            for key in a:
                x, y = left[key].double(), right[key].double()
                delta = y - x
                squared += delta.square().sum().item()
                base += x.square().sum().item()
                maximum = max(maximum, delta.abs().max().item())
                changed += int((delta != 0).sum().item())
                count += delta.numel()
            entry.update(equal_values=changed == 0, changed_elements=changed, elements=count,
                         max_abs_difference=maximum, l2_difference=squared ** .5,
                         relative_l2=(squared/base) ** .5 if base else None)
        result[group] = entry
    return result


def compact(checkpoint):
    state = checkpoint['bf_texture_conditioner']
    meta = checkpoint.get('meta') or checkpoint.get('metadata') or {}
    if meta.get('texture_mode', 'patch_resampled') != 'patch_resampled':
        raise ValueError('历史比较要求patch_resampled，不自动转换模型')
    if list(meta.get('stage_token_hw', [8, 8])) != [8, 8]:
        raise ValueError('历史pool尺寸不是8x8，需单独适配后再比较')
    if any(k.startswith(('film.', 'nexus.', 'text_guidance.')) for k in state):
        raise ValueError('历史BF不应包含后续实验模块')
    return dict(bf_texture_conditioner=state, meta=meta)


def run(args):
    from checkpoint_utils import load_checkpoint_file
    import torch
    paths = {stage: Path(args.checkpoint_root) / relative for stage, relative in CHECKPOINTS.items()}
    # 先检查所有文件，禁止静默跳过或自动选择其他epoch。
    for stage, path in paths.items():
        print(stage, path, flush=True)
        if not path.is_file():
            raise FileNotFoundError(str(path))
    inputs, root = Path(args.inputs), Path(args.output)
    rows = read_labels(inputs / 'feature_labels.csv')
    from tools.e14_real_orientation import make_splits
    make_splits(rows)
    if len(rows) != 20:
        raise ValueError('本轮要求原来的20张参考图')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(inputs, root / 'inputs')
    audit = dict(complete=False, stages={}, comparisons=[],
                 note='阶段排列用于比较，不代表线性继承链；BF相同不能证明训练来源。',
                 scope='同一输入、预处理和BF层；不加载U-Net，不比较TCPM。')
    previous, previous_stage, seen, jobs = None, None, {}, []
    # 临时小权重仅供抽取，退出后删除；输出目录不保存原始大checkpoint。
    with tempfile.TemporaryDirectory(prefix='bf_weights_', dir=str(root)) as temp:
        for stage, path in paths.items():
            checkpoint = load_checkpoint_file(str(path))
            small = compact(checkpoint)
            del checkpoint
            gc.collect()
            state, meta = small['bf_texture_conditioner'], small['meta']
            digest, modules = fingerprint(state)
            config = dict(texture_mode=meta.get('texture_mode', 'patch_resampled'),
                          texture_num_tokens=int(state['resampler_queries'].shape[1]),
                          stage_token_hw=list(meta.get('stage_token_hw', [8, 8])))
            identity = (digest, json.dumps(config, sort_keys=True))
            canonical = seen.setdefault(identity, stage)
            audit['stages'][stage] = dict(checkpoint=str(path), bytes=path.stat().st_size,
                bf_sha256=digest, module_sha256=modules, meta=meta,
                feature_stage=canonical, bf_config=config)
            if previous is not None:
                audit['comparisons'].append(dict(left=previous_stage, right=stage,
                                                 modules=compare(previous, state)))
            previous, previous_stage = state, stage
            if canonical == stage:
                slim = Path(temp) / (stage + '.pt')
                torch.save(small, slim)
                jobs.append((stage, slim))
            # 每完成一个阶段就落盘，便于审计中途错误。
            write_json(root / 'weight_audit.json', audit)
            print(stage, 'BF=', digest[:12], 'feature_stage=', canonical, flush=True)
        del previous, state, small
        gc.collect()
        audit['complete'] = True
        write_json(root / 'weight_audit.json', audit)
        if args.audit_only:
            return
        for stage, slim in jobs:
            command = [sys.executable, '-m', 'tools.e14_pattern_probe', 'extract',
                '--checkpoint', str(slim), '--bf-only', '--allow-unmatched-extraction',
                '--labels', str(root / 'inputs/feature_labels.csv'), '--data-root', str(root / 'inputs'),
                '--base-model', args.base_model, '--clip-model', args.clip_model,
                '--output', str(root / stage / 'representations')]
            subprocess.run(command, check=True)
        write_json(root / 'extraction_complete.json', dict(complete=True,
            feature_stages=[stage for stage, _ in jobs], samples=len(rows)))


def evaluate(root):
    from tools.e14_real_orientation import evaluate as classify
    root = Path(root)
    audit = json.loads((root / 'weight_audit.json').read_text(encoding='utf-8'))
    if not (root / 'extraction_complete.json').is_file():
        raise ValueError('历史特征尚未全部提取完成')
    reports, result = {}, {}
    for stage, entry in audit['stages'].items():
        canonical = entry['feature_stage']
        features = root / canonical / 'representations'
        if canonical not in reports:
            classify(features, no_pca=True)
            reports[canonical] = json.loads((features / 'real_orientation_no_pca_report.json').read_text(encoding='utf-8'))
        result[stage] = dict(feature_stage=canonical, results={key: dict(
            balanced_accuracy=value['balanced_accuracy'], pairs_both_correct=value['pairs_both_correct'],
            fixed_C1_accuracy=value['fixed_C1_control']['balanced_accuracy'])
            for key, value in reports[canonical]['results'].items()})
    write_json(root / 'history_orientation_report.json', dict(stages=result,
        note='相同BF复用结果；阶段差异不是训练因果证明，结合元数据和训练记录判断。'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('run')
    p.add_argument('--checkpoint-root', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--clip-model', required=True)
    p.add_argument('--audit-only', action='store_true')
    p = sub.add_parser('evaluate')
    p.add_argument('--root', required=True)
    args = parser.parse_args()
    if args.action == 'run':
        run(args)
    else:
        evaluate(args.root)
