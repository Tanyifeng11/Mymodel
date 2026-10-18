"""复用原始 color-safe 命令，只固定 seed42 并隔离新实验产物。"""
import argparse
import csv
import hashlib
import json
import os
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path('/share/home/u2515283058/Mymodel')
ROOT = PROJECT / 'output_eval/m54_b1_screen_20261017/e2b_color_safe_retrain_seed42_20261018'
SOURCE = Path(__file__).resolve().parents[1]
OLD_RUN = PROJECT / 'wandb/run-20260629_110629-hrb5mea9/files'
BASE = '598818b791acc0ef8d5ade2f85a346b293a32c33'
HISTORICAL_FILES = ['train_GAM_texture_joint.py', 'adapter/attention_processor.py',
                    'models/bf_texture_module.py', 'texture_preprocess.py']
EXPECTED = dict(num_train_epochs=3, learning_rate=1e-5, num_warmup_steps=500,
                train_batch_size=1, gradient_accumulation_steps=2,
                width=384, height=512, mixed_precision='fp16',
                texture_condition_mode='token', texture_mode='patch_resampled',
                texture_preprocess_mode='plain_resize', bf_num_tokens=16,
                layer_group_enabled=1, use_texture_gate=1, gate_type='layer',
                gate_init='identity', gate_reg_weight=5e-4, gate_min=0.7,
                gate_max=1.3, freeze_except_gate=0, lambda_style=1.0,
                lambda_edge=0.05, lambda_texture_color=0.2,
                lambda_region_texture=0.1, lambda_region_color_lab=0.05,
                lambda_boundary=0.05, lambda_leak=0.1, region_kernel_size=9,
                joint_t_drop_rate=0.2, joint_i_drop_rate=0.05,
                joint_ti_drop_rate=0.05, max_grad_norm=1.0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare():
    import yaml
    assert not (ROOT / 'protocol.json').exists(), '本次协议已存在，请读取既有状态，不能重复准备/提交'
    assert not (ROOT / 'training').exists(), '独立训练目录必须尚未存在'
    git('diff', '--exit-code', BASE, '--', *HISTORICAL_FILES)
    metadata = json.loads((OLD_RUN / 'wandb-metadata.json').read_text())
    config = {k: v.get('value') for k, v in yaml.safe_load((OLD_RUN / 'config.yaml').read_text()).items()
              if isinstance(v, dict)}
    for key, expected in EXPECTED.items():
        actual = config[key]
        assert (float(actual) == expected if isinstance(expected, (int, float)) else actual == expected), (key, actual)
    assert not config.get('resume_from_checkpoint')
    assert config.get('seed') is None
    e2a = PROJECT / 'output/phase1_e2a_region_e5/checkpoint-final/joint_model.pt'
    assert Path(config['gam_init_ckpt']).resolve() == e2a.resolve()
    dataset_path = Path(config['dataset_json_path'])
    dataset = json.loads(dataset_path.read_text())
    assert len(dataset) == 45384
    data_root = Path(config['data_root_path'])
    images = {data_root / row[key] for row in dataset for key in ('cloth', 'sketch', 'texture')}
    missing = [str(p) for p in images if not p.is_file()]
    assert not missing, missing[:8]
    source_paths = [e2a, Path(config['texture_adapter_ckpt']), dataset_path]
    source_hashes = {}
    for p in source_paths:
        print('核验源文件:', p, flush=True)
        source_hashes[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
    argv = list(metadata['args'])
    # 不改原损失、mask、模型结构和数据；只隔离输出、关闭外部日志及显式重置本次计数。
    argv += ['--output_dir', str(ROOT / 'training'), '--report_to', 'none',
             '--wandb_mode', 'disabled', '--wandb_run_name', 'e2b_color_safe_retrain_seed42_20261018',
             '--resume_from_checkpoint', '', '--start_global_step', '0']
    ROOT.mkdir(parents=True, exist_ok=True)
    write('protocol.json', dict(
        authorization='直接从 E2a 按原配置重训 3 epoch，固定 seed42，保存为独立新实验',
        created_utc=datetime.now(timezone.utc).isoformat(), seed=42, original_seed=None,
        historical_commit=BASE, execution_commit=git('rev-parse', 'HEAD'),
        original_run=str(OLD_RUN), original_argv=metadata['args'], effective_argv=argv,
        expected=EXPECTED, target_effective_updates=17019, dataset_records=45384,
        unique_input_files_presence_checked=len(images), input_images_full_sha_not_recomputed=True,
        source_hashes=source_hashes,
        historical_file_sha256={p: sha(SOURCE / p) for p in HISTORICAL_FILES},
        historical_metadata_sha256=sha(OLD_RUN / 'wandb-metadata.json'),
        historical_config_sha256=sha(OLD_RUN / 'config.yaml'),
        note='这是固定新 seed 的独立重训，不宣称恢复已删除文件的原字节或原评价数值。'))
    print('准备通过，目标 17019 次有效更新；尚未提交 GPU 作业。', flush=True)


def train():
    from accelerate.utils import set_seed
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert int(os.environ.get('WORLD_SIZE', '1')) == 4
    assert git('rev-parse', 'HEAD') == protocol['execution_commit']
    assert not list((ROOT / 'training').glob('checkpoint-*/joint_model.pt')), '禁止重复开始已有训练'
    for p, digest in protocol['historical_file_sha256'].items():
        assert sha(SOURCE / p) == digest
    set_seed(42)
    sys.path.insert(0, str(SOURCE))
    sys.argv = [str(SOURCE / 'train_GAM_texture_joint.py')] + protocol['effective_argv']
    runpy.run_path(sys.argv[0], run_name='__main__')


def verify():
    import torch
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    checkpoint = ROOT / 'training/checkpoint-final/joint_model.pt'
    assert checkpoint.resolve().parent.name == 'checkpoint-17019'
    data = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    meta = data['meta']
    for key in ('texture_condition_mode', 'layer_group_enabled', 'use_texture_gate', 'gate_type',
                'gate_reg_weight', 'gate_min', 'gate_max', 'lambda_style', 'lambda_texture_color',
                'lambda_region_texture', 'lambda_region_color_lab', 'lambda_boundary', 'lambda_leak'):
        assert meta[key] == EXPECTED[key], (key, meta[key])
    assert meta['texture_num_tokens'] == 16
    checked = 0
    for group, state in data.items():
        if not isinstance(state, dict) or group == 'meta':
            continue
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                assert torch.isfinite(tensor).all().item(), (group, name)
                checked += 1
    for p, record in protocol['source_hashes'].items():
        assert sha(p) == record['sha256'], ('源文件发生变化', p)
    with (ROOT / 'training/gate_stats.csv').open() as f:
        gates = list(csv.DictReader(f))
    assert max(int(r['global_step']) for r in gates) == 17019
    assert all(0.7 <= float(r['gate_value']) <= 1.3 for r in gates)
    write('training_verification.json', dict(
        passed=True, training_process_returned_successfully=True, slurm_final_state='AWAITING_SACCT',
        seed=42, effective_updates=17019, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        finite_state_tensors=checked, frozen_initialization_sources_unchanged=True,
        scientific_evaluation='NOT_RUN', original_file_byte_recovery=False))
    print('新检查点完整性通过，仍须核对 Slurm 最终状态。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'verify'])
    action = parser.parse_args().action
    {'prepare': prepare, 'train': train, 'verify': verify}[action]()
