"""E2a-ATTR-01：统一路径、冻结配置和结构化产物。"""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(os.environ.get('E2A_PROJECT', '/share/home/u2515283058/Mymodel'))
SOURCE = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'output_eval/e2a_attr_01_20261019'
TRAIN = PROJECT / 'data/train_bf_texture.json'
E1 = PROJECT / 'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt'
HISTORY = PROJECT / 'output_eval/e48_b1_structure_20261014'
BASE = 'e7b6d2566a69bb6146eca86c95b12b03344e82e7'
PLAN_SHA = '85ab40e880e94527a04c3578b5e1a1e3a709f3a970f2e78392e000c0bbd4db23'
FACTORS = ['000', '100', '010', '001', '110', '101', '011', '111']
PRIMARY = ['edge_f1', 'leak', 'iou', 'edge_l1']
APPEARANCE = ['lpips', 'tcf_lab']
DIRECTION = dict(edge_f1=1, leak=-1, iou=1, edge_l1=-1, lpips=-1, tcf_lab=-1)
CONFIG = dict(protocol='E2a-ATTR-01', source_plan_sha256=PLAN_SHA,
    historical_commit=BASE, evaluation_source_ref='codex/m54-b1-screen',
    body_rgb_target='reference_texture', boundary_target='GT', outside_target='GT',
    size=[384, 512], updates=28365, effective_batch=8, lr=5e-5, warmup=500,
    seed_A=42, seed_B=43, confirmation_seeds=[42,43,44], inference_seed=42,
    ddim_steps=50, cfg=7.0, sketch_scale=.6, texture_scale=1.0,
    bootstrap_seed=219042, bootstrap_samples=10000, holm_family='7 effects x 6 prespecified metrics',
    p1_n=24, p2_n=32, lock_initial_n=96,
    # 在查看输出前固定这些数值；正值为可接受退化的绝对界限。
    noninferiority=dict(edge_f1=.020, iou=.010, leak=.005, edge_l1=.002, lpips=.010, tcf_lab=1.0),
    meaningful_gain=dict(edge_f1=.020, leak=.005, iou=.010, edge_l1=.002, lpips=.010, tcf_lab=1.0),
    power=.80, family_alpha=.05, lock_max_n=512,
    scheduler='cosine/warmup stepped once per successful optimizer update',
    amp=dict(precision='fp16', init_scale=128.0, growth_interval=1000000000,
             skipped_update_policy='fail; preserve failure; do not silently consume unequal sequences'),
    mask_training='historical Pillow flood fill + MaxFilter9/5; no GT fallback',
    mask_evaluation='frozen OpenCV sketch_only; independent of generated output',
    independence='training-manifest identity/hash/near-duplicate exclusion; pretrained SD exposure unknown')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, ('FROZEN_CHANGED', str(path))
    else:
        write(path, value)
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def commit():
    return subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def train_folder(seed, factor, smoke=False):
    return OUT / ('smoke' if smoke else 'training') / ('seed%d'%seed) / ('F'+factor)


def checkpoint(seed, factor):
    return train_folder(seed, factor) / 'checkpoint-final/joint_model.pt'


def state(**changes):
    path = OUT/'status.json'
    value = read(path) if path.exists() else dict(preparation='NOT_RUN', A='NOT_RUN', B='NOT_RUN',
        confirmation='NOT_RUN', LOCK='NOT_RUN', complete=False)
    value.update(changes); write(path, value); return value
