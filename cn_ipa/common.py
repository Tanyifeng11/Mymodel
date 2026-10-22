import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get('CNIPA_ROOT', '/share/home/u2515283058/cn_ipa_repro_20261022'))
HISTORY = Path('/share/home/u2515283058/Mymodel')
CONFIRM = HISTORY / 'output_eval/m12_gain_independent_confirm_20261020'
M54 = HISTORY / 'output_eval/m54_b1_screen_20261017'
SD = HISTORY / 'models/stable-diffusion-v1-5'
CN = ROOT / 'pretrained/controlnet'
IPA = ROOT / 'pretrained/ip_adapter'
ENCODER = ROOT / 'pretrained/image_encoder'
SIZE = (384, 512)
KEYS = ['lpips_gt', 'contour_f1', 'iou', 'leak']
CONFIG = dict(width=384, height=512, steps=50, cfg=7., seeds=[42, 43, 44],
    cn_updates=5000, cn_checkpoints=[1000,2000,3000,4000,5000], cn_lr=1e-5,
    ipa_updates=3000, ipa_checkpoints=[500,1000,1500,2000,2500,3000], ipa_lr=5e-5,
    effective_batch=4, seed=42, mixed_precision='fp16',
    cn_scales=[.75,1.,1.25], ip_scales=[.4,.65,.9],
    # 预先固定：结构/外溢相对同一 CN-only 的开发集均值边界，然后最小 LPIPS。
    scale_selection=dict(f1_delta_min=-.02, iou_delta_min=-.01, leak_delta_max=.005,
        fallback='若没有合格组合，先最小化三项归一化越界总量，再按 LPIPS 排序；完整披露'),
    checkpoint_selection='CN:最大平均 F1，依次 IoU、最小 Leak、最小 LPIPS、较早更新；IPA:固定 CN/IP=1/.65，以同一边界和 LPIPS 选择',
    train_geometry='沿用 M12 已冻结的同步直接 BILINEAR Resize384x512；原图256x256，不做新中心裁剪',
    reference='原数据集单独 texture 文件，CLIP ViT-H/14 processor；绝不使用 GT 回退',
    test_status='TEST-A 为研究者已经查看的既有96身份确认集；不声称新盲测',
    comparisons='C0/C1/C2 对 M12/M12_GAIN，C1-C0，C2-C1；身份级探索性95%CI，非多重校正确认性优势',
    bootstrap=dict(draws=10000, seed=32042, unit='identity', missing='任一seed缺失则该身份该指标NA'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def state_hash(module, exclude=()):
    import torch
    h=hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if any(part in name for part in exclude): continue
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def require_audit():
    audit=read(ROOT/'manifests/split_audit.json')
    assert audit['passed'], audit
    for name in ['train','dev','test_A']:
        assert sha(ROOT/'manifests'/f'{name}.json')==audit['manifest_sha256'][name]
    return audit
