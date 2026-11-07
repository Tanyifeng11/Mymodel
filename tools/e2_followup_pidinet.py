"""官方 PiDiNet 的冻结离线包装；不接入生成模型。"""
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from tools.e2_followup_data import read, sha


def detector():
    root = Path(__file__).resolve().parents[1]/'vendor/pidinet_followup'
    provenance = read(root/'provenance.json')
    for name, item in provenance['files'].items():
        assert sha(root/name) == item['sha256'], '官方依赖指纹改变：'+name
    from vendor.pidinet_followup.pidinet import pidinet
    model = pidinet(Namespace(config='carv4', dil=True, sa=True))
    # 只读取经过来源与SHA验证的作者检查点，拒绝缺键/多键。
    state = torch.load(root/'table5_pidinet.pth', map_location='cpu', weights_only=True)
    state = state['state_dict'] if 'state_dict' in state else state
    model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state.items()}, strict=True)
    model.eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None,:,None,None]
    std = torch.tensor([.229, .224, .225])[None,:,None,None]

    def predict(pixels):
        value = torch.from_numpy(np.array(pixels, copy=True)).permute(2,0,1)[None].float()/255.
        with torch.no_grad():
            return model((value-mean)/std)[-1][0,0].numpy()

    return predict, dict(provenance, status='executed_frozen_offline', device='cpu',
                         input='RGB [0,1], official ImageNet mean/std', output='last sigmoid fusion; no BSDS NMS')
