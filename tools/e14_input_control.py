"""诊断用预处理后颜色控制；不改变正式 E5 推理默认路径。"""
import numpy as np
import torch


def rank_binary(rgb):
    """输入BCHW的[0,1] RGB；在每张图内固定25%深色，75%浅色。

    对亮度并列使用固定空间顺序；记录阈值并列数量，便于检查边界伪影。
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('预期 BCHW RGB 输入')
    original = rgb.detach().float().cpu().numpy()
    result, stats = [], []
    for source in original:
        gray = source.mean(axis=0)
        n = gray.size // 4
        if gray.size % 4:
            raise ValueError('像素数必须能被4整除')
        order = np.argsort(gray.ravel(), kind='stable')
        mask = np.zeros(gray.size, dtype=bool)
        mask[order[:n]] = True
        output = np.where(mask.reshape(gray.shape), 32/255, 224/255).astype(np.float32)
        output = np.repeat(output[None], 3, axis=0)
        cut = gray.ravel()[order[n-1]]
        stats.append(dict(dark_pixels=n, pixels=gray.size,
                          cutoff_tied_pixels=int(np.count_nonzero(gray == cut)),
                          mean_abs_change=float(np.abs(output-source).mean())))
        result.append(output)
    return torch.from_numpy(np.stack(result)).to(device=rgb.device, dtype=rgb.dtype), stats


def channel_signature(tensor):
    """实际送入编码器的张量边缘分布；不是可视化PNG的分布。"""
    array = tensor.detach().float().cpu().numpy()
    result = []
    for channel in array[0]:
        vals, counts = np.unique(channel, return_counts=True)
        result.append({'levels': vals.tolist(), 'counts': counts.tolist()})
    return result
