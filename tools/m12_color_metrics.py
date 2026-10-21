"""独立颜色评价：分布指标不把纹样位置、亮度和衣外污染混为一谈。"""
import cv2
import numpy as np
from scipy.stats import wasserstein_distance


def pixels(rgb, mask=None, maximum=4096):
    a = np.asarray(rgb, np.uint8).reshape(-1, 3)
    if mask is not None:
        a = a[np.asarray(mask, bool).ravel()]
    if len(a) > maximum:
        a = a[np.linspace(0, len(a)-1, maximum).astype(int)]
    return a


def rgbuv(a):
    # HistoGAN RGBuv 的逆二次核；只取真实有效像素，不填充背景再统计。
    # 为小规模独立复算固定最多4096像素；float64、64 bins、sigma=.02。
    a = np.asarray(a, np.float64)/255
    logs = np.log(a+1e-6)
    intensity = np.sqrt((a*a).sum(1)+1e-6)[:, None]
    centers = np.linspace(-3., 3., 64)[None]
    hist = []
    for c, x, y in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
        u = 1/(1+((logs[:, c:c+1]-logs[:, x:x+1]-centers)/.02)**2)
        v = 1/(1+((logs[:, c:c+1]-logs[:, y:y+1]-centers)/.02)**2)
        hist.append((u*intensity).T @ v)
    h = np.stack(hist)
    return h/h.sum()


def hellinger(a, b):
    return float(np.sqrt(((np.sqrt(a)-np.sqrt(b))**2).sum()/2))


def descriptor(rgb, mask=None):
    p = pixels(rgb, mask)
    if not len(p):
        return None
    lab = cv2.cvtColor(p.astype(np.float32)[None]/255, cv2.COLOR_RGB2LAB)[0].astype(np.float64)
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    colored = chroma > 5
    hue = None
    if chroma.mean() > 5 and colored.mean() >= .10:
        angle = (np.arctan2(lab[colored, 2], lab[colored, 1])+2*np.pi) % (2*np.pi)
        hue = np.histogram(angle, bins=np.linspace(0, 2*np.pi, 25), weights=chroma[colored])[0]
        hue = hue/hue.sum()
    return dict(rgbuv=rgbuv(p), lab=lab, chroma=float(chroma.mean()), hue=hue,
                pixels=len(p), colored_fraction=float(colored.mean()))


def compare(a, b, prefix):
    if a is None or b is None:
        return {prefix+k: None for k in ['rgbuv_h', 'ab_w1', 'l_w1', 'chroma_delta', 'chroma_log_ratio', 'hue_h']}
    return {prefix+'rgbuv_h': hellinger(a['rgbuv'], b['rgbuv']),
            prefix+'ab_w1': float(np.mean([wasserstein_distance(a['lab'][:, i], b['lab'][:, i]) for i in [1, 2]])),
            prefix+'l_w1': float(wasserstein_distance(a['lab'][:, 0], b['lab'][:, 0])),
            prefix+'chroma_delta': a['chroma']-b['chroma'],
            prefix+'chroma_log_ratio': float(abs(np.log(a['chroma']/b['chroma']))) if b['chroma'] > 5 and a['chroma'] > 0 else None,
            prefix+'hue_h': hellinger(a['hue'], b['hue']) if a['hue'] is not None and b['hue'] is not None else None}


def regions(mask):
    raw = np.asarray(mask, np.uint8)
    inside = cv2.distanceTransform(raw, cv2.DIST_L2, 5) > 9
    # 面积不足时明确NA，不能偷偷切换评价区域。
    if inside.sum() < 50:
        inside[:] = False
    outside = cv2.dilate(raw, np.ones((19, 19), np.uint8)) == 0
    return inside, outside


def outside_metrics(rgb, outside):
    p = pixels(rgb, outside, maximum=1000000)
    if not len(p):
        return dict(outside_color=None, outside_white_distance=None)
    lab = cv2.cvtColor(p.astype(np.float32)[None]/255, cv2.COLOR_RGB2LAB)[0]
    hsv = cv2.cvtColor(p[None], cv2.COLOR_RGB2HSV)[0].astype(float)
    colored = (np.linalg.norm(lab[:, 1:], axis=1) > 8) & (hsv[:, 1]/255 > .15) & (hsv[:, 2]/255 > .15)
    return dict(outside_color=float(colored.mean()), outside_white_distance=float(np.linalg.norm((255-p.astype(float))/255, axis=1).mean()))


def calibrate(rgb, reference, soft_mask, valid):
    """免训练RGB后处理基线：保留L，只校准衣内a/b矩；不是HistoGAN复现。"""
    original = np.asarray(rgb, np.uint8)
    if not valid:
        return original.copy()
    mask = np.asarray(soft_mask, np.float32)
    lab = cv2.cvtColor(original.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    ref = cv2.cvtColor(np.asarray(reference, np.float32)/255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    sample = lab[mask >= .99]
    assert len(sample) >= 50
    edited = lab.copy()
    for i in [1, 2]:
        gain = np.clip(ref[:, i].std()/max(sample[:, i].std(), 1e-6), .5, 2.)
        edited[:, :, i] = (lab[:, :, i]-sample[:, i].mean())*gain+ref[:, i].mean()
    corrected = cv2.cvtColor(edited, cv2.COLOR_LAB2RGB).clip(0, 1)*255
    mixed = np.rint(original*(1-mask[:, :, None])+corrected*mask[:, :, None]).clip(0, 255).astype(np.uint8)
    mixed[mask == 0] = original[mask == 0]
    return mixed


def bootstrap(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], np.float64)
    if not len(values):
        return dict(n=0, mean=None, ci95=[None, None])
    rng = np.random.default_rng(32042)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return dict(n=len(values), mean=float(values.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist())


def statistics(records, baseline='BASE'):
    arms = list(dict.fromkeys(r['arm'] for r in records))
    keys = [k for k, v in records[0].items() if isinstance(v, (float, int)) and k not in ['seed', 'mask_valid']]
    means = {arm: {k: bootstrap([r.get(k) for r in records if r['arm'] == arm]) for k in keys} for arm in arms}
    base = {r['id']: r for r in records if r['arm'] == baseline}
    paired = {arm: {k: bootstrap([r[k]-base[r['id']][k] for r in records if r['arm'] == arm and r.get(k) is not None and base[r['id']].get(k) is not None]) for k in keys} for arm in arms if arm != baseline}
    return dict(means=means, paired=paired, baseline=baseline, unit='identity, seed42 only; exploratory development intervals, no independent confirmation')
