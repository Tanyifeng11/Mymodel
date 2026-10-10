"""简单固定表示与扰动；从 RGB 提取与正确 mask 下的表示实验分开。"""
import re
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from tools.e38_protocol import SIZE

DIAGONAL = float(np.hypot(*SIZE))


def foreground(rgb):
    im, diag = estimate_cloth_foreground_mask(Image.fromarray(rgb), *SIZE)
    mask = np.asarray(im) > 127
    valid = not diag['mask_low_confidence'] and .02 <= mask.mean() <= .95
    return mask, bool(valid), diag


def contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    p = max(cs, key=cv2.contourArea)[:, 0].astype(np.float64)
    if len(p) < 32: return None
    # 统一方向，只消除轮廓起点的任意性；保持绝对位置和长度信息。
    if cv2.contourArea(p.astype(np.float32), oriented=True) < 0: p = p[::-1]
    p = np.vstack([p, p[0]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, s[-1], 256, endpoint=False)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def garment_type(caption):
    text = caption.lower()
    if re.search(r'\bpants\b|\bjeans\b|\btrousers\b|\bshorts\b|\bskirt\b', text): return 'lower'
    if re.search(r'\bsleeveless\b|\btank\b|\bstrapless\b|\bhalter\b', text): return 'sleeveless'
    if re.search(r'\bsleeves?\b|\bjacket\b|\bshirt\b|\bsweater\b|\bblouse\b|\bhoodie\b|\bcoat\b', text): return 'sleeved_upper'
    return 'unknown'


def graph(mask, caption):
    """候选语义地标，必须另经图板审核；不把几何极值当已验证的部件标签。"""
    kind = garment_type(caption)
    result = dict(kind=kind, nodes={}, edges=[], valid=False, reason=None)
    if kind not in ['sleeved_upper', 'sleeveless']:
        result['reason'] = 'unsupported_or_ambiguous_garment_semantics'; return result
    p = contour(mask)
    if p is None: result['reason'] = 'invalid_contour'; return result
    ys, xs = np.where(mask); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1-x0, y1-y0; cx = (x0+x1)/2
    center = p[(abs(p[:, 0]-cx) < .15*w) & (p[:, 1] < y0+.30*h)]
    if len(center) < 2: result['reason'] = 'collar_not_resolved'; return result
    collar = center[np.argmax(center[:, 1])]
    # Collar notch must be visible in the external contour; a guessed top-center is insufficient.
    if collar[1] < y0+.015*h: result['reason'] = 'collar_notch_not_resolved'; return result
    mid = (ys >= y0+.40*h) & (ys <= y0+.70*h)
    body = np.array([xs[mid].mean(), ys[mid].mean()])
    hem = p[p[:, 1] >= y1-.035*h].mean(0)
    nodes = dict(collar=collar.tolist(), body=body.tolist(), hem=hem.tolist())
    if kind == 'sleeved_upper':
        upper = p[(p[:, 1] > y0+.04*h) & (p[:, 1] < y0+.65*h)]
        if not len(upper): result['reason'] = 'sleeves_not_resolved'; return result
        left, right = upper[np.argmin(upper[:, 0])], upper[np.argmax(upper[:, 0])]
        body_pixels = xs[(ys >= y0+.55*h) & (ys <= y0+.65*h)]
        if len(body_pixels) == 0 or left[0] > body_pixels.min()-.025*w or right[0] < body_pixels.max()+.025*w:
            result['reason'] = 'sleeve_body_separation_uncertain'; return result
        nodes.update(left_sleeve=left.tolist(), right_sleeve=right.tolist())
    edges = []
    for node, pt in nodes.items():
        if node == 'body': continue
        path = np.linspace(pt, body, 40).round().astype(int)
        fraction = mask[path[:, 1].clip(0, SIZE[1]-1), path[:, 0].clip(0, SIZE[0]-1)].mean()
        if fraction < .70: result['reason'] = 'connection_not_supported_by_foreground'; return result
        edges.append([node, 'body'])
    result.update(nodes=nodes, edges=edges, valid=True)
    return result


def representations(mask, caption):
    p = contour(mask)
    c = None if p is None else np.fft.fft((p[:, 0]+1j*p[:, 1])/DIAGONAL)/len(p)
    return dict(A_SDF=(distance_transform_edt(~mask)-distance_transform_edt(mask))/DIAGONAL,
                B_FOURIER=c, C_GRAPH=graph(mask, caption), one_minus_IoU=mask)


def distance(a, b, name):
    if name == 'A_SDF': return float(np.abs(a-b).mean())
    if name == 'B_FOURIER':
        if a is None or b is None: return None
        freqs = np.r_[np.arange(0, 17), np.arange(-16, 0)]
        shifts = np.arange(256)/256
        aligned = b[freqs][None, :] * np.exp(2j*np.pi*shifts[:, None]*freqs[None, :])
        return float(np.sqrt(np.mean(abs(a[freqs][None, :]-aligned)**2, axis=1)).min())
    if name == 'C_GRAPH':
        if not a['valid'] or not b['valid']: return None
        shared = sorted(set(a['nodes']) & set(b['nodes']))
        geometry = np.mean([np.linalg.norm(np.array(a['nodes'][k])-b['nodes'][k])/DIAGONAL for k in shared])
        union = set(a['nodes']) | set(b['nodes'])
        missing = len(set(a['nodes']) ^ set(b['nodes']))/len(union)
        ea, eb = {tuple(e) for e in a['edges']}, {tuple(e) for e in b['edges']}
        topology = len(ea ^ eb)/max(1, len(ea | eb))
        return float(geometry+missing+topology)
    union = (a | b).sum()
    return float(1-(a & b).sum()/union) if union else None


def perturb(rgb, mask, kind, level):
    """appearance 不改变 mask；geometry 的 mask 与 RGB 使用相同逆映射。"""
    arr = rgb.astype(np.float32)/255
    h, w = mask.shape; y, x = np.mgrid[:h, :w].astype(np.float32)
    if kind in ['hue', 'brightness', 'texture']:
        change = arr.copy()
        if kind == 'hue':
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV); hsv[..., 0] = (hsv[..., 0]+level*360)%360
            change = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif kind == 'brightness': change = arr*level
        else:
            # 边界内外不混色，增加条纹与高频纹理；包括接近白色区域的分割压力测试。
            pattern = np.sin(x*2*np.pi/9)*np.cos(y*2*np.pi/13)
            change = arr+level*pattern[..., None]*np.array([1., -.7, .5])
        changed = np.where(mask[..., None], change, arr)
        return np.uint8(np.clip(changed, 0, 1)*255+.5), mask.copy()
    ys, xs = np.where(mask); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = max(1, x1-x0), max(1, y1-y0); cx = (x0+x1)/2
    xx, yy = x.copy(), y.copy()
    if kind == 'sleeve':
        band = np.exp(-.5*((y-(y0+.30*bh))/(.15*bh))**2)
        xx = cx+(x-cx)/(1+level*2*band)
    elif kind == 'hem':
        # 固定上半身；下半身向上压缩，真实移动衣摆而不是仅改变颜色。
        pivot = y0+.55*bh
        yy = np.where(y > pivot, pivot+(y-pivot)/(1-level), y)
    elif kind == 'contour':
        band = np.exp(-.5*((y-(y0+.65*bh))/(.22*bh))**2)
        xx = cx+(x-cx)/(1-level*band*(1+.4*np.sign(x-cx)))
    changed = cv2.remap(rgb, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    changed_mask = cv2.remap(mask.astype(np.uint8), xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    return changed, changed_mask
