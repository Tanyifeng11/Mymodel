"""S0/S1 的只读 CPU 指标；沿用原 PNG 评测的 Lab、衣内掩膜和频带口径。"""
import cv2
import numpy as np


def lab(pixels):
    return cv2.cvtColor(np.asarray(pixels, np.float32) / 255, cv2.COLOR_RGB2LAB)


def highpass(field):
    return field - cv2.blur(field, (17, 17))


def precise_l(pixels):
    # 与11/16独立纹理评测的双精度sRGB→Lab一致，避免OpenCV近似导致幅度偏差。
    rgb = np.asarray(pixels, np.float64)/255
    linear = np.where(rgb>.04045, ((rgb+.055)/1.055)**2.4, rgb/12.92)
    y = linear @ np.array([.2126729,.7151522,.0721750])
    f = np.where(y>.008856, np.maximum(y,.008856)**(1/3), 7.787037*y+16/116)
    return (116*f-16)/100


def palette(ab):
    ab = np.asarray(ab).reshape(-1, 2)
    if len(ab) > 4096:
        ab = ab[np.linspace(0, len(ab)-1, 4096).round().astype(int)]
    angles = np.arange(32)*np.pi/32
    return np.quantile(ab @ np.stack((np.cos(angles), np.sin(angles))),
                       (np.arange(1024)+.5)/1024, axis=0)


def residue_components(current, positive, anchor):
    """输入衣内 Lab 高频，保留旧综合投影并显示跨通道抵消。"""
    template = anchor-positive
    template = template-positive*(np.sum(template*positive)/max(float(np.sum(positive*positive)), 1e-6))
    energy = float(np.mean(template**2))
    if energy < .25:
        signed = [None]*3
        combined = absolute_sum = None
    else:
        signed = [float(np.mean((current[:, c]-positive[:, c])*template[:, c])/(3*energy)) for c in range(3)]
        combined = float(abs(np.mean((current-positive)*template))/energy)
        absolute_sum = float(sum(abs(x) for x in signed))
    # L 独立正交，避免颜色贡献正负抵消；它与原 Lab 投影是不同代理指标。
    p, a, z = positive[:, 0], anchor[:, 0], current[:, 0]
    ltemplate = a-p
    ltemplate = ltemplate-p*(np.sum(ltemplate*p)/max(float(np.sum(p*p)), 1e-6))
    lenergy = float(np.mean(ltemplate**2))
    lprojection = float(abs(np.mean((z-p)*ltemplate))/lenergy) if lenergy >= .25 else None
    return dict(old_projection=combined, old_projection_L_signed=signed[0],
                old_projection_a_signed=signed[1], old_projection_b_signed=signed[2],
                old_projection_channel_abs_sum=absolute_sum, old_l_projection=lprojection,
                old_energy=energy, old_l_energy=lenergy)


def score_pixels(picture, positive, goal, anchor, mask, coords):
    mask = np.asarray(mask, bool)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) >= 16
    assert inside.any()
    current, target, original, base = map(lab, (picture, positive, goal, anchor))
    high = [highpass(v) for v in (current, target, base)]
    lhigh = [highpass(precise_l(v)) for v in (picture,positive)]
    a, p = lhigh[0][inside], lhigh[1][inside]
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    amplitude = rms(a)/max(rms(p), 1e-6)
    cosine = lambda x,y: float(np.sum(x*y)/max(float(np.linalg.norm(x)*np.linalg.norm(y)), 1e-12))
    corr = cosine(a, p)
    local = []
    for x, y in coords:
        take = inside[y:y+64, x:x+64]
        if take.any():
            local.append(cosine(lhigh[0][y:y+64, x:x+64][take], lhigh[1][y:y+64, x:x+64][take]))
    den = cv2.GaussianBlur(mask.astype(np.float32), (0,0), 24).clip(.001)
    def low_error(reference):
        delta = cv2.GaussianBlur((current[...,0]-reference[...,0])*mask, (0,0), 24)/den
        return float(np.abs(delta[inside]).mean())
    error = (current-target)/100
    dx = np.pad(np.diff(error, axis=1), ((0,0),(0,1),(0,0)))
    dy = np.pad(np.diff(error, axis=0), ((0,1),(0,0),(0,0)))
    result = dict(l_amplitude=amplitude, l_amplitude_error=abs(amplitude-1), l_correlation=corr,
        l_target_gain=amplitude*corr, l_target_gain_error=abs(amplitude*corr-1),
        local_l_correlation=float(np.mean(local)) if local else corr,
        l_hp_mae=float(np.abs(a-p).mean()), output_l_hp_rms=rms(a), target_l_hp_rms=rms(p),
        palette_swd=float(np.abs(palette(current[inside,1:])-palette(original[inside,1:])).mean()),
        spatial_ab_mae_positive=float(np.abs(current[inside,1:]-target[inside,1:]).mean()),
        spatial_ab_mae_goal=float(np.abs(current[inside,1:]-original[inside,1:]).mean()),
        low_l=low_error(original), low_l_positive=low_error(target),
        edge=float(np.abs(np.concatenate((dx,dy),axis=2))[inside].mean()),
        l_edge=float((np.abs(dx[...,0][inside]).mean()+np.abs(dy[...,0][inside]).mean())/2),
        amplitude_pass=bool(.75 <= amplitude <= 1.25),
        l_quality_pass=bool(.75 <= amplitude <= 1.25 and corr >= .5))
    result.update(residue_components(high[0][inside], high[1][inside], high[2][inside]))
    return result


COMPARE = ('l_amplitude_error', 'l_correlation', 'l_target_gain_error', 'l_hp_mae',
           'palette_swd', 'spatial_ab_mae_positive', 'spatial_ab_mae_goal', 'low_l', 'low_l_positive',
           'old_projection', 'old_projection_channel_abs_sum', 'old_l_projection', 'edge', 'l_edge')


def summarize(rows):
    def group(rr):
        result = dict(count=len(rr), metrics={})
        for key in COMPARE:
            valid = [(r['S0'][key],r['S1'][key]) for r in rr if r['S0'][key] is not None and r['S1'][key] is not None]
            if not valid:
                continue
            values = np.asarray(valid)
            delta = values[:,1]-values[:,0]
            improvement = delta if key == 'l_correlation' else -delta
            result['metrics'][key] = dict(S0=float(values[:,0].mean()),S1=float(values[:,1].mean()),
                delta=float(delta.mean()),better=int((improvement>1e-6).sum()),
                worse=int((improvement < -1e-6).sum()),tied=int((np.abs(improvement)<=1e-6).sum()),n=len(valid))
        for arm in ('S0','S1'):
            result[arm] = dict(amplitude_pass=sum(r[arm]['amplitude_pass'] for r in rr),
                l_quality_pass=sum(r[arm]['l_quality_pass'] for r in rr),
                under=sum(r[arm]['l_amplitude']<.75 for r in rr),
                over=sum(r[arm]['l_amplitude']>1.25 for r in rr))
        return result
    return dict(all=group(rows), by_group={k:group([r for r in rows if r['group']==k]) for k in sorted({r['group'] for r in rows})},
                by_case={k:group([r for r in rows if r['case']==k]) for k in sorted({r['case'] for r in rows})})
