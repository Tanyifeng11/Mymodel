"""独立保护亮度纹样；投影的是含动量、预调节和衰减的实际参数位移。"""
import numpy as np
import torch

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_material_losses import highpass

TASKS = ('pattern', 'amplitude', 'correlation', 'residue', 'color', 'light', 'edge')


def luminance(rgb, positive, weight, coords):
    a, b = [highpass(rgb_to_lab(x)[:, :1] / 100) for x in (rgb, positive)]
    inside = (weight >= .999).to(a.dtype)
    fields = [(a, b, inside)]
    fields += [(a[..., y:y+64, x:x+64], b[..., y:y+64, x:x+64],
                inside[..., y:y+64, x:x+64]) for x, y in coords]
    amplitudes, correlations, ratios, cosines = [], [], [], []
    for p, q, w in fields:
        count = w.sum().clamp_min(1)
        p2, q2 = (p.square()*w).sum()/count, (q.square()*w).sum()/count
        ratio = ((p2+1e-10)/(q2+1e-10)).sqrt()
        cosine = ((p*q*w).sum()/count) / ((p2+1e-10)*(q2+1e-10)).sqrt()
        valid = (q2.detach() > 1e-8).to(p.dtype)
        amplitudes.append(valid*ratio.log().square())
        correlations.append(valid*(1-cosine).clamp_min(0))
        ratios.append(ratio.detach());cosines.append(cosine.detach())
    return dict(amplitude=torch.stack(amplitudes).mean(),
                correlation=torch.stack(correlations).mean()), dict(
                    l_amplitude=float(ratios[0]), l_correlation=float(cosines[0]))


def groups(old, extra):
    return dict(pattern=old['latent']+2*old['rgb']+old['frequency']+old['high']+2*old['protection'],
                amplitude=extra['amplitude'], correlation=extra['correlation'],
                residue=old['projection'],
                color=old['chroma']+old['mean_ab']+old['palette'], light=old['light'], edge=10*old['edge'])


def task_gradients(values, parameters):
    rows=[]
    for index,key in enumerate(TASKS):
        parts=torch.autograd.grad(values[key],parameters,retain_graph=index<len(TASKS)-1,allow_unused=True)
        rows.append(torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                               for p,g in zip(parameters,parts)]))
    return torch.stack(rows)


def project_update(displacement, gradients, tolerance=1e-6, margin=.01, max_sweeps=512):
    """每项有下降余量，避免投影在切向边界后被曲率和半精度舍入推向上升。"""
    norms=gradients.norm(dim=1)
    rows=gradients[norms>1e-12]/norms[norms>1e-12,None]
    length=displacement.norm()
    if not len(rows) or float(length)==0:
        return displacement.clone(),dict(feasible=True,minimum_dot=0.,relative_correction=0.)
    # 只在CPU解至多14维的问题，避免每次坐标更新触发GPU同步。
    gram=(rows@rows.T).double().cpu().numpy()
    offset=(rows@(displacement/length)).double().cpu().numpy()-margin
    dual=np.zeros(len(rows))
    for _ in range(max_sweeps):
        for i in range(len(rows)):
            dual[i]=max(0.,dual[i]-(offset[i]+gram[i]@dual)/max(gram[i,i],1e-12))
        if (offset+gram@dual).min()>=-tolerance/4:break
    result=displacement+length*(rows.T@torch.as_tensor(dual,device=rows.device,dtype=rows.dtype))
    minimum=float((rows@result).min()/length)
    feasible=minimum>=margin-tolerance
    # 未收敛时拒绝本步，不能将零位移记成成功训练。
    return result if feasible else torch.zeros_like(result),dict(
        feasible=feasible,minimum_dot=minimum,descent_margin=margin,
        relative_correction=float((result-displacement).norm()/length))


def flatten(parameters):
    return torch.cat([p.detach().flatten() for p in parameters]).clone()


@torch.no_grad()
def set_parameters(parameters, vector):
    offset=0
    for p in parameters:
        p.copy_(vector[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    assert offset==vector.numel()
