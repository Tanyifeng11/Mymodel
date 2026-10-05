"""生成图重新运行冻结E26读出；case为唯一独立统计单位。"""
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from eval.eval_utils import estimate_foreground_mask
from eval.metrics import _dilate_binary, patch_texture_similarity, _binary_edges
from tools.e22_o4_metrics import contour
from data.e32_target_pseudogt import FrozenFeatures, image_at, upsample_geometry
from tools.e33r_evaluate import axial
from tools.e33tm_protocol import *

def bootstrap(values):
    values = np.asarray(values,float)
    if not len(values): return dict(mean=None,ci95=None,n=0)
    samples = np.random.default_rng(32042).choice(values,(2000,len(values)),replace=True).mean(1)
    return dict(mean=float(values.mean()),ci95=np.percentile(samples,[2.5,97.5]).tolist(),n=len(values))

def mean_available(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None

class Evaluator:
    def __init__(self):
        config = CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
        self.model = CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).cuda().eval().requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
        self.extractor = FrozenFeatures(None,'cpu')

    @torch.inference_mode()
    def evaluate(self,image,caption,reference,mask,sketch):
        args = self.processor(text=[caption],images=[image,reference],return_tensors='pt',padding=True,truncation=True)
        value = self.model(**{k:v.cuda() for k,v in args.items()})
        pixels = np.asarray(mask)>127
        foreground = estimate_foreground_mask(image,image.size)
        a,b = contour(foreground),contour(pixels)
        precision = (a&_dilate_binary(b,5)).sum()/max(a.sum(),1)
        recall = (b&_dilate_binary(a,5)).sum()/max(b.sum(),1)
        sketch_edges = _binary_edges(np.asarray(sketch.convert('L'),dtype=np.float32),.04,.12)
        generated_edges = _binary_edges(np.asarray(image.convert('L'),dtype=np.float32),.08,.16)
        # sketch similarity只测原sketch边缘的召回，不把衣身纹理边缘当假阳性。
        sketch_similarity = float((sketch_edges&_dilate_binary(generated_edges,5)).sum()/max(sketch_edges.sum(),1))
        inner = cv2.erode(pixels.astype(np.uint8),np.ones((17,17),np.uint8))>0
        geo = upsample_geometry(self.extractor.extract(image.resize((384,512),Image.Resampling.BICUBIC),
                                                     perceptual=False)['geometry'])
        scores = dict(text_score=float((value.text_embeds[0]*value.image_embeds[0]).sum()),
            clip_texture=float((value.image_embeds[0]*value.image_embeds[1]).sum()),
            texture_score=float(patch_texture_similarity(image,reference.resize(image.size),
                                mask=Image.fromarray(inner.astype(np.uint8)*255),patch=8)) if inner.any() else None,
            texture_support_pixels=int(inner.sum()),
            contour_f1=float(2*precision*recall/max(precision+recall,1e-12)),
            foreground_iou=float((foreground&pixels).sum()/max((foreground|pixels).sum(),1)),
            sketch_similarity=sketch_similarity)
        assert all(np.isfinite(v) for v in scores.values() if v is not None)
        return scores,geo.transpose(2,0,1)

def pair_metrics(geometries,support,weight):
    a,b,c = (geometries[k] for k in ('R0','R90','R180'))
    fixed = support.astype(bool)
    response = {}
    for arm,other,target in [('r90',b,-a[:2]),('r180',c,a[:2])]:
        valid = fixed&(a[3]>=.25)&(other[3]>=.25)
        coverage = float(valid.sum()/max(fixed.sum(),1))
        err = float(np.average(axial(other[:2],target)[valid],weights=weight[valid])) if valid.any() else None
        # 对读出不足的case保留分母；防止只挑少数可读patch获得成功。
        response[arm+'_error'] = err
        response[arm+'_coverage'] = coverage
        response[arm+'_success'] = bool(fixed.any() and valid.any() and err is not None and err<=15)
    response['fixed_support_cells'] = int(fixed.sum())
    return response

def summarize(rows):
    # 多个diffusion seed先按identity平均，bootstrap绝不把它们当独立N。
    ids = sorted({r['id'] for r in rows})
    fields = ('r90_success','r180_success','text_score','texture_score','clip_texture',
              'contour_f1','foreground_iou','sketch_similarity','r90_coverage','r180_coverage')
    cases = [{k:mean_available([r[k] for r in rows if r['id']==sid]) for k in fields} for sid in ids]
    stats = {k:bootstrap([r[k] for r in cases if r[k] is not None]) for k in fields}
    for k in ('r90_error','r180_error'):
        means = [float(np.mean([r[k] for r in rows if r['id']==sid and r[k] is not None]))
                 for sid in ids if any(r['id']==sid and r[k] is not None for r in rows)]
        stats[k] = bootstrap(means)
    arms={}
    if rows and 'arm_metrics' in rows[0]:
        for arm in ('R0','R90','R180','Rzero'):
            eligible=[r for r in rows if arm in r['arm_metrics']]
            if not eligible: continue
            arms[arm]={k:bootstrap([value for sid in ids
                                   for value in [mean_available([r['arm_metrics'][arm][k] for r in eligible if r['id']==sid])]
                                   if value is not None])
                       for k in ('text_score','texture_score','clip_texture','contour_f1','sketch_similarity')}
    return dict(case_count=len(ids),repeated_measures=len(rows),statistics=stats,arm_statistics=arms)

def compare(full,baseline):
    assert {r['id'] for r in full} == {r['id'] for r in baseline}
    keys = ('r90_success','r180_success','contour_f1','text_score','texture_score','sketch_similarity')
    lookup = {r['id']:r for r in baseline}
    result = {k:bootstrap([r[k]-lookup[r['id']][k] for r in full
                          if r[k] is not None and lookup[r['id']][k] is not None]) for k in keys}
    denom = np.mean([r['text_score'] for r in baseline])
    result['relative_text_drop'] = -result['text_score']['mean']/max(denom,1e-8)
    arm_drop={arm:(np.mean([r['arm_metrics'][arm]['text_score'] for r in baseline])-
                   np.mean([r['arm_metrics'][arm]['text_score'] for r in full]))/
                   max(np.mean([r['arm_metrics'][arm]['text_score'] for r in baseline]),1e-8)
              for arm in ('R0','R90','R180')}
    original_text=np.mean([r['arm_metrics']['R0']['text_score'] for r in full])
    rotated_drop={arm:(original_text-np.mean([r['arm_metrics'][arm]['text_score'] for r in full]))/
                  max(original_text,1e-8) for arm in ('R90','R180')}
    result['text_drop_by_arm']={k:float(v) for k,v in arm_drop.items()}
    result['text_rotation_drop']={k:float(v) for k,v in rotated_drop.items()}
    result['checks'] = dict(r90=np.mean([r['r90_success'] for r in full])>=.5,
        gain=result['r90_success']['mean']>=.2,paired_ci=result['r90_success']['ci95'][0]>0,
        r180=np.mean([r['r180_success'] for r in full])>=.8,
        contour=result['contour_f1']['mean']>=-.02,
        text=max(arm_drop.values())<=.02 and max(rotated_drop.values())<=.02,
        sketch=result['sketch_similarity']['ci95'][0]>=-.02)
    result['checks'] = {name:bool(value) for name,value in result['checks'].items()}
    result['pass'] = all(result['checks'].values())
    return result
