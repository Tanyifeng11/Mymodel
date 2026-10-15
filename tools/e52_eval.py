"""GPU 旧指标回放；合成纹样指标另列，不能用 LPIPS 替代。"""
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from tools.e52_protocol import *
from tools.e44_metrics import contour_f1,external_contour

def replay_metrics(rows):
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask
    from tools.e44_runtime import sketch_masks
    metric,backend=frozen_lpips();metric=metric.cuda()
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    old={(x['id'],x['variant']):x for x in read(SOURCE/'dev32/metrics.json')}
    records=[];errors=[]
    with torch.inference_mode():
        for row in rows:
            gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            garment=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')['garment']
            valid=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE),'cpu')[2]['valid']
            for arm,historical in [('E5','B'),('B1','D')]:
                image=Image.open(image_path('p0',arm,row['id'])).convert('RGB')
                rec=dict(id=row['id'],arm=arm,lpips_gt=float(metric(tensor(image),tensor(gt)).mean()),contour_f1=None,iou=None)
                if valid:
                    fg=estimate_foreground_mask(image,SIZE)
                    if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
                    rec.update(contour_f1=contour_f1(fg,garment),iou=float((fg&garment).sum()/max((fg|garment).sum(),1)))
                for key in ['lpips_gt','contour_f1','iou']:
                    expected=old[(row['id'],historical)][key]
                    assert (expected is None)==(rec[key] is None)
                    if expected is not None:
                        error=abs(rec[key]-expected);errors.append(error)
                        assert error<=2e-6,('GPU_METRIC_REPLAY_FAIL',row['id'],arm,key,error)
                records.append(rec)
    write(OUT/'audit/replay_metrics.json',records)
    return dict(backend=backend,n_images=len(records),max_abs_error=max(errors),tolerance=2e-6)
