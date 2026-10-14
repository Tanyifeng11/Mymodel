"""CPU 评价前回放固定历史 GPU 指标；不改变模型、指标口径或冻结门槛。"""
import argparse
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tools.e48_protocol import *

AUDIT=OUT/'cpu_backend_audit.json'
CODE=['tools/e48_eval.py','tools/e48_cpu_eval.py']


@torch.inference_mode()
def audit():
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    from tools.e33tmoc_appearance_eval import frozen_lpips
    started=time.perf_counter();metric,info=frozen_lpips()
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None]
    historic={(r['variant'],r['id']):r for r in read(PARENT/'p2/metrics.json')}
    records=[];hashes={}
    # 固定开发名单前两例、三种已有对照；只核验计算后端，不按结果挑身份。
    for row in read(OUT/'splits/dev32.json')[:2]:
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        ref=Image.open(row['reference']).convert('RGB')
        for arm in ['B','D','SAFE']:
            path=image_path(arm,row['id'],'dev32');image=Image.open(path).convert('RGB')
            hashes[str(path)]=sha(path)
            value=dict(lpips_gt=float(metric(tensor(image),tensor(gt)).mean()))
            inputs=processor(text=[row['caption']],images=[image,ref,gt],return_tensors='pt',padding=True,truncation=True)
            text=F.normalize(model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask')),dim=-1)
            features=F.normalize(model.get_image_features(pixel_values=inputs['pixel_values']),dim=-1)
            value.update(clip_text=float((text[0]*features[0]).sum()),clip_reference=float((features[0]*features[1]).sum()),clip_real=float((features[0]*features[2]).sum()))
            old=historic['D_safe' if arm=='SAFE' else arm,row['id']]
            differences={k:abs(v-old[k]) for k,v in value.items()}
            records.append(dict(id=row['id'],arm=arm,values=value,absolute_differences=differences))
            passed=all(v<=2e-6 for v in differences.values())
            write(AUDIT,dict(passed=passed and len(records)==6,records=records,input_hashes=hashes,
                tolerance=2e-6,metric='same verified AlexNet LPIPS and frozen CLIP, FP32',lpips=info,
                code_sha256={p:sha(p) for p in CODE},torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
                seconds=time.perf_counter()-started,commit=commit()))
            assert passed,(row['id'],arm,differences)
            print('E48 CPU REPLAY',row['id'],arm,max(differences.values()),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','evaluate_all']);args=parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    if args.action=='audit':audit();return
    record=read(AUDIT);assert record['passed'] and len(record['records'])==6
    assert record['code_sha256']=={p:sha(p) for p in CODE}
    assert all(sha(p)==h for p,h in record['input_hashes'].items())
    from tools.e48_eval import evaluate
    from tools.e48_train import TRAIN_ARMS
    # 全量 B/D/SAFE 仍逐身份执行原有 2e-6 历史一致性断言，失败立即停止。
    evaluate('probe',arms=ARMS+COMPONENT_ARMS+TRAIN_ARMS,device='cpu')
    evaluate('dev32',arms=ARMS+TRAIN_ARMS,device='cpu')


if __name__=='__main__':main()
