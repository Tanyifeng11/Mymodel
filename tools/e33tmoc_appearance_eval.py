"""固定支持与patch列表的外观评测；真实LPIPS，缺失值保留并报告。"""
import argparse
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from models.local_pattern_field import patch_geometry
from eval.metrics import patch_texture_similarity
from data.e32_target_pseudogt import image_at
from data.e33rf_real_rotation_dataset import rotate
from tools.e33tmif_metrics import bootstrap
from tools.e33tmoc_protocol import *

APPEARANCE_PROTOCOL=dict(source_mask='existing estimate_cloth_foreground_mask',
    color_support='source foreground and target17px-eroded sketch garment mask, no GT RGB',
    color_distance='Euclidean distance of foreground mean OpenCV Lab; OpenCV8bit units',
    histogram='cosine of concatenated8bins/channel Lab histograms',
    texture='unchanged patch_texture_similarity,8px argument, original target interior mask',
    lpips='actual pretrained alex LPIPS, fixed up to16 patches per arm,unaligned source/target crops; orientation may affect LPIPS',
    patch_size=64,patch_stride=32,source_q_min=.25,source_std_min=.02,source_foreground_min=.95,
    patch_selection='SHA256 OC/appearance/id/arm/source_or_target/y/x; no generated-image selection',
    missing='no eligible source/target patch => LPIPS NA; preserve direction fixed denominator',training_steps=0)

def patches(image,mask,sid,arm,kind):
    result=[]
    for y in range(0,image.height-63,32):
        for x in range(0,image.width-63,32):
            if mask[y:y+64,x:x+64].mean()<.95:continue
            patch=image.crop((x,y,x+64,y+64))
            if kind=='source':
                info=patch_geometry(patch)
                if info['confidence']<.25 or np.asarray(patch.convert('L'),float).std()/255<.02:continue
            result.append((hashlib.sha256(('OC/appearance/%s/%s/%s/%d/%d'%(sid,arm,kind,y,x)).encode()).hexdigest(),[x,y,x+64,y+64]))
    return [box for _,box in sorted(result)[:16]]

def array(image,boxes):
    return torch.from_numpy(np.stack([np.asarray(image.crop(tuple(box)),np.float32).transpose(2,0,1)/127.5-1 for box in boxes]))

def histogram(values):
    hist=np.concatenate([np.histogram(values[:,c],bins=8,range=(0,256))[0] for c in range(3)]).astype(float)
    return hist/max(np.linalg.norm(hist),1e-12)

def measure(image,reference,inner,source_mask,target_boxes,source_boxes,metric):
    rgb=np.asarray(image);ref_rgb=np.asarray(reference)
    target=rgb[inner].astype(float);source=ref_rgb[source_mask].astype(float)
    result=dict(target_pixels=len(target),source_pixels=len(source),patch_lpips=None,
        lpips_patch_count=min(len(target_boxes),len(source_boxes)),texture_score=None,
        Lab_color_distance=None,color_histogram_similarity=None,mean_RGB_distance=None,
        target_mean_RGB=target.mean(0).tolist() if len(target) else None,
        source_mean_RGB=source.mean(0).tolist() if len(source) else None,
        target_std_RGB=target.std(0).tolist() if len(target) else None,
        source_std_RGB=source.std(0).tolist() if len(source) else None)
    if len(target) and len(source):
        lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)[inner].astype(float)
        ref_lab=cv2.cvtColor(ref_rgb,cv2.COLOR_RGB2LAB)[source_mask].astype(float)
        result.update(Lab_color_distance=float(np.linalg.norm(lab.mean(0)-ref_lab.mean(0))),
            color_histogram_similarity=float(histogram(lab)@histogram(ref_lab)),
            mean_RGB_distance=float(np.linalg.norm(target.mean(0)-source.mean(0))))
    if inner.any():
        result['texture_score']=float(patch_texture_similarity(image,reference,
            mask=Image.fromarray(inner.astype(np.uint8)*255),patch=8))
    n=result['lpips_patch_count']
    if n:
        with torch.inference_mode():result['patch_lpips']=float(metric(array(image,target_boxes[:n]),array(reference,source_boxes[:n])).mean())
    return result

def model_hash(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def frozen_lpips():
    """使用服务器现有官方AlexNet缓存，避免新版文件名触发在线下载。"""
    import lpips
    path=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-4df8aa71.pth'
    digest=sha(path)
    assert digest.startswith('4df8aa71'), '必须匹配TorchVision官方旧AlexNet文件指纹'
    metric=lpips.LPIPS(net='alex',pnet_rand=True)
    pretrained=torch.load(path,map_location='cpu')
    # LPIPS按slice分组保留原features索引；严格载入全部卷积张量后才允许评测。
    metric.net.load_state_dict({k:pretrained['features.'+k.split('.',1)[1]] for k in metric.net.state_dict()},strict=True)
    metric.eval().requires_grad_(False)
    return metric,dict(alexnet_checkpoint=str(path),alexnet_checkpoint_sha256=digest,
        trunk_initialization='all AlexNet features replaced with verified official pretrained checkpoint before inference',
        lpips_linear_calibration='bundled pretrained LPIPS v0.1 alex weights')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='+',default=['C0_current','C1_analytic']);args=parser.parse_args()
    import lpips
    torch.set_num_threads(2);metric,initialization=frozen_lpips()
    digest=model_hash(metric)
    record=dict(**APPEARANCE_PROTOCOL,lpips_module=str(Path(lpips.__file__)),lpips_state_sha256=digest,**initialization)
    p=OUT/'protocol/appearance_metrics.json'
    if p.exists():assert read(p)==record
    else:write(p,record)
    rows=prepare();ids=read(OUT/'splits/diagnostic64.json');input_hashes=freeze_carrier_inputs(rows,ids)
    collected={name:{stage:[] for stage in ('S1','S2')} for name in args.names}
    for row in rows:
        sid=row['id']
        if sid not in ids:continue
        folder=OUT/'geometry_audit/real'/sid
        with np.load(folder/'mapping.npz') as z:mask=z['target_support'].copy()
        inner=cv2.erode(mask.astype(np.uint8),np.ones((17,17),np.uint8))>0
        ref=image_at(DATASET/row['reference']);refs=dict(zip(('R0','R90','R180'),(ref,rotate(ref,90),rotate(ref,180))))
        target_boxes=patches(ref,inner,sid,'shared','target')
        for arm,reference in refs.items():
            source_mask=np.asarray(estimate_cloth_foreground_mask(reference,*reference.size)[0])>127
            source_boxes=patches(reference,source_mask,sid,arm,'source')
            write(OUT/'appearance_metrics/patch_lists'/sid/(arm+'.json'),dict(source_boxes=source_boxes,target_boxes=target_boxes))
            for name in args.names:
                for stage in ('S1','S2'):
                    path=OUT/'candidates'/name/'seed42'/sid/stage/(arm+'.png')
                    result=measure(Image.open(path).convert('RGB'),reference,inner,source_mask,target_boxes,source_boxes,metric)
                    result.update(id=sid,arm=arm,stage=stage,carrier=name,image_sha256=sha(path))
                    write(OUT/'appearance_metrics'/name/sid/stage/(arm+'.json'),result)
                    collected[name][stage].append(result)
        print('[OC appearance]',sid,flush=True)
    summary={}
    for name,stages in collected.items():
        summary[name]={}
        for stage,records in stages.items():
            cases=[dict(id=sid,**{key:float(np.mean(values)) if values else None for key in
                ('Lab_color_distance','color_histogram_similarity','texture_score','patch_lpips','mean_RGB_distance')
                for values in [[r[key] for r in records if r['id']==sid and r[key] is not None]]}) for sid in ids]
            summary[name][stage]=dict(case_count=len(cases),statistics={key:bootstrap([r[key] for r in cases])
                for key in ('Lab_color_distance','color_histogram_similarity','texture_score','patch_lpips','mean_RGB_distance')})
            write(OUT/'appearance_metrics'/name/(stage+'_cases.json'),cases)
    write(OUT/'appearance_metrics/summary.json',summary)
    assert model_hash(metric)==digest and {p:sha(p) for p in input_hashes}==input_hashes
    frozen_check();print('[OC appearance complete]',summary,flush=True)

if __name__=='__main__':main()
