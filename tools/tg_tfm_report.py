"""开发集配对评价；CLIP 只是辅助证据，不能替代属性人工评审。"""
from collections import defaultdict
import csv
import sys
import numpy as np
from PIL import Image,ImageDraw
from tools.tg_tfm_experiment import PROJECT,ROOT,ARMS,read,write,sha
from tools.bf_t3_three_arm_report import paired


def table(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def evaluate_screen():
    import torch
    import torch.nn.functional as F
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation,compute_texture_color_fidelity,compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import bf_t3_frozen_core as core
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows=read(ROOT/'dev_evaluation.json')
    completes={arm:read(ROOT/'screen'/arm/'complete.json') for arm in ARMS}
    first=completes['T0']
    for arm,value in completes.items():
        assert value['updates']==1000 and value['run']==first['run']
        assert value['stream_hashes']==first['stream_hashes']
    write(ROOT/'audit/screen_comparability.json',dict(passed=True,training=completes))
    lpips,assets=frozen_lpips()
    # 服务器models/clip仅含嵌套的vision-only模型，不能据此随机补齐文本侧。
    # 统一复用已缓存的官方完整ViT-B/32；这只是辅助评价网络，未改变生成模型。
    clip_root=PROJECT/'models/CLIP'
    sys.path.insert(0,str(clip_root))
    import clip as clip_api
    weights=PROJECT.parent/'.cache/clip/ViT-B-32.pt'
    expected='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    actual=sha(weights)
    if actual!=expected:
        raise ValueError('CLIP缓存权重与官方SHA256不一致')
    clip,processor=clip_api.load(str(weights),device='cuda',jit=False)
    clip=clip.float().eval().requires_grad_(False)
    write(ROOT/'audit/clip_assets.json',dict(backend='OpenAI native CLIP',model='ViT-B/32',
        weight_path=str(weights),weight_sha256=actual,official_sha256=expected,precision='float32; TF32 off',
        source_sha256={name:sha(clip_root/'clip'/name) for name in ('clip.py','model.py','simple_tokenizer.py')},
        evaluation_source_sha256=sha(core.SOURCE/'tools/tg_tfm_report.py')))
    material=['silk','satin','velvet','denim','leather','cotton','linen','wool','knitted','lace','chiffon']
    texts=['a photograph of a '+label+' garment.' for label in material]
    with torch.inference_mode():
        text_features=F.normalize(clip.encode_text(clip_api.tokenize(texts).cuda()).float(),dim=-1)
    records=[]
    with torch.inference_mode():
        for row in rows:
            masks=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
            latent_hashes=[]
            labels=[('knitted' if x=='knit' else x) for x in row['attributes']['material']]
            positive=[material.index(x) for x in labels if x in material]
            for arm in ARMS:
                path=ROOT/'screen'/arm/'outputs'/row['condition']/(row['id']+'.png')
                meta=read(path.with_suffix('.json'))
                assert meta['output_sha256']==sha(path) and not meta['GT_used_in_inference']
                assert meta['caption']==row['caption'] and meta['cfg_calls']==[50,50]
                latent_hashes.append(meta['initial_latent_sha256'])
                kwargs=dict(gen_path=str(path),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=masks)
                structure=compute_structure_preservation(**kwargs)
                image=Image.open(path).convert('RGB')
                image_features=F.normalize(clip.encode_image(processor(image).unsqueeze(0).cuda()).float(),dim=-1)
                scores=(image_features@text_features.T)[0]
                wrong=[i for i in range(len(material)) if i not in positive]
                attr_margin=float(scores[positive].max()-scores[wrong].max()) if positive else None
                record=dict(arm=arm,id=row['id'],category=row['category'],condition=row['condition'],
                    edge_f1=structure.get('edge_f1'),iou=structure.get('sketch_iou'),edge_l1=structure.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'),
                    lpips=float(lpips(core.tensor(path).cuda(),core.tensor(row['gt']).cuda()).mean()),
                    tcf_lab=compute_texture_color_fidelity(texture_path=row['reference'],**kwargs).get('tcf_lab_delta'),
                    material_clip_margin=attr_margin,
                    material_clip_correct=float(int(scores.argmax()) in positive) if positive else None,
                    seconds=meta['seconds'])
                records.append(record)
            assert len(set(latent_hashes))==1
    write(ROOT/'screen_metrics/per_image.json',records)
    table(ROOT/'screen_metrics/per_image.csv',records)
    metrics=['edge_f1','iou','edge_l1','leak','lpips','tcf_lab','material_clip_margin','material_clip_correct','seconds']
    summary=[]
    for arm in ARMS:
        for condition in ['original','neutral']:
            selected=[r for r in records if r['arm']==arm and r['condition']==condition]
            for metric in metrics:
                values=[r[metric] for r in selected if r[metric] is not None]
                summary.append(dict(arm=arm,condition=condition,metric=metric,n=len(values),
                                    mean=float(np.mean(values)) if values else None))
    write(ROOT/'screen_metrics/summary.json',summary);table(ROOT/'screen_metrics/summary.csv',summary)
    lookup={(r['arm'],r['id'],r['condition']):r for r in records}
    comparisons=[]
    identities=sorted({r['id'] for r in rows})
    for arm in ARMS[1:]:
        for metric in metrics[:-1]:
            values=[]
            for ident in identities:
                a,b=lookup[arm,ident,'original'][metric],lookup['T0',ident,'original'][metric]
                if a is not None and b is not None:values.append(a-b)
            comparisons.append(dict(arm=arm,metric=metric,contrast='original minus T0',**paired(values)))
        values=[]
        for ident in identities:
            a,b,c,d=[lookup[x,ident,y]['material_clip_margin'] for x,y in
                     [(arm,'original'),(arm,'neutral'),('T0','original'),('T0','neutral')]]
            if all(v is not None for v in [a,b,c,d]): values.append((a-b)-(c-d))
        comparisons.append(dict(arm=arm,metric='material_clip_response',contrast='(caption-neutral) minus T0',**paired(values)))
    write(ROOT/'screen_metrics/paired_statistics.json',comparisons)
    # 固定输入哈希选样，所有32件另存；AI审图必须明确与人工盲评分开。
    originals=sorted([r for r in rows if r['condition']=='original'],key=lambda r:sha(r['sketch']))
    for page,start in enumerate(range(0,len(originals),4)):
        panel=Image.new('RGB',(1080,1040),'white');draw=ImageDraw.Draw(panel)
        for j,row in enumerate(originals[start:start+4]):
            paths=[row['sketch'],row['reference']]+[ROOT/'screen'/a/'outputs/original'/(row['id']+'.png') for a in ARMS]
            for col,(label,path) in enumerate(zip(['Sketch','Reference']+ARMS,paths)):
                draw.text((col*180+3,j*260+2),label,fill='black')
                image=Image.open(path).convert('RGB');image.thumbnail((174,226))
                panel.paste(image,(col*180,j*260+22))
            draw.text((3,j*260+246),row['id'],fill='black')
        (ROOT/'figures').mkdir(exist_ok=True)
        panel.save(ROOT/'figures'/('screen_%02d.png'%page))
    write(ROOT/'screen_complete.json',dict(complete=True,outputs=len(records),identities=32,
        stage='A engineering screen; not formal paper comparison',human_reviewers=0,
        next='inspect paired statistics and images before deciding B/C and formal budget',lpips_assets=assets))


if __name__=='__main__':evaluate_screen()
