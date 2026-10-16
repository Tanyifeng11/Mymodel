"""输入审阅R1：冻结CLIP仅排序参考裁片候选，最终类别须由助手目视确认。"""
from pathlib import Path
import json
import tarfile
from collections import Counter
import torch
from PIL import Image,ImageDraw
from tools.m54_prepare import order
from tools.m54_protocol import *

PROMPTS=[
 'a close-up textile fabric with visible parallel stripes',
 'a close-up textile fabric with a visible plaid or checked grid pattern',
 'a textile print showing a recognizable logo, letters, text or discrete graphic symbols',
 'a close-up textile fabric with a non-periodic floral, paisley, abstract or mottled pattern',
 'a plain solid color fabric with no visible printed pattern',
]


def clothing(row):
    text=row['caption'].lower()
    excluded=['bag','handbag','crossbody','tote','shoe','boot','sandal','sneaker','belt','scarf','hat','cap ']
    # "belt"可能是衣服自带腰带，不作排除；包/鞋等非服装明确排除。
    if any(word in text for word in excluded if word!='belt'):return None
    if any(word in text for word in ['pants','trousers','jeans','shorts']):return 'pants'
    if any(word in text for word in ['shirt','blouse','sweater','coat','jacket','blazer','cardigan','dress','vest','top','poncho','romper','jumpsuit','skirt']):return 'clothing'
    return None


@torch.inference_mode()
def build():
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    from tools.e48_holdout import exposure_sources
    folder=OUT/'audit/input_candidates_r1';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    prior=read(HISTORY/'holdout_audit/eligible.json')
    exposed,registry=exposure_sources({r['id'] for r in prior})
    # 本轮初版只是输入预览，未消费模型结果；剔除其registry条目后重建历史暴露。
    own_ids={r['id'] for name in ['pilot','lock','natural'] for r in read(OUT/'splits'/(name+'.json'))}
    historical=set()
    for entry in registry:
        if str(entry['path']).replace('\\','/').startswith(str(OUT).replace('\\','/')+'/'):continue
        historical.update(entry['validation_ids'])
    held={r['id'] for name in ['dev32','train1024'] for r in read(OUT/'splits'/(name+'.json'))}
    pool=[dict(r,clothing_group=clothing(r)) for r in prior if r['id'] not in historical|held and clothing(r)]
    pool=sorted(pool,key=lambda r:order(r['id']))
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    text=processor(text=PROMPTS,return_tensors='pt',padding=True)
    text_feature=torch.nn.functional.normalize(model.get_text_features(**text).float(),dim=-1)
    rows=[];cache=folder/'scores.json'
    if cache.exists():rows=read(cache)
    done={r['id'] for r in rows}
    remaining=[r for r in pool if r['id'] not in done]
    for start in range(0,len(remaining),4):
        batch=remaining[start:start+4]
        images=[Image.open(r['reference']).convert('RGB') for r in batch]
        inputs=processor(images=images,return_tensors='pt')
        feature=torch.nn.functional.normalize(model.get_image_features(**inputs).float(),dim=-1)
        scores=(feature@text_feature.T).tolist()
        rows.extend(dict(r,reference_prompt_scores=s) for r,s in zip(batch,scores))
        if start%100==0:write(cache,rows);print('M54 INPUT LABEL SORT',start,len(remaining),flush=True)
    write(cache,rows)
    candidates={};chosen=set()
    patterns=['stripe','grid','symbol','non-periodic']
    for garment in ['pants','clothing']:
        for i,pattern in enumerate(patterns):
            group=[r for r in rows if r['clothing_group']==garment]
            group.sort(key=lambda r:(-(r['reference_prompt_scores'][i]-max(r['reference_prompt_scores'][j] for j in range(5) if j!=i)),order(r['id'])))
            candidates[garment+'_'+pattern]=group[:32]
            pagefolder=folder/(garment+'_'+pattern);pagefolder.mkdir(exist_ok=True)
            for start in range(0,min(32,len(group)),8):
                page=group[start:start+8];canvas=Image.new('RGB',(4*480,2*350),'white');draw=ImageDraw.Draw(canvas)
                for k,row in enumerate(page):
                    x=(k%4)*480;y=(k//4)*350
                    draw.text((x+3,y+3),row['id']+' '+garment+' '+pattern,fill='black')
                    for j,key in enumerate(['sketch','reference','gt']):canvas.paste(Image.open(row[key]).convert('RGB').resize((160,213)),(x+j*160,y+24))
                    caption=row['caption'].encode('ascii','replace').decode()
                    for line in range(5):draw.text((x+3,y+242+line*17),caption[line*64:(line+1)*64],fill='black')
                canvas.save(pagefolder/('page%03d.png'%start))
    write(folder/'candidates.json',candidates)
    write(folder/'complete.json',dict(completed=True,n=len(rows),groups=dict(Counter(r['clothing_group'] for r in rows)),
        prompt_texts=PROMPTS,method='frozen CLIP ranking of actual reference crop; assistant must verify labels; no generation used',
        scientific_training_updates=0,independent_human_review='NOT_RUN',original_input_selection_preserved=True,commit=commit()))
    archive=OUT/'m54_input_candidates_r1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in folder.rglob('*'):
            if p.is_file() and p.name!='scores.json':tar.add(p,arcname=str(p.relative_to(folder)))
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print('M54 INPUT CANDIDATES READY',len(rows),flush=True)


if __name__=='__main__':build()
