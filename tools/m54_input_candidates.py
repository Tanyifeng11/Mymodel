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


def ranked(rows,garment,pattern_index):
    group=[r for r in rows if r['clothing_group']==garment]
    group.sort(key=lambda r:(-(r['reference_prompt_scores'][pattern_index]-max(
        r['reference_prompt_scores'][j] for j in range(5) if j!=pattern_index)),order(r['id'])))
    return group


def panels(group,folder,garment,pattern,offset=0):
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(group),8):
        page=group[start:start+8];canvas=Image.new('RGB',(4*480,2*350),'white');draw=ImageDraw.Draw(canvas)
        for k,row in enumerate(page):
            x=(k%4)*480;y=(k//4)*350
            draw.text((x+3,y+3),row['id']+' '+garment+' '+pattern,fill='black')
            for j,key in enumerate(['sketch','reference','gt']):canvas.paste(Image.open(row[key]).convert('RGB').resize((160,213)),(x+j*160,y+24))
            caption=row['caption'].encode('ascii','replace').decode()
            for line in range(5):draw.text((x+3,y+242+line*17),caption[line*64:(line+1)*64],fill='black')
        canvas.save(folder/('page%03d.png'%(offset+start)))


def extend():
    """只扩展已冻结排序的下一段输入；不是依据候选模型输出重新挑样。"""
    assert not (OUT/'training').exists() and not (OUT/'round1/images').exists()
    source=OUT/'audit/input_candidates_r1/scores.json';rows=read(source)
    folder=OUT/'audit/input_candidates_ext1';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    group=ranked(rows,'pants',2)[32:128]
    panels(group,folder/'pants_symbol','pants','symbol',32)
    freeze(folder/'candidates.json',dict(pants_symbol=group))
    freeze(folder/'complete.json',dict(completed=True,group='pants_symbol',rank_start_zero_based=32,rank_stop_exclusive=128,
        n=len(group),frozen_scores_sha256=sha(source),ranking_unchanged=True,model_generation_used=False,
        reason='actual visual review of initial32 candidates established only4 pants-symbol identities including relabeling; fixed quota needs5',
        independent_human_review='NOT_RUN',scientific_training_updates=0,commit=commit()))
    archive=OUT/'m54_input_candidates_ext1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in folder.rglob('*'):
            if path.is_file():tar.add(path,arcname=str(path.relative_to(folder)))
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print('M54 INPUT EXTENSION READY',len(group),flush=True)


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
            group=ranked(rows,garment,i)
            candidates[garment+'_'+pattern]=group[:32]
            panels(group[:32],folder/(garment+'_'+pattern),garment,pattern)
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
