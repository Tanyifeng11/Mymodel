"""冻结输入及候选卡；先复核历史签名，再用输入标签分层，不查看候选输出。"""
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
from tools.m54_protocol import *


def order(identity):return hashlib.sha256(('M54_FIXED_54042_'+identity).encode()).hexdigest()


def labels(row):
    text=row['caption'].lower()
    if any(v in text for v in ['stripe','striped']):pattern='stripe'
    elif any(v in text for v in ['plaid','checkered','checked','gingham','grid']):pattern='grid'
    elif any(v in text for v in ['logo','graphic','letter','symbol']):pattern='symbol'
    elif any(v in text for v in ['floral','flower','abstract','paisley','leopard','camouflage','print','pattern']):pattern='non-periodic'
    else:pattern='unknown'
    pants=any(v in text for v in ['pants','trousers','jeans','shorts'])
    return pattern,pants


def balanced(pool,n,excluded):
    groups=['sleeve','hem','collar/shoulder','pant-leg'];patterns=['stripe','grid','symbol','non-periodic'];selected=[]
    candidates=sorted(pool,key=lambda r:order(r['id']))
    # 先选择要求裤装的槽位，避免上衣槽位消耗稀少裤装。
    slots=sorted(range(n),key=lambda k:(k%4!=3,k))
    byslot={}
    for k in slots:
        kind=groups[k%4];pattern=patterns[(k+k//4)%4]
        match=next((r for r in candidates if r['id'] not in excluded and labels(r)==(pattern,kind=='pant-leg')),None)
        if match is None:
            write(OUT/'audit/stratum_shortage.json',dict(n=n,slot=k,geometry=kind,pattern=pattern,
                counts=dict(Counter(str(labels(r)) for r in candidates if r['id'] not in excluded)),
                action='input annotation/eligibility review required; no relaxed quota or model-based substitution'))
            raise RuntimeError('INPUT_STRATUM_SHORTAGE; see audit/stratum_shortage.json')
        excluded.add(match['id']);byslot[k]=dict(match,geometry_group=kind,pattern_group=pattern,
            label_source='caption keyword plus predefined geometry focus; assistant visual review pending')
    return [byslot[k] for k in range(n)]


def panels(rows,name):
    folder=OUT/'audit/input_review'/name;folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),8):
        page=rows[start:start+8];canvas=Image.new('RGB',(4*384,2*300),'white');draw=ImageDraw.Draw(canvas)
        for k,row in enumerate(page):
            x=(k%4)*384;y=(k//4)*300
            draw.text((x+3,y+3),row['id']+' '+row.get('geometry_group','natural')+' '+row.get('pattern_group',''),fill='black')
            for j,key in enumerate(['sketch','reference','gt']):canvas.paste(Image.open(row[key]).convert('RGB').resize((128,171)),(x+j*128,y+25))
            text=row['caption'].encode('ascii','replace').decode()
            for line in range(4):draw.text((x+3,y+200+line*18),text[line*52:(line+1)*52],fill='black')
        canvas.save(folder/('page%03d.png'%start))


def prepare():
    freeze(OUT/'config.json',CONFIG)
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    model_files=read('output_eval/e53_b1_oracle_20261016/manifest/model_file_sha256.json')
    assert all(sha(p)==h for p,h in model_files.items()),'FROZEN_MODEL_CHANGED'
    freeze(OUT/'manifests/model_file_hashes.json',model_files)
    for name in ['train1024.json','train1024_records.json','dev32.json']:
        freeze(OUT/'splits'/name,read(HISTORY/'splits'/name))
    freeze(OUT/'manifests/e5_training_meta.json',read(HISTORY/'manifests/e5_training_meta.json'))
    for arm in TRAIN_ARMS+ZERO_ARMS:
        kind=arm.split('_')[0];folder=OUT/'modules'/arm
        write(folder/'config.yaml',dict(arm=arm,protocol_sha256=sha(OUT/'config.json'),paper=PAPERS.get(kind)))
        # 用户方案明确要求每模块卡；只描述实际实现，不冒充完整论文复现。
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'module_card.md').write_text('# '+arm+'\n\n文献机制适配，非完整论文复现。\n\n'+str(CONFIG['revisions'].get(kind,CONFIG.get(kind,'按冻结计算路径实现')))+'\n',encoding='utf-8')
    if (OUT/'audit/prepare_complete.json').exists():return
    # 历史eligibility已审计训练身份/文件/像素/pHash；再排除本轮之前已消费的自然身份。
    eligible=read(HISTORY/'holdout_audit/eligible.json')
    from tools.e48_holdout import exposure_sources
    exposed,registry=exposure_sources({r['id'] for r in eligible})
    freeze(OUT/'audit/exposure_registry.json',registry)
    held={r['id'] for r in read(OUT/'splits/dev32.json')+read(OUT/'splits/train1024.json')}
    pool=[r for r in eligible if r['id'] not in exposed|held]
    counts=dict(Counter(str(labels(r)) for r in pool))
    write(OUT/'audit/data_inventory.json',dict(prior_eligible=len(eligible),historical_exposed=len(exposed),available=len(pool),caption_strata=counts,
        limitation='caption labels require assistant input visual review; not independent human labels; Writer pretraining overlap unknown'))
    excluded=set(held)|exposed
    pilot=balanced(pool,24,excluded);lock=balanced(pool,64,excluded)
    natural=sorted([r for r in pool if r['id'] not in excluded],key=lambda r:order(r['id']))[:96]
    assert len(natural)==96,'NATURAL96_INSUFFICIENT'
    for name,rows in [('pilot',pilot),('lock',lock),('natural',natural)]:freeze(OUT/'splits'/(name+'.json'),rows);panels(rows,name)
    selected=pilot+lock+natural+read(OUT/'splits/dev32.json')+read(OUT/'splits/train1024.json')
    hashes={r[k]:sha(r[k]) for r in selected for k in ['gt','sketch','reference']}
    old_signatures=read(HISTORY/'holdout_audit/validation_signatures.json')
    for row in pilot+lock+natural:
        for k in ['gt','sketch','reference']:assert hashes[row[k]]==old_signatures[row[k]]['file_sha256']
    freeze(OUT/'manifests/input_hashes.json',hashes)
    # Oracle只作已合格合成Pilot上的路线诊断，不把其结果冒充自然候选验收。
    freeze(OUT/'splits/oracle_diagnostic.json',read('output_eval/e53_b1_oracle_20261016/splits/pilot_r2.json'))
    write(OUT/'audit/prepare_complete.json',dict(passed=True,identity_disjoint=True,pilot=24,lock=64,natural=96,dev=32,train=1024,
        input_files=len(hashes),label_review='PENDING_ASSISTANT_INPUT_REVIEW',model_outputs_used_for_selection=False,
        independent_human_review='NOT_RUN',strict_unseen=False,commit=commit()))
    import tarfile
    with tarfile.open(OUT/'m54_input_review.tar.gz','w:gz') as archive:
        for name in ['pilot','lock','natural']:
            archive.add(OUT/'splits'/(name+'.json'),arcname='splits/'+name+'.json')
            archive.add(OUT/'audit/input_review'/name,arcname='input_review/'+name)
        archive.add(OUT/'audit/data_inventory.json',arcname='data_inventory.json')
    (OUT/'m54_input_review.tar.gz.sha256').write_text(sha(OUT/'m54_input_review.tar.gz')+'  m54_input_review.tar.gz\n',encoding='utf-8')
    decision(prepared=True)
    print('M54 PREPARE PASS',counts,flush=True)


if __name__=='__main__':prepare()
