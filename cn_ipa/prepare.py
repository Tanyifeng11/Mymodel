"""只使用冻结历史身份及原始参考文件，先审计后导出非方形训练数据。"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import platform
import numpy as np
import cv2
from PIL import Image
from cn_ipa.common import *


def signature(path):
    image=Image.open(path).convert('RGB')
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),np.float32)
    dct=cv2.dct(gray)[:8,:8].copy(); bits=(dct>np.median(dct.ravel()[1:])).ravel(); bits[0]=False
    return dict(file_sha256=sha(path),rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        phash=np.packbits(bits).tobytes().hex(),size=list(image.size))


def prepare():
    train=read(M54/'splits/train1024.json')
    dev=read(CONFIRM.parent/'m12_gain_dev32_20261020/protocol/dev32.json')
    test=read(CONFIRM/'protocol/identities.json')
    assert len(train)==1024 and len(dev)==32 and len(test)==96
    for name,rows in [('train',train),('dev',dev),('test_A',test)]:
        assert len({r['id'] for r in rows})==len(rows),name
        for row in rows:
            assert all(row.get(k) for k in ['gt','sketch','reference','caption'])
            assert Path(row['reference']).resolve()!=Path(row['gt']).resolve()
            assert '/texture/' in row['reference'].replace('\\','/')
    paths=sorted({r[k] for rows in [train,dev,test] for r in rows for k in ['gt','sketch','reference']})
    with ThreadPoolExecutor(max_workers=4) as pool: signatures=dict(zip(paths,pool.map(signature,paths)))
    write(ROOT/'manifests/source_hashes.json',signatures)
    # 冻结 TEST-A，不以新模型结果改变任何身份。先核验历史文件哈希。
    old={r['id']:r for r in read(CONFIRM/'protocol/input_manifest.json')}
    for row in test:
        for key in ['gt','sketch','reference']:
            assert signatures[row[key]]['file_sha256']==old[row['id']][key+'_sha256']
    lookup=np.array([bin(i).count('1') for i in range(256)],np.uint8)

    def cross(row,others):
        causes=[]
        for other in others:
            if row['id']==other['id']:causes.append(dict(other=other['id'],reason='identity'))
        other_paths={r[k] for r in others for k in ['gt','sketch','reference']}
        hashes={signatures[p][k] for p in other_paths for k in ['file_sha256','rgb_sha256']}
        for kind in ['gt','sketch','reference']:
            a=signatures[row[kind]]
            if a['file_sha256'] in hashes or a['rgb_sha256'] in hashes:
                causes.append(dict(kind=kind,reason='exact_cross_split'))
        photo_paths=sorted({r[k] for r in others for k in ['gt','reference']})
        matrix=np.array([list(bytes.fromhex(signatures[p]['phash'])) for p in photo_paths],np.uint8)
        for kind in ['gt','reference']:
            a=np.array(list(bytes.fromhex(signatures[row[kind]]['phash'])),np.uint8)
            distances=lookup[matrix^a].sum(1)
            if distances.min()<=6:
                i=int(distances.argmin()); causes.append(dict(kind=kind,reason='phash_cross_split',
                    distance=int(distances[i]),other_path=photo_paths[i]))
        return causes

    dev_conflicts=[dict(id=r['id'],causes=cross(r,test)) for r in dev if cross(r,test)]
    assert not dev_conflicts,('FROZEN_DEV_TEST_CONFLICT',dev_conflicts)
    excluded=[];accepted=[];seen_gt=set()
    for row in train:
        causes=cross(row,dev+test)
        caption=row['caption'].lower()
        if any(w in caption for w in ['backpack','briefcase','handbag','crossbody bag','tote bag','sneaker','sandal']):
            causes.append(dict(reason='obvious_non_garment'))
        gt=signatures[row['gt']]['rgb_sha256']
        if gt in seen_gt:causes.append(dict(reason='duplicate_training_GT'))
        if causes:excluded.append(dict(id=row['id'],causes=causes))
        else:accepted.append(row);seen_gt.add(gt)
    assert accepted
    manifests={}
    for name,rows in [('train',accepted),('dev',dev),('test_A',test)]:
        for row in rows:
            row['fabric_id']=signatures[row['reference']]['rgb_sha256']
            row['split']=name
        path=ROOT/'manifests'/f'{name}.json';write(path,rows);manifests[name]=sha(path)
        with path.with_suffix('.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(ROOT/'manifests/excluded_train.json',excluded)
    write(ROOT/'manifests/qualitative_ids.json',[r['id'] for r in sorted(test,
        key=lambda r:hashlib.sha256(('CNIPA_QUAL_20261022_'+r['id']).encode()).hexdigest())[:12]])
    write(ROOT/'manifests/split_audit.json',dict(passed=True,original_train=1024,train=len(accepted),dev=32,test_A=96,
        excluded_train=len(excluded),manifest_sha256=manifests,reference_groups=dict(Counter(r['fabric_id'] for r in accepted)),
        identity_cross_split=0,exact_cross_split=0,GT_reference_phash_cross_split_distance_min=7,
        sketch_phash='线稿只做精确/像素排重；pHash对共享空白和轮廓过于敏感，不作为近重身份判据',
        public_pretraining_exposure='UNKNOWN',writer_pretraining_exposure='UNKNOWN',
        original_reference_acquisition='数据集提供的单独texture文件；不声称已证明与GT在采集源上独立',
        geometry=CONFIG['train_geometry'],historical_test_input_hashes_verified=True))
    folder=ROOT/'cn_imagefolder/train';folder.mkdir(parents=True,exist_ok=True)
    # datasets3.0.1仅识别file_name为主图；线稿放在ImageFolder扫描范围外，显式cast为第二图像列。
    sketch_folder=ROOT/'preprocessed_sketch'
    if (folder/'sketch').exists() and not sketch_folder.exists():(folder/'sketch').rename(sketch_folder)
    sketch_folder.mkdir(exist_ok=True)
    metadata=[]
    for row in accepted:
        for key,sub in [('gt','gt'),('sketch','sketch')]:
            path=(folder/'gt' if key=='gt' else sketch_folder)/(row['id']+'.png');path.parent.mkdir(exist_ok=True)
            Image.open(row[key]).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR).save(path)
        metadata.append(dict(file_name=f"gt/{row['id']}.png",
            conditioning_image=str(sketch_folder/(row['id']+'.png')),text=row['caption']))
    with (folder/'metadata.jsonl').open('w',encoding='utf-8') as f:
        for row in metadata:f.write(json.dumps(row)+'\n')
    from datasets import load_dataset, Image as DatasetImage
    dataset=load_dataset('imagefolder',data_dir=str(ROOT/'cn_imagefolder'),split='train')
    dataset=dataset.cast_column('conditioning_image',DatasetImage())
    sample=dataset[0]
    assert isinstance(sample['image'],Image.Image) and isinstance(sample['conditioning_image'],Image.Image)
    assert sample['image'].size==sample['conditioning_image'].size==SIZE
    write(ROOT/'audit/imagefolder_smoke.json',dict(columns=dataset.column_names,size=list(sample['image'].size),n=len(dataset)))
    write(ROOT/'audit/protocol.json',dict(config=CONFIG,prepare_commit=commit(),
        historical_confirm_commit=read(CONFIRM/'protocol/protocol.json')['commit'],
        python=platform.python_version(),source_hashes_sha256=sha(ROOT/'manifests/source_hashes.json')))
    print('CNIPA_PREPARED',len(accepted),len(dev),len(test),flush=True)


if __name__=='__main__':prepare()
