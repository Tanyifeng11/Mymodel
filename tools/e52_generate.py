"""按身份分片，每个身份先生成共同 E5，再执行冻结实验臂。"""
import cv2
import numpy as np
from tools.e52_protocol import *
from tools.e52_runtime import runtime,release,case

def pooled_uv(uv,labels,size):
    valid=(labels>0).astype('float32');weight=cv2.resize(valid,size,interpolation=cv2.INTER_AREA)
    values=cv2.resize(np.maximum(uv,0)*valid[...,None],size,interpolation=cv2.INTER_AREA)
    values/=np.maximum(weight[...,None],1e-8)
    parts=cv2.resize(labels,size,interpolation=cv2.INTER_NEAREST)
    return values.reshape(-1,2),parts.ravel(),weight.ravel()>=.5

def mapping(row,mode,sigma,position_valid=None):
    data=np.load(row['geometry']);target,tl,tvalid=pooled_uv(data['target_uv'],data['target_labels'],(24,32))
    reference,rl,rvalid=pooled_uv(data['reference_uv0'],data['target_labels'],(16,16))
    if position_valid is not None:rvalid &= np.asarray(position_valid,bool)
    distance=((target[:,None]-reference[None])**2).sum(-1)
    allowed=(tl[:,None]==rl[None])&tvalid[:,None]&rvalid[None]
    matrix=np.exp(-distance/(2*sigma*sigma))*allowed
    matrix/=np.maximum(matrix.sum(-1,keepdims=True),1e-20)
    rng=np.random.default_rng(int(order(row['id'])[:8],16))
    if mode in ['WRONG','RANDOM']:
        for label in np.unique(tl[tvalid]):
            columns=np.where((rl==label)&rvalid)[0];queries=np.where((tl==label)&tvalid)[0]
            if not len(columns):continue
            permutation=np.roll(columns,max(1,len(columns)//3)) if mode=='WRONG' else rng.permutation(columns)
            matrix[np.ix_(queries,columns)]=matrix[np.ix_(queries,permutation)]
    import torch
    return torch.from_numpy(matrix.astype('float32'))

def run(stage,requested=None,shard=0,shards=1):
    assert read(OUT/'audit/p0_complete.json')['passed']
    rows=read(OUT/'splits'/(stage+'.json'))
    provenance=read(OUT/'audit/token_provenance.json') if (OUT/'audit/token_provenance.json').exists() else None
    if stage in ['pilot','lock']:
        arms=A_ARMS+C_ARMS+(B_ARMS if provenance and provenance['oracle_valid'] else [])
        if stage=='lock':assert read(OUT/'manifest/confirmation.json')['arms']==A_ARMS+(B_ARMS if provenance['oracle_valid'] else [])+C_ARMS
    elif stage=='natural':
        assert (OUT/'manifest/confirmation.json').exists();arms=CONFIG['natural_arms']
    else:arms=['E5','B1']
    if requested not in [None,'all']:arms=[requested]
    r=runtime(stage+'/runtime_shard%d'%shard)
    try:
        # 同身份/同参考/同 t 的 Writer 缓存，保留原 FP16 张量，不改变运算顺序。
        original_write=r.writer.write;cache={};r.writer_cache_key=None
        def cached(t):
            key=(r.writer_cache_key,int(t))
            if key not in cache:cache[key]=original_write(t).detach()
            return cache[key]
        r.writer.write=cached
        for index,row in enumerate(rows):
            if index%shards!=shard:continue
            cache.clear()
            for arm in arms:
                writer=None
                if stage in ['pilot','lock']:
                    key=arm if arm in row['writer_refs'] else 'G1T0' if arm in ['G1_EARLY','SHRINK_EARLY'] else 'G0T0'
                    writer=row['writer_refs'][key]
                r.writer_cache_key=sha(writer or row['reference'])
                teacher=image_path(stage,'E5',row['id']).with_suffix('.pt') if arm!='E5' else None
                oracle=mapping(row,arm,provenance['mapping_sigma'],provenance['position_valid']) if arm in B_ARMS else None
                case(r,row,arm,stage,writer,teacher,oracle,patch=stage=='probe' and index in [0,7] and arm=='B1')
            if stage=='pilot' and index==0 and provenance['oracle_valid'] and requested in [None,'all']:
                from tools.e47_runtime import exact_trace
                alpha_path=image_path(stage,'ORACLE_ALPHA0',row['id'])
                case(r,row,'ORACLE_ALPHA0',stage,row['writer_refs']['G0T0'],
                    image_path(stage,'E5',row['id']).with_suffix('.pt'),
                    mapping(row,'ORACLE',provenance['mapping_sigma'],provenance['position_valid']))
                original=image_path(stage,'G0T0',row['id'])
                check=exact_trace(alpha_path.with_suffix('.pt'),original.with_suffix('.pt'))
                assert check['bit_exact'] and sha(alpha_path)==sha(original),'ALPHA0_REPLAY_FAIL'
                write(OUT/'audit/alpha0_replay.json',dict(passed=True,trace=check,png_exact=True))
            print('E52 IDENTITY COMPLETE',stage,index,len(rows),flush=True)
    finally:release(r)
    write(OUT/'resources'/('%s_shard%d_complete.json'%(stage,shard)),dict(stage=stage,shard=shard,shards=shards,
        arms=arms,n=sum(i%shards==shard for i in range(len(rows))),commit=commit()))
