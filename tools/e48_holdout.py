"""先审计历史暴露与重复；通过开发门槛后才选择新96，不查看生成效果选身份。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from data.prepare_texture_json import index_dir,IMAGE_EXTS
from tools.e47_audit import normalize,image_signature
from tools.e48_protocol import *

ROOT=Path('/share/home/u2515283058/datasets/BF/validation')
KINDS=['gt','sketch','reference']


def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)


def validation_rows():
    indexes={k:index_dir(ROOT/v,IMAGE_EXTS if k!='caption' else {'.txt'})
        for k,v in [('gt','gt'),('sketch','sketch'),('reference','texture'),('caption','text')]}
    identities=set.intersection(*(set(v) for v in indexes.values()))
    rows=[dict(id=i,**{k:str(indexes[k][i]) for k in KINDS},
        caption=indexes['caption'][i].read_text(encoding='utf-8').strip()) for i in sorted(identities)]
    return rows,{k:len(v) for k,v in indexes.items()}


def exposure_sources(identities):
    # 全量 resampler_full 只枚举数据，消费证据另查实际 benchmark_samples。
    sources=set(Path('eval/benchmarks').glob('*.json'))
    sources.discard(Path('eval/benchmarks/bf_validation_resampler_full.json'))
    sources.update(Path('eval_outputs').glob('**/benchmark_samples.json'))
    for pattern in ['*/splits/*.json','*/split_sources.json','*/references.json']:
        sources.update(p for p in Path('output_eval').glob(pattern) if OUT not in p.parents)
    extra=Path('output_eval/e18_1/clean/validation.json')
    if extra.exists():sources.add(extra)
    exposed=set();registry=[]
    for p in sorted(sources):
        content=read(p)
        if p.name=='benchmark_samples.json':
            content=[r for r in content if r.get('generation_status')=='generated'
                or bool(r.get('gen_path') and Path(r['gen_path']).exists())]
        found={Path(v.replace('\\','/')).stem for v in strings(content)}&identities
        exposed.update(found)
        registry.append(dict(path=str(p),sha256=sha(p),validation_ids=sorted(found)))
    return exposed,registry


def signature(path):
    result=image_signature(path);result.pop('thumbnail');return result


def phash_matrix(paths,cache):
    return np.asarray([list(bytes.fromhex(cache[p]['phash'])) for p in paths],np.uint8)


def audit(workers):
    folder=OUT/'holdout_audit'
    assert not (folder/'complete.json').exists(),'已有完整审计，不重复选择确认身份'
    rows,counts=validation_rows();identities={r['id'] for r in rows}
    train=[normalize(r) for r in read('data/train_bf_texture.json')]
    overlap=identities&{r['id'] for r in train}
    exposed,registry=exposure_sources(identities)
    write(folder/'exposure_registry.json',registry)
    old_path=PARENT/'manifests/file_signatures.json';cache=read(old_path)
    historical_paths={r[k] for r in train for k in KINDS if r[k]}
    assert historical_paths<=set(cache),'历史训练签名缺失'
    paths=sorted({r[k] for r in rows for k in KINDS})
    signatures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,s in enumerate(pool.map(signature,paths)):
            signatures[s['path']]=s
            if i%1000==0:print('E48 HOLDOUT SIGNATURE',i,len(paths),flush=True)
    cache.update(signatures);write(folder/'validation_signatures.json',signatures)
    old_rows=[r for r in rows if r['id'] in exposed]
    historical_paths.update(r[k] for r in old_rows for k in KINDS)
    exact={key:{cache[p][key] for p in historical_paths} for key in ['file_sha256','pixel_sha256']}
    reference_paths=sorted({r[k] for r in train+old_rows for k in ['gt','reference'] if r[k]})
    matrix=phash_matrix(reference_paths,cache)
    lookup=np.asarray([bin(i).count('1') for i in range(256)],np.uint8)
    excluded=[];eligible=[]
    for n,row in enumerate(rows):
        identity=row['id'];reasons=[];near=[]
        if identity in overlap:reasons.append('training_identity')
        if identity in exposed:reasons.append('historical_development_identity')
        if not reasons:
            for kind in KINDS:
                s=cache[row[kind]]
                for key in exact:
                    if s[key] in exact[key]:reasons.append(kind+'_'+key)
            # 跨 GT/参考模态筛查；pHash 命中仅作保守排除，不声称同源已确认。
            for kind in ['gt','reference']:
                probe=np.asarray(list(bytes.fromhex(cache[row[kind]]['phash'])),np.uint8)
                distances=lookup[matrix^probe].sum(1);index=int(distances.argmin())
                if distances[index]<=6:
                    reasons.append(kind+'_near_candidate')
                    near.append(dict(kind=kind,other=reference_paths[index],distance=int(distances[index])))
        if reasons:excluded.append(dict(id=identity,reasons=reasons,near=near))
        else:eligible.append(row)
        if n%500==0:print('E48 HOLDOUT AUDIT',n,len(rows),'eligible',len(eligible),flush=True)
    write(folder/'excluded.json',excluded);write(folder/'eligible.json',eligible)
    value=dict(complete=True,total=len(rows),modality_counts=counts,eligible=len(eligible),
        training_identity_overlap=len(overlap),historical_development_ids=len(exposed),
        excluded=len(excluded),selection='not_selected; freeze one candidate before fixed seed48096 selection',
        exact='file and RGB pixel SHA256, all three modalities cross-compared',
        near='64bit pHash Hamming<=6 across GT/reference; conservative exclusion, not source identity proof',
        training_signatures=dict(path=str(old_path),sha256=sha(old_path),reused=True),
        input_sha256=sha('data/train_bf_texture.json'),registry_sha256=sha(folder/'exposure_registry.json'),
        validation_signatures_sha256=sha(folder/'validation_signatures.json'),
        exposure_limit='audited accessible historical benchmarks and split/reference registries; missing consumption logs remain unknown',
        writer_pretraining_overlap=None,strict_unseen_generalization=False,commit=commit())
    write(folder/'complete.json',value);print('E48 HOLDOUT COMPLETE',value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();audit(args.workers)
