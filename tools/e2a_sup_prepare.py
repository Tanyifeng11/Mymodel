"""只使用生成前信息：核查 E1/E2a 清单、字节、近重复，冻结新 P2 与 LOCK。"""
import hashlib
import inspect
import re
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from tools.e2a_sup_common import *
from tools.e2a_attr_prepare import all_strings


def stable(value):
    return hashlib.sha256(('E2A_SUP_221042/'+str(value)).encode()).hexdigest()


def strata(row):
    text=row['caption'].lower()
    category=next((n for n,pattern in [('dress',r'\bdress\b'),('skirt',r'\bskirt\b'),
        ('trousers',r'\b(pants|trousers|jeans|shorts)\b'),('outerwear',r'\b(coat|jacket|blazer)\b'),
        ('top',r'\b(top|shirt|blouse|sweater|hoodie)\b')] if re.search(pattern,text)),'other')
    im=np.asarray(Image.open(row['reference']).convert('L').resize((64,64)),np.float32)/255.
    complexity=float((np.abs(np.diff(im,axis=0)).mean()+np.abs(np.diff(im,axis=1)).mean())/2)
    bright=float(im.mean())
    return dict(category=category, complexity=['low','medium','high'][int(complexity>.06)+int(complexity>.18)],
        brightness=['dark','medium','light'][int(bright>.35)+int(bright>.65)],
        occlusion='caption_marked' if re.search(r'\b(occluded|overlap|hidden|obscured)\b',text) else 'unknown',
        complexity_value=complexity, brightness_value=bright)


def prepare(workers=8):
    freeze(ROOT/'config.json',CONFIG)
    if (ROOT/'audit/preparation_complete.json').exists(): return
    assert read(OLD/'status.json')['complete']
    oldcfg=read(OLD/'training/seed42/F000/config.json')
    for name,digest in oldcfg['source_sha256'].items():
        assert sha(SOURCE/name)==digest,('HISTORICAL_CODE_CHANGED',name)
    freeze(ROOT/'audit/historical_source_sha256.json',oldcfg['source_sha256'])
    for arm,expected in EXPECTED.items():
        assert sha(model_path(arm))==expected,('CHECKPOINT_CHANGED',arm)
    freeze(ROOT/'checkpoint_registry.json',{a:dict(path=str(model_path(a)),sha256=h) for a,h in EXPECTED.items()})
    # 从各历史 manifest 解析真实清单，不能假定 E1 与 E2a 路径相同。
    histories={name:read(PROJECT/'output'/name/'experiment_manifest.json') for name in
        ['phase1_e1_grouped_e5','phase1_e2a_region_e5']}
    manifests={PROJECT/'data/train_bf_texture.json'}
    manifests.update(PROJECT/h['dataset_json_path'] for h in histories.values())
    trainroot=PROJECT.parent/'datasets/BF/training'
    records={str(p):read(p) for p in manifests}
    assert all(isinstance(rows,list) and len(rows)==45384 for rows in records.values())
    paths=sorted({str(trainroot/r[k]) for rows in records.values() for r in rows for k in ['cloth','sketch','texture']})
    train_ids={Path(r[k]).stem for rows in records.values() for r in rows for k in ['cloth','sketch','texture']}
    hold=PROJECT/'output_eval/e48_b1_structure_20261014/holdout_audit'
    complete=read(hold/'complete.json'); sigpath=Path(complete['training_signatures']['path'])
    if not sigpath.is_absolute(): sigpath=PROJECT/sigpath
    assert sha(sigpath)==complete['training_signatures']['sha256']
    signatures=read(sigpath)
    # 缓存之外的 E1 文件不能静默跳过：先停下来审查补全签名。
    assert set(paths)<=set(signatures), 'E1_ADDITIONAL_FILES_REQUIRE_SIGNATURE_AUDIT'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(p,digest) in enumerate(pool.map(lambda p:(p,sha(p)),paths),1):
            assert digest==signatures[p]['file_sha256'],('TRAIN_BYTES_CHANGED',p)
            if i%5000==0: print('TRAIN_HASH',i,len(paths),flush=True)
    exact={key:{signatures[p][key] for p in paths} for key in ['file_sha256','pixel_sha256']}
    refpaths=[p for p in paths if '/sketch/' not in p]
    trainph=np.array([list(bytes.fromhex(signatures[p]['phash'])) for p in refpaths],np.uint8)
    bits=np.array([bin(i).count('1') for i in range(256)],np.uint8)
    valpath=hold/'validation_signatures.json'
    assert sha(valpath)==complete['validation_signatures_sha256']
    vals=read(valpath); candidates=read(hold/'eligible.json'); identities={r['id'] for r in candidates}
    exposed=set(); exposure=[]
    # 保守排除既有实验 split（包括未打开的旧 LOCK），新协议重新分配。
    sources=set((PROJECT/'eval/benchmarks').glob('*.json'))
    sources.discard(PROJECT/'eval/benchmarks/bf_validation_resampler_full.json')
    for pattern in ['*/splits/*.json','*/split_sources.json','*/references.json']:
        sources.update(p for p in (PROJECT/'output_eval').glob(pattern) if ROOT not in p.parents)
    for p in sorted(sources):
        try: found={Path(v.replace('\\','/')).stem for v in all_strings(read(p))}&identities
        except (OSError,ValueError): continue
        exposed.update(found); exposure.append(dict(path=str(p),sha256=sha(p),ids=sorted(found)))
    bins=defaultdict(list)
    for row in candidates:
        if row['id'] in train_ids or row['id'] in exposed: continue
        info=strata(row); item=dict(row,stratum=info)
        bins[(info['category'],info['complexity'],info['brightness'],info['occlusion'])].append(item)
    queues=[deque(sorted(v,key=lambda r:stable(r['id']))) for k,v in sorted(bins.items(),key=lambda kv:stable(kv[0]))]
    ordered=[]
    while queues:
        for q in list(queues):
            ordered.append(q.popleft())
            if not q: queues.remove(q)
    chosen=[]; excluded=[]; pixels=set(); phashes=[]
    for row in ordered:
        sigs=[vals[row[k]] for k in ['gt','sketch','reference']]; reasons=[]
        if any(s[k] in exact[k] for s in sigs for k in exact): reasons.append('training_exact')
        if any(s['pixel_sha256'] in pixels for s in sigs): reasons.append('selected_exact')
        probes=[np.array(list(bytes.fromhex(vals[row[k]]['phash'])),np.uint8) for k in ['gt','reference']]
        for probe in probes:
            if int(bits[trainph^probe].sum(1).min())<=6: reasons.append('training_near')
            if phashes and int(bits[np.array(phashes)^probe].sum(1).min())<=6: reasons.append('selected_near')
        if reasons: excluded.append(dict(id=row['id'],reasons=sorted(set(reasons)))); continue
        for sig in sigs:
            assert sha(sig['path'])==sig['file_sha256'],('CANDIDATE_BYTES_CHANGED',sig['path'])
            pixels.add(sig['pixel_sha256'])
        phashes.extend(probes); chosen.append(row)
        if len(chosen)==256: break
    assert len(chosen)==256,('INDEPENDENT_POOL_SHORTAGE',len(chosen))
    # 交错分配，避免轮询顺序让某个 split 集中于最前的 strata。
    slots=sorted([('P2a',i) for i in range(32)]+[('P2b',i) for i in range(96)]+[('LOCK',i) for i in range(128)],
                 key=lambda v:stable('split/'+str(v)))
    split=defaultdict(list)
    for row,(name,_) in zip(chosen,slots): split[name].append(row)
    for name in ['P2a','P2b','LOCK']: freeze(ROOT/'splits'/(name+'.json'),split[name])
    swaps=[]
    for row in sorted(split['P2b'],key=lambda r:stable('swap/'+r['id'])):
        alternatives=[r for r in split['P2b'] if r['id']!=row['id'] and
            r['stratum']['category']==row['stratum']['category']]
        first=np.asarray(Image.open(row['reference']).convert('RGB').resize((32,32)),float)/255
        alternatives.sort(key=lambda r:stable('ref/'+row['id']+'/'+r['id']))
        for alternative in alternatives:
            second=np.asarray(Image.open(alternative['reference']).convert('RGB').resize((32,32)),float)/255
            difference=float(np.abs(first-second).mean())
            if difference<.05:continue
            swaps.append(dict(row,reference_alternative=alternative['reference'],alternative_id=alternative['id'],
                reference_category=alternative['stratum']['category'],input_rgb_difference=difference,
                references=[row['reference'],alternative['reference']],
                counterpart_GT_available=False,selection_used_generated_images=False))
            break
        if len(swaps)==32:break
    assert len(swaps)==32,'SAME_CATEGORY_REFERENCE_SWAP_POOL_SHORTAGE'
    freeze(ROOT/'splits/reference_swap.json',swaps)
    from PIL import ImageDraw
    panels=ROOT/'audit/swap_reference_review';panels.mkdir(parents=True,exist_ok=True)
    for start in range(0,32,8):
        canvas=Image.new('RGB',(3*128,8*195),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(swaps[start:start+8]):
            for k,key in enumerate(['sketch','reference','reference_alternative']):
                canvas.paste(Image.open(row[key]).convert('RGB').resize((128,171)),(k*128,j*195+24))
            draw.text((2,j*195+3),row['id']+' '+row['stratum']['category'],fill='black')
        canvas.save(panels/('page_%02d.png'%start))
    freeze(ROOT/'audit/exposure_registry.json',exposure)
    write(ROOT/'audit/excluded_candidates.json',excluded)
    freeze(ROOT/'audit/train_manifest_registry.json',{p:dict(sha256=sha(p),n=len(rows)) for p,rows in records.items()})
    freeze(ROOT/'audit/history_manifests.json',histories)
    freeze(ROOT/'audit/calibration_protocol.json',dict(effective_batches=64,samples=512,seed=42,
        sampler='historical EpochSampler and two-rank Accelerate loader from epoch0',
        no_optimizer_updates=True,select_coefficients_before_P2=True,
        pooling='median of positive per-sample epsilon-gradient RMS ratios; zeros and active rate also reported',
        split_ids={'TRAIN_CAL':'first 512 training samples in frozen distributed loader order'},
        pilot='800 updates from E1; formal restarts at E1/update0; separate resource ledger'))
    freeze(ROOT/'audit/reused_mask_audit.json',dict(masks_root=str(OLD/'masks/train'),
        mask_audit_sha256=sha(OLD/'mask_audit.csv'),visual=read(OLD/'audit/mask_visual_review.json'),
        known_defects_retained=17,mask_algorithm_changed=False))
    from tools.e2a_sup_check import check
    freeze(ROOT/'audit/operator_selfcheck.json',check())
    freeze(ROOT/'audit/preparation_complete.json',dict(passed=True,commit=commit(),training_files_verified=len(paths),
        train_manifest_identity_sets_equal=len({frozenset(Path(r['cloth']).stem for r in rows) for rows in records.values()})==1,
        splits={k:len(v) for k,v in split.items()},independent_from_declared_training=True,
        pretraining_exposure_unknown=True,occlusion_stratification='caption keyword proxy; otherwise unknown',
        category_source='caption regex',complexity_source='reference grayscale adjacent-difference proxy',
        byte_and_pixel_exact=True,near_exclusion='GT/reference phash Hamming<=6; train and cross-new-split',
        LOCK_generated_or_evaluated=False,reference_swap='PENDING_LEGAL_REFERENCE_REVIEW'))
    status(preparation='PASSED',stage='CALIBRATION_PENDING',engineering_failure=None)
    print('PREPARATION_PASSED P2a32 P2b96 LOCK128 unopened',flush=True)


if __name__=='__main__': prepare(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
