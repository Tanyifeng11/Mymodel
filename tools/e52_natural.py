"""刷新历史暴露审计；按预设类别/复杂度/纹样分层和 SHA 排序冻结新96。"""
import cv2
import numpy as np
from PIL import Image
from tools.e52_protocol import *

def category(caption):
    text=caption.lower()
    for name,words in [('pants',['pant','trouser','jeans','shorts']),('skirt',['skirt']),
        ('dress',['dress']),('outerwear',['coat','jacket','blazer']),('top',['shirt','blouse','top','sweater','vest'])]:
        if any(word in text for word in words):return name
    return 'other'

def pattern(caption):
    text=caption.lower()
    for name,words in [('stripe',['stripe','striped']),('check',['check','plaid']),
        ('floral',['floral','flower']),('graphic',['graphic','print','pattern']),('plain',['plain','solid'])]:
        if any(word in text for word in words):return name
    return 'unknown'

def select():
    from tools.e48_holdout import validation_rows,exposure_sources,phash_matrix
    from tools.e44_runtime import sketch_masks
    path=OUT/'splits/natural.json'
    if path.exists():return read(path)
    audit=SOURCE/'holdout_audit'
    prior=read(audit/'complete.json');assert prior['eligible']>=96
    previous={r['id']:r for r in read(audit/'eligible.json')}
    current,counts=validation_rows();lookup={r['id']:r for r in current}
    signatures=read(audit/'validation_signatures.json')
    exposed,registry=exposure_sources(set(lookup))
    write(OUT/'audit/natural_exposure_registry.json',registry)
    new_paths=sorted({lookup[i][key] for i in exposed if i in lookup for key in ['gt','sketch','reference']})
    assert all(p in signatures for p in new_paths),'新增身份未在旧完整签名池中'
    exact={key:{signatures[p][key] for p in new_paths} for key in ['file_sha256','pixel_sha256']}
    comparison=[p for p in new_paths if '/sketch/' not in p.replace('\\','/')]
    matrix=phash_matrix(comparison,signatures) if comparison else None
    pop=np.array([bin(i).count('1') for i in range(256)],np.uint8)
    eligible=[];excluded=[]
    for identity in sorted(set(previous)&set(lookup),key=order):
        row=lookup[identity];reasons=[]
        if identity in exposed:reasons.append('historical_exposure')
        for key in ['gt','sketch','reference']:
            signature=signatures[row[key]]
            if any(signature[k] in exact[k] for k in exact):reasons.append(key+'_new_exact_duplicate')
            if key!='sketch' and matrix is not None:
                h=np.array(list(bytes.fromhex(signature['phash'])),np.uint8)
                if int(pop[matrix^h].sum(1).min())<=6:reasons.append(key+'_new_near_candidate')
        if not reasons:
            sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            _,_,info=sketch_masks(sketch,'cpu')
            if not info['valid']:reasons.append('objective_invalid_sketch')
        if reasons:excluded.append(dict(id=identity,reasons=reasons));continue
        gray=np.asarray(sketch.convert('L'));edge=float((gray<128).mean())
        eligible.append(dict(row,target_category=category(row['caption']),pattern_type=pattern(row['caption']),complexity=edge))
    assert len(eligible)>=96,('NEW96_INSUFFICIENT',len(eligible))
    thresholds=np.quantile([r['complexity'] for r in eligible],[1/3,2/3])
    strata={}
    for row in eligible:
        row['complexity_bin']=int(np.searchsorted(thresholds,row['complexity']))
        key=(row['target_category'],row['pattern_type'],row['complexity_bin'])
        strata.setdefault(key,[]).append(row)
    exact_quota={key:len(values)*96/len(eligible) for key,values in strata.items()}
    quotas={key:int(value) for key,value in exact_quota.items()}
    remaining=96-sum(quotas.values())
    for key in sorted(strata,key=lambda k:(-(exact_quota[k]-quotas[k]),str(k)))[:remaining]:quotas[key]+=1
    chosen=[]
    for key,values in sorted(strata.items()):chosen.extend(sorted(values,key=lambda r:order(r['id']))[:quotas[key]])
    chosen=sorted(chosen,key=lambda r:order(r['id']));assert len(chosen)==96
    frozen={r[key]:sha(r[key]) for r in chosen for key in ['gt','sketch','reference']}
    assert all(frozen[p]==signatures[p]['file_sha256'] for p in frozen),'历史签名后输入发生变化'
    freeze(path,chosen);freeze(OUT/'manifest/natural_input_sha256.json',frozen)
    write(OUT/'audit/natural_excluded.json',excluded)
    freeze(OUT/'audit/natural_selection.json',dict(n=96,current_complete_intersection=len(current),modalities=counts,
        prior_eligible=len(previous),current_eligible=len(eligible),exposed_ids=len(exposed),
        previous_audit_sha256=sha(audit/'complete.json'),signatures_sha256=sha(audit/'validation_signatures.json'),
        exposure_registry_sha256=sha(OUT/'audit/natural_exposure_registry.json'),
        strata=[dict(category=k[0],pattern=k[1],complexity=k[2],population=len(strata[k]),quota=quotas[k]) for k in sorted(strata)],
        complexity_thresholds=thresholds.tolist(),fixed_sha_seed=CONFIG['natural_selection_seed'],
        prior_training_and_near_duplicate_exclusions_inherited=True,
        new_development_exposure_and_crossmodality_near_duplicates_rechecked=True,
        pattern_category_source='caption keyword strata; not manually verified texture labels',
        writer_pretraining_exposure='unknown',missing_historical_consumption_logs='unknown',strict_unseen=False,
        held_out_after_freeze=True,model_outputs_used_for_selection=False))
    return chosen
