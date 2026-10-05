"""只用原caption/输入RGB修复donor资格，不改变冻结Primary/Conflict队列。"""
from pathlib import Path
import re
import numpy as np
from PIL import Image
from tools.e33tm_protocol import *

def strict_category(caption):
    result=category(caption)
    if result!='other': return result
    for name,words in [('bodysuit','bodysuit'),('bra','bra|bralette'),('poncho','poncho'),
                       ('swimsuit','swimsuit|bikini'),('vest','vest|waistcoat'),('jumpsuit','jumpsuit|romper')]:
        if re.search(r'\b('+words+r')\b',caption,re.I): return name
    return None

def prepare(cohort):
    path=OUT/'manifests/intervention_manifest.json'
    if path.exists():
        value=read(path)
        assert value['source_cohort_sha256']==sha(OUT/'manifests/cohorts.json')
        assert value['manifest_sha256']==sha(OUT/'manifests/intervention_data.json')
        return read(OUT/'manifests/intervention_data.json')
    split=read(RF/'split_manifest.json')
    captions={Path(r['cloth']).stem:r['caption'] for r in read('data/train_bf_texture.json')}
    overrides_path=OUT/'manifests/category_overrides.json'
    overrides=read(overrides_path) if overrides_path.exists() else {}
    source={r['id']:dict(r,caption=captions[r['id']],category=strict_category(captions[r['id']]),
                         pattern=pattern(captions[r['id']])) for group in ('train','dev') for r in split[group]}
    for sid,group in overrides.items(): source[sid]['category']=group
    by_category={}
    for r in source.values():
        if r['category'] is not None: by_category.setdefault(r['category'],[]).append(r)
    choices={};near={};used=set();unavailable=[]
    def pixels(row,key):
        with Image.open(DATASET/row[key]) as image:
            return np.asarray(image.convert('RGB').resize((32,32)),dtype=float)
    for row in cohort['dev']:
        r=source[row['id']]
        candidates=order([x for x in by_category.get(r['category'],[]) if x['id']!=r['id']],
                         'E33TM/strict_donor/'+r['id'])
        if not candidates:
            choices[r['id']]=None
            unavailable.append(dict(id=r['id'],reason='caption does not establish a garment category'))
        else:
            different=next((x for x in candidates if x['caption']!=r['caption']),None)
            original=pixels(r,'sketch')
            sk=max(candidates[:8],key=lambda x:float(np.mean(abs(pixels(x,'sketch')-original))))
            assert different
            choices[r['id']]=dict(shuffled_text=different['id'],wrong_sketch=sk['id'],wrong_texture=candidates[-1]['id'])
            used.update(choices[r['id']].values())
        compatible=order([x for x in candidates if r['pattern']!='unspecified' and x['pattern']==r['pattern']],
                         'E33TM/strict_near/'+r['id'])[:32]
        color=pixels(r,'reference').mean((0,1))
        distances=[(float(np.linalg.norm(pixels(x,'reference').mean((0,1))-color)),x['id']) for x in compatible]
        closest=min(distances) if distances else None
        available=bool(closest and closest[0]<=40)
        near[r['id']]=dict(donor=closest[1] if available else None,available=available,
                          rgb_mean_distance=closest[0] if closest else None,
                          color_threshold=40.,category=r['category'],pattern=r['pattern'],
                          reason=None if available else 'known same pattern/category and similar color not established')
        if available: used.add(closest[1])
    value=dict(donors=choices,text_compatible_near=near,donor_rows={sid:source[sid] for sid in used},
               unavailable_donor_cases=unavailable,primary_ids=[r['id'] for r in cohort['primary']],
               selection='input-only fixed hash pools; RGB color<=40; other is not treated as a garment category; unspecified is not treated as a pattern category')
    write(OUT/'manifests/intervention_data.json',value)
    write(path,dict(source_cohort_sha256=sha(OUT/'manifests/cohorts.json'),
                   manifest_sha256=sha(OUT/'manifests/intervention_data.json'),
                   frozen_before_any_donor_generations=True,original_primary_conflict_unchanged=True,
                   source_captions_sha256=sha('data/train_bf_texture.json'),git_commit=commit()))
    print('[E33TM donor audit]',len(unavailable),'unavailable donors;',sum(v['available'] for v in near.values()),'strict near cases',flush=True)
    return value
