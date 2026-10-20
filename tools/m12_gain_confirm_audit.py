"""在生成前更新可访问历史暴露、重复与服装资格；只按输入固定排序。"""
import re
import shutil
from collections import Counter
import numpy as np
from PIL import Image
from tools.m12_gain_confirm_protocol import *
from tools.m12_protocol import write_csv


def strings(value):
    if isinstance(value,str):
        yield value
    elif isinstance(value,dict):
        for key,item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value,list):
        for item in value:yield from strings(item)


def exposures(identities):
    sources=set(Path('eval/benchmarks').glob('*.json'))
    sources.discard(Path('eval/benchmarks/bf_validation_resampler_full.json'))
    sources.update(Path('eval_outputs').glob('**/benchmark_samples.json'))
    patterns=['**/splits/*.json','**/split_sources.json','**/references.json','**/active_inputs.json',
        '**/protocol/dev32.json','**/protocol/identities.json','**/per_image.json',
        '**/audit/**/candidates.json','**/audit/**/reviewed*.json']
    for pattern in patterns:sources.update(Path('output_eval').glob(pattern))
    sources={p for p in sources if RUN.resolve() not in p.resolve().parents}
    extra=Path('output_eval/e18_1/clean/validation.json')
    if extra.exists():sources.add(extra)
    registry=[];exposed=set()
    for path in sorted(sources):
        content=read(path)
        if path.name=='benchmark_samples.json':
            content=[r for r in content if r.get('generation_status')=='generated' or bool(r.get('gen_path') and Path(r['gen_path']).exists())]
        found=set()
        for value in strings(content):
            stem=Path(value.replace('\\','/')).stem
            if stem in identities:found.add(stem)
            # 兼容 identity_seed42 的历史文件名，避免漏掉已实际评分身份。
            match=re.match(r'^(\d+)[_-](?:seed)?\d+',stem)
            if match and match[1] in identities:found.add(match[1])
        exposed.update(found)
        registry.append(dict(path=str(path.resolve()),sha256=sha(path),identities=sorted(found)))
    return exposed,registry


def run_audit():
    from tools.e47_audit import normalize
    from tools.m54_input_candidates import clothing
    from tools.e44_runtime import sketch_masks
    history=PROJECT/'output_eval/e48_b1_structure_20261014/holdout_audit'
    prior=read(history/'eligible.json')
    signatures=read(history/'validation_signatures.json')
    signature_source=PROJECT/'output_eval/e47_b1_reference_20261014/manifests/file_signatures.json'
    # 使用原审计记录中的真实来源，不推测历史目录名称。
    signature_source=Path(read(history/'complete.json')['training_signatures']['path'])
    cache=read(signature_source);cache.update(signatures)
    all_validation={Path(p).stem for p in signatures}
    exposed,registry=exposures(all_validation)
    train=[normalize(r) for r in read('data/train_bf_texture.json')]
    exposed.update(r['id'] for r in train)
    # 显式纳入本轮前两次实验与 M54 训练/dev；不依赖通用 glob 恰好命中。
    explicit=[M12/'splits/pilot.json',M12/'splits/calibration.json',M12/'splits/dev32.json',
        OLD/'protocol/dev32.json',M54/'splits/train1024.json',M54/'splits/dev32.json']
    for path in explicit:
        found={r['id'] for r in read(path)}
        exposed.update(found)
        registry.append(dict(path=str(path),sha256=sha(path),identities=sorted(found),explicit=True))
    historical_paths={r[k] for r in train for k in ['gt','sketch','reference'] if r[k]}
    historical_paths.update(p for p in signatures if Path(p).stem in exposed)
    assert historical_paths<=set(cache),'HISTORICAL_SIGNATURE_MISSING'
    exact={key:{cache[p][key] for p in historical_paths} for key in ['file_sha256','pixel_sha256']}
    reference_paths=sorted(p for p in historical_paths if '/sketch/' not in p.replace('\\','/'))
    matrix=np.asarray([list(bytes.fromhex(cache[p]['phash'])) for p in reference_paths],np.uint8)
    lookup=np.asarray([bin(i).count('1') for i in range(256)],np.uint8)
    rejected=[];eligible=[];selected=[];selected_signatures=[];selected_hashes=set()
    for index,row in enumerate(sorted(prior,key=lambda r:identity_order(r['id']))):
        reasons=[];near=[]
        if row['id'] in exposed:reasons.append('known_training_development_or_candidate_exposure')
        if not clothing(row) or non_clothing_caption(row['caption']):reasons.append('original_clothing_ineligible')
        if not reasons:
            for kind in ['gt','sketch','reference']:
                signature=cache[row[kind]]
                for key in exact:
                    if signature[key] in exact[key]:reasons.append(kind+'_'+key+'_historical_duplicate')
            for kind in ['gt','reference']:
                probe=np.asarray(list(bytes.fromhex(cache[row[kind]]['phash'])),np.uint8)
                distances=lookup[matrix^probe].sum(1)
                if len(distances) and distances.min()<=6:
                    hit=int(distances.argmin());reasons.append(kind+'_historical_near_duplicate')
                    near.append(dict(kind=kind,other=reference_paths[hit],distance=int(distances[hit])))
        if reasons:
            rejected.append(dict(id=row['id'],reasons=sorted(set(reasons)),near=near))
        else:
            eligible.append(row)
            if len(selected)<96:
                hashes={cache[row[k]][key] for k in ['gt','sketch','reference'] for key in ['file_sha256','pixel_sha256']}
                duplicate=bool(hashes&selected_hashes)
                phashes=[int(cache[row[k]]['phash'],16) for k in ['gt','reference']]
                near_selected=any(bin(a^b).count('1')<=6 for a in phashes for b in selected_signatures)
                if duplicate or near_selected:
                    rejected.append(dict(id=row['id'],reasons=['confirmation_within_cohort_duplicate'],near=[]))
                else:
                    selected.append(row);selected_hashes.update(hashes);selected_signatures.extend(phashes)
        if index%1000==0:print('CONFIRM_AUDIT',index,len(eligible),len(selected),flush=True)
    folder=RUN/'protocol'
    write(folder/'exposure_registry.json',registry)
    write(folder/'excluded_identities.json',rejected)
    write(folder/'eligible_identities.json',eligible)
    write(folder/'split_audit.json',dict(original_eligible=len(prior),updated_eligible=len(eligible),selected=len(selected),
        exposed_validation_ids=len(exposed&all_validation),source_count=len(registry),
        exclusion_counts=dict(Counter(reason for r in rejected for reason in r['reasons'])),
        exact='all three modalities file/RGB hashes against known history and within selected cohort',
        near='GT/reference cross modality pHash64 Hamming<=6; conservative exclusion',
        input_candidate_policy='exclude rendered candidate lists; exhaustive automatic input scores/signatures/enumerations are qualification data, not model-output development consumption',
        historical_audit_sha256=sha(history/'complete.json'),training_signatures_sha256=sha(signature_source),
        validation_signatures_sha256=sha(history/'validation_signatures.json'),
        cached_signatures='reused historical immutable audit; selected inputs file hashes recomputed now',
        writer_pretraining_overlap='UNKNOWN',backbone_full_history='NOT_FULLY_RECOVERABLE',strict_pretraining_unseen=False,
        ordering='SHA256 M12_GAIN_CONFIRM_20261020_96042_ + identity, independent of generated outputs'))
    assert len(selected)==96,'INSUFFICIENT_NEW_IDENTITIES_STOP_BEFORE_GENERATION'
    manifest=[]
    for row in selected:
        record=dict(id=row['id'],caption=row['caption'],caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),
            selection='fixed hash order; eligible after historical exposure and exact/near audit')
        for kind in ['gt','sketch','reference']:
            source=Path(row[kind]);assert sha(source)==cache[str(source)]['file_sha256']
            record[kind+'_sha256']=sha(source)
            destination=RUN/'inputs'/row['id']/(kind+'_canonical.png');destination.parent.mkdir(parents=True,exist_ok=True)
            decoded=Image.open(source).convert('RGB');decoded.save(destination)
            record[kind+'_rgb_sha256']=hashlib.sha256(decoded.tobytes()).hexdigest()
        _,_,gate=sketch_masks(Image.open(row['sketch']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR),'cpu')
        record.update(inference_gate_valid=gate['valid'],gate=gate)
        manifest.append(record)
    write(folder/'identities.json',selected)
    write(folder/'input_manifest.json',manifest)
    write_csv(folder/'identity_manifest.csv',[{k:v for k,v in r.items() if k!='gate'} for r in manifest])
    return selected
