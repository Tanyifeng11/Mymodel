"""官方验证数据的输入审阅与冻结；不生成模型输出，不用GT生成几何条件。"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_geometry import InteriorPatchSampler, build_geometry
from tools.e2_followup_data import rgb, color_summary, color_distance, spectral_proxy, neutral_prompt
from tools.e2_fft_experiment import SOURCE, read, write, sha

SIZE = (384, 512)
PROTOCOL = dict(version='e2_b_screen_v3_input_frozen', seed=110820,
    data='official_validation; target identity/pixel/near-duplicate exclusions against existing training audit',
    candidate_labels='caption-derived only; AI input review required for clean pattern eligibility',
    same_color=dict(mean_lab_delta_max=5., quantile_delta_max=10.),
    patch=dict(size=64, minimum=4, maximum=8, source='input sketch only', seed='110803+index'),
    scales=[.75, 1., 1.33], distance='mean of all cross-set DISTS patch distances; no nearest-patch matching',
    primary='gray_strict_response_change', primary_holm_family='four alternatives vs B0',
    secondary_holm_family='other three texture response metrics x four alternatives',
    paired_statistics='10000 identity bootstrap/signflip; seed110142',
    training='same mature E2, updates2000, effective_batch8; single screening seed',
    upstream_pretraining_exposure='unknown; current E2 training overlap checked, no claim of unseen upstream pretraining',
    selection_uses_generated_outputs=False, human_blind_reviewers=0)


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def render_pages(out,data,selected,start=0):
    pages=[]
    for page in range((len(selected)+15)//16):
        panel=Image.new('RGB',(1536,880),'white');draw=ImageDraw.Draw(panel)
        group=selected[page*16:(page+1)*16]
        for slot,row in enumerate(group):
            x,y=slot%4*384,slot//4*220
            draw.text((x+3,y+3),'%s %s n=%s'%(row['index'],row['pattern_candidate'],len(row['patch_coordinates'])),fill='black')
            for col,key in enumerate(['sketch','gt','texture']):panel.paste(rgb(data/row[key],(128,184)),(x+128*col,y+22))
        name='inputs/review_%02d.png'%(start+page+1)
        panel.save(out/name);pages.append(dict(file=name,indices=[r['index'] for r in group]))
    return pages


def prepare(args):
    from eval.eval_utils import prepare_evaluation_masks
    out, data = Path(args.out), Path(args.data)
    source = read(args.candidates)
    rows = []
    torch.set_num_threads(2)
    for i, row in enumerate(source['records']):
        hashes = {k: sha(data / row[k]) for k in ['sketch', 'gt', 'texture']}
        assert hashes['gt']==row['audited_gt_file_sha256'], '既有去重审计的GT字节已改变: %s'%row['sample_id']
        reference = rgb(data / row['texture'], SIZE)
        mask = prepare_evaluation_masks(SIZE, sketch_path=str(data / row['sketch']), mask_policy='sketch_only')['garment']
        geometry = build_geometry(torch.from_numpy(mask.astype(np.float32))[None, None], details=False)
        coordinates, number = InteriorPatchSampler(110803 + row['index']).coordinates(geometry.body[0], 64, 8)
        rows.append(dict(row, hashes=hashes, color=color_summary(reference),
            spectral_proxy=spectral_proxy(np.asarray(reference.resize((64,64), Image.Resampling.BICUBIC))),
            patch_coordinates=coordinates, patch_candidates=number,
            mask_valid=bool(geometry.valid[0]), mask_fraction=float(mask.mean()),
            neutral_prompt=neutral_prompt(row['caption'])))
        if i % 50 == 0:
            print('B_INPUT', i, len(source['records']), flush=True)
    registry = dict(protocol=PROTOCOL, candidates_sha256=sha(args.candidates), records=rows,
        input_map_sha256=canonical({str(r['index']): r['hashes'] for r in rows}))
    write(out / 'inputs/registry.json', registry)
    # 初轮每类固定128/64/64；如资格不足，可在尚未生成任何输出时审阅剩余候选。
    initial = []
    for pattern, limit in [('stripes',128), ('checks',64), ('dots',64)]:
        group = [r for r in rows if r['pattern_candidate'] == pattern]
        initial += sorted(group, key=lambda r: canonical(['b_input_review', r['sample_id']]))[:limit]
    selected = initial if args.review_all is False else sorted(rows, key=lambda r: canonical(['b_input_review',r['sample_id']]))
    if Path(args.review).exists() and not args.review_all:
        labels=read(args.review)['records'];by_index={r['index']:r for r in rows}
        selected=[by_index[int(index)] for index in labels]
        mismatches={str(row['index']):dict(expected=labels[str(row['index'])]['hashes'],actual=row['hashes'])
            for row in selected if labels[str(row['index'])]['hashes']!=row['hashes']}
        write(out/'audit/input_version_comparison.json',dict(passed=not mismatches,reviewed_records=len(labels),
            mismatch_records=len(mismatches),mismatches=mismatches))
    pages = render_pages(out,data,selected)
    write(out/'inputs/review_pages.json',dict(pages=pages,records_review_requested=len(selected),
        registry_sha256=sha(out/'inputs/registry.json'),input_map_sha256=registry['input_map_sha256']))
    print('B_INPUTS_READY',len(rows),len(selected),len(pages),registry['input_map_sha256'],flush=True)


def extend_review(args):
    out,data=Path(args.out),Path(args.data)
    assert not(out/'strict_pairs.json').exists() and not(out/'outputs').exists()
    registry=read(out/'inputs/registry.json');labels=read(args.review)['records'];rows=registry['records']
    known=[r for r in rows if labels.get(str(r['index']),{}).get('clean')]
    candidates=[]
    for row in rows:
        if str(row['index']) in labels:continue
        tentative=dict(labels,**{str(row['index']):dict(clean=True,pattern=row['pattern_candidate'])})
        if any(other['identity_group']!=row['identity_group'] and qualifies(row,other,tentative) is not None
               for other in known):candidates.append(row)
    selected=sorted(candidates,key=lambda r:canonical(['b_extension_input_only',r['sample_id']]))[:64]
    table=read(out/'inputs/review_pages.json')
    pages=render_pages(out,data,selected,start=len(table['pages']))
    table['pages']+=pages;table['records_review_requested']+=len(selected)
    table['registry_sha256']=sha(out/'inputs/registry.json')
    table.setdefault('extension_rule','unreviewed caption candidate with numeric color/period eligibility against already-reviewed clean reference; stable SHA order; no generated output')
    write(out/'inputs/review_pages.json',table)
    print('EXTEND_REVIEW',len(selected),'PAGES',[p['file'] for p in pages],flush=True)


def qualifies(a,b,review):
    qa,qb=review.get(str(a['index']),{}),review.get(str(b['index']),{})
    if not (qa.get('clean') and qb.get('clean')):
        return None
    if qa.get('pattern') not in ['stripes','checks','dots'] or qb.get('pattern') not in ['stripes','checks','dots']:
        return None
    delta,palette=color_distance(a['color'],b['color'])
    if delta>5 or palette>10:
        return None
    pa,pb=a['spectral_proxy'],b['spectral_proxy']
    if not(pa['valid'] and pb['valid']):
        return None
    ratio=max(pa['period'],pb['period'])/min(pa['period'],pb['period'])
    angle=abs(pa['line_angle']-pb['line_angle']);angle=min(angle,180-angle)
    family=qa['pattern']!=qb['pattern']
    if not family and ratio<1.5 and not(qa['pattern']=='stripes' and angle>=45):
        return None
    return dict(color_delta=delta,palette_delta=palette,period_ratio=ratio,
        orientation_difference=angle,pattern_family_change=family,
        old_pattern=qa['pattern'],new_pattern=qb['pattern'])


def freeze_pairs(args):
    out=Path(args.out)
    assert not (out/'outputs').exists(), '生成输出后禁止再选择配对'
    registry=read(out/'inputs/registry.json');review=read(args.review)
    rows=registry['records'];labels=review['records']
    # 资格必须绑定实际运行环境的原始输入哈希，不把本地与服务器版本差异静默放过。
    for row in rows:
        label=labels.get(str(row['index']))
        if label is not None:
            assert label['hashes']==row['hashes'], '需重新审阅实际输入: %s'%row['index']
    used=set();used_groups=set();pairs=[]
    ordered=sorted(rows,key=lambda r:canonical(['b_screen_pairs',r['sample_id']]))
    for target in ordered:
        if (target['index'] in used or target['identity_group'] in used_groups
                or not target['mask_valid'] or len(target['patch_coordinates'])<4
                or not labels.get(str(target['index']),{}).get('target_eligible')):
            continue
        candidates=[]
        for reference in ordered:
            if (reference['index'] in used or reference['index']==target['index']
                    or reference['identity_group'] in used_groups or reference['identity_group']==target['identity_group']):
                continue
            score=qualifies(target,reference,labels)
            if score is not None:
                candidates.append((score['color_delta'],canonical([target['index'],reference['index']]),reference,score))
        if not candidates:
            continue
        _,_,reference,score=min(candidates,key=lambda x:x[:2])
        used.update([target['index'],reference['index']])
        used_groups.update([target['identity_group'],reference['identity_group']])
        pairs.append(dict(case_id='validation_%s'%target['index'],kind='real_reference_pair',index=target['index'],
            sample_id=target['sample_id'],identity_group=target['identity_group'],
            caption=target['neutral_prompt'],sketch=target['sketch'],gt=target['gt'],
            original_reference=target['texture'],swapped_reference=reference['texture'],
            reference_indices=[target['index'],reference['index']],seed=110820+target['index'],qualification=score,
            patch_coordinates=target['patch_coordinates'],
            hashes=dict(sketch=target['hashes']['sketch'],gt=target['hashes']['gt'],
                original_reference=target['hashes']['texture'],swapped_reference=reference['hashes']['texture'])))
        if len(pairs)==32:
            break
    value=dict(protocol=PROTOCOL,pairs=pairs,actual_real_targets=len(pairs),planned_minimum=32,
        sufficient_for_planned_screening=len(pairs)>=32,actual_derived_controls=0,
        registry_sha256=sha(out/'inputs/registry.json'),review_sha256=sha(args.review),
        reviewed_records=len(labels),excluded_training_overlap=read(args.candidates)['exclusions'],
        qualification_note='input-only AI review; no human semantic ground truth; same-color thresholds are image proxies')
    path=out/('strict_pairs.json' if len(pairs)>=32 else 'pair_draft_insufficient_%s.json'%len(labels))
    if path.exists():
        assert read(path)==value, '不覆盖已经冻结的配对；需新版本目录'
    else:
        write(path,value)
    print('B_FROZEN_PAIRS',len(pairs),'PASSED_MINIMUM',value['sufficient_for_planned_screening'],flush=True)
