"""保存首批100条视觉复核及明确文本修订；不改原图/原标注。"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_bf_full import dump
from screen_bf_annotations import ROOT, OUT


CORRECTIONS = {
    5: ('A white sleeveless collared dress with a flared skirt.', '原图未见所述胸袋、Adidas贴片及山形胸标；保留可见领型与裙型'),
    15: ('A white zip-up jacket with long sleeves, a ribbed collar and two diagonal zip pockets.', '原图为拉链夹克与斜插拉链袋，不支持胸袋、按扣和所述品牌胸标'),
    22: ('A white short-sleeved T-shirt with a round neckline and contrasting collar trim.', '原图短袖圆领T恤，无所述胸袋、按扣及品牌贴片'),
    28: ('A white long-sleeved denim jacket with a collar, chest pocket, and snap buttons.', '仅移除原图不支持的左胸山形标识，其余原描述保留'),
    31: ('a white short-sleeved top with a round neckline, a layered ruffle and a gathered hem.', '原图为短袖荷叶层叠上衣，并非长袖罗纹上衣'),
    36: ('a plain white sleeveless tank top with a round neckline.', '原图无袖，原文short sleeves与图像及tank top类别矛盾'),
    63: ('A black sleeveless top with a high neck and a front cutout with crisscross lacing.', '胸口白色为镂空透出的背景，不应描述为白色面料拼片'),
    65: ('a deep red velvet crop top with thin crossed straps and a twisted front detail.', '将原英文句中夹杂的交叉译为crossed，不改其他内容'),
    67: ('a black sleeveless fitted dress with a high neckline and a large keyhole cutout.', '原图为高领镂空结构，不是普通深V领'),
    78: ('A black cropped jacket with a collar, snap buttons, and decorative embroidery.', '仅移除不支持的左胸山形标识，保留原有装饰描述'),
    97: ('A white loose-fitting dress with wide short sleeves, yellow trim and a V-neckline.', '原图有宽短袖，不是无袖')
}


def main():
    rows=[json.loads(s) for s in (OUT/'selected_100.jsonl').read_text(encoding='utf-8').splitlines()]
    severe_mask={21,25,29,31,33,35,38,39,50,57}
    local_mask={20,22,24,26,28,30,32,34,36,37,40,41,42,49,51,53,54,55,59,65,67,69,75,79}
    weak_sketch=set(range(37))|{38,39}
    records=[]; patches=[]
    for r in rows:
        i=r['review_index']; issues=[]
        mask=('not_available' if 'mask' not in r else 'no_obvious_failure_at_preview_scale')
        if i in severe_mask:
            mask='visible_large_foreground_omission';issues.append('mask大面积漏掉可见衣物，需要修复，未自动生成替代')
        elif i in local_mask:
            mask='local_boundary_or_hole_review';issues.append('mask局部边界、肩带、袖身/裤腿空隙或镂空需复核')
        sketch='weak_or_missing_structure' if i in weak_sketch else 'no_obvious_global_shift_at_preview_scale'
        if i in weak_sketch: issues.append('原sketch结构线很弱或大面积缺失；不等同于全图平移错位')
        reference='not_certified'
        if 60<=i<80:
            reference='visible_background_in_reference';issues.append('参考裁块包含大量背景，不代表目标主体外观；需重裁复核')
        elif i in {20,23}:
            reference='label_dominated_reference';issues.append('参考主要包含领标/标签，应复核裁块位置')
        elif i in {0,1,5,14,15,16,17,18,19,22,27,28,32,34,35,36,37,38,40,48,55,56} or i>=80:
            reference='local_content_or_pattern_review';issues.append('参考为局部/低信息/拼接内容，不能认证整件一致，需任务相关复核')
        caption=r['caption']
        if i in CORRECTIONS:
            new,reason=CORRECTIONS[i]
            patches.append(dict(sample_id=r['sample_id'],original_caption=caption,revised_caption=new,reason=reason,
                                target_sha256=hashlib.sha256((ROOT/r['cloth']).read_bytes()).hexdigest()))
            caption=new
        records.append(dict(r,review=dict(reviewer='Codex视觉初审，非独立双人或像素真值',
                            mask=mask,sketch=sketch,reference=reference,issues=issues,
                            caption_status='corrected' if i in CORRECTIONS else 'unchanged_not_fully_certified',
                            caption_after=caption,training_ready=False)))
    dump(OUT/'reviewed_100.jsonl',records)
    dump(OUT/'caption_corrections.jsonl',patches)
    dump(OUT/'needs_review.jsonl',[r for r in records if r['review']['issues']])
    dest=Path('data/processed/bf_annotation_patch_v1')
    dest.mkdir(parents=True,exist_ok=False)
    lookup={p['sample_id']:p for p in patches}; split_counts={}
    for split in ('training','validation','test'):
        src=Path(f'data/processed/bf_full_audit_v1/{split}_clean.jsonl')
        source=[json.loads(s) for s in src.read_text(encoding='utf-8').splitlines()]
        result=[]; count=0
        for r in source:
            original=dict(r)
            if r['sample_id'] in lookup:
                p=lookup[r['sample_id']]
                assert r['caption']==p['original_caption']
                assert hashlib.sha256((ROOT/r['cloth']).read_bytes()).hexdigest()==p['target_sha256']
                r=dict(r,caption=p['revised_caption']);count+=1
            assert all(r[k]==v for k,v in original.items() if k!='caption')
            result.append(r)
        dump(dest/f'{split}_clean.jsonl',result)
        (dest/f'{split}_clean.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        split_counts[split]=dict(samples=len(result),caption_changes=count)
    assert sum(s['caption_changes'] for s in split_counts.values())==len(patches)
    # 校验所有复核图像仍与上一轮全量审计的文件哈希一致。
    selected={r['sample_id']:r for r in records}
    with Path('data/processed/bf_full_audit_v1/audit.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['sample_id'] in selected:
                for k,info in r['images'].items():
                    assert hashlib.sha256((ROOT/r['paths'][k]).read_bytes()).hexdigest()==info['file_sha256']
    summary=dict(reviewed=100,caption_corrections=len(patches),split_counts=split_counts,
                 mask_status=dict(Counter(r['review']['mask'] for r in records)),
                 sketch_status=dict(Counter(r['review']['sketch'] for r in records)),
                 reference_status=dict(Counter(r['review']['reference'] for r in records)),
                 selected_split_counts=dict(Counter(r['sample_id'].split('/')[0] for r in records)),
                 source_images_unchanged=True,mask_sketch_replaced=0,
                 limits=['100条为启发式高风险配额选样，不能估计总体错误率','同分按ID选取，混合纹样组偏向测试包类，不代表类别覆盖均衡',
                         '无自动合格认证，不用启发式修mask或替换sketch','文本更新含验证/测试条目，旧新提示词结果不能直接混比','所有图像路径仍相对BF根目录'])
    (OUT/'review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__': main()
