"""记录 pilot100 的逐页视觉初审；不是像素真值或训练准入判定。"""
import json
from collections import Counter
from pathlib import Path

from rebuild_bf_pilot import sha


def main():
    out = Path('data/processed/bf_rebuild_pilot100_v1')
    root = Path('F:/fuxian/dataset/datasets/BF/training')
    rows = [json.loads(s) for s in (out/'draft.jsonl').read_text(encoding='utf-8').splitlines()]
    # 依据 page_00 至 page_09；只记录明显现象，不把其余样本默认为合格。
    loss = {1, 17, 44, 49}
    bridge = {0, 2, 3, 5, 9, 10, 13, 18, 21, 27, 28, 31, 36, 37,
              40, 47, 50, 53, 54, 58, 60, 64, 65, 69, 74, 76, 78,
              82, 84, 87, 95, 99}
    improved_gap = {4, 23, 30, 34, 39, 41, 46, 48, 51, 57, 59, 63, 75, 86}
    corrections = {
        0: 'A printed jacket with a striped hem.',
        3: 'A light blue hoodie with a white graphic.',
        4: 'A dark hooded dress with sleeve graphics and stripes.',
        10: 'A bomber jacket with star patches and striped ribbing.',
        16: 'A striped cropped blouse with puff sleeves.',
        18: 'A dark long-sleeved dress with a contrasting vertical stripe.',
        21: 'A striped blazer.',
        22: 'A sleeveless dress with mixed geometric patterns.',
        24: 'A striped skirt with front buttons.',
        25: 'A striped blazer.',
        39: 'A striped halter-neck maxi dress.',
        42: 'A short skirt with contrasting patterned panels.',
        43: 'A fringed accessory with contrasting bands.',
        44: 'A patterned jacket with a contrasting neck bow.',
        45: 'A red hoodie with contrasting sleeve graphics.',
        47: 'A patterned bomber jacket with striped ribbing.',
        55: 'A patterned long-sleeved shirt.',
        56: 'A checked coat.',
        57: 'A double-breasted blazer.',
        58: 'A dark dress with checked collar and hem panels.',
        59: 'A long-sleeved top with a fine pattern.',
        61: 'A checked cape.',
        66: 'A windowpane-check blazer.',
        71: 'A belted patterned coat.',
        82: 'A belted jacket.',
        85: 'A pair of grey trousers.',
        89: 'A pair of trousers with a fine pattern.',
        90: 'A checked poncho.',
        92: 'A dark button-front shirt.',
    }
    for r in rows:
        i = r['pilot_index']
        for key, digest in r['source_hashes'].items():
            assert sha(root/r[key]) == digest, (i, key)
        assert sha(root/'mask'/(Path(r['cloth']).stem+'.png')) == r['old_mask_sha256']
        for path in r['generated'].values():
            assert (out/path).is_file(), path
        tag = ('明显误删服装区域' if i in loss else
               '仍有或新增空隙桥接' if i in bridge else
               '可见空隙恢复改善_不代表完整合格' if i in improved_gap else
               '缩略图未见灾难性错误_仍需边界复核')
        caption = corrections.get(i, r['caption_draft'])
        r.update(review_status='visual_screened_not_train_ready', mask_visual_note=tag,
                 caption_reviewed=caption, caption_review_basis='接触表视觉初审；不确定细纹样不强行标为格纹',
                 training_ready=False)
        (out/r['generated']['text']).write_text(caption+'\n', encoding='utf-8')
    (out/'reviewed.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    summary = dict(count=len(rows), mask_screening=dict(Counter(r['mask_visual_note'] for r in rows)),
                   caption_corrections=len(corrections), source_hashes_unchanged=True,
                   split='training_only', training_ready=False,
                   decision='不要全量替换；先处理白纹误删及袖身桥接。轮廓草图不等价于原结构草图。',
                   preview_note='preview保留初始caption草稿；修订文本以reviewed.jsonl和text目录为准')
    (out/'review_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
