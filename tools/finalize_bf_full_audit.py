"""补充pHash汉明距离<=4的近重复候选，验证清洗结果，生成候选预览。"""
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from audit_bf_full import dump


def main():
    out = Path('data/processed/bf_full_audit_v1')
    root = Path('F:/fuxian/dataset/datasets/BF')
    rows = [json.loads(s) for s in (out/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for r in rows:
        h = r['images'].get('cloth',{}).get('phash')
        if h is not None:
            groups[int(h,16)].append(r)
    buckets = [defaultdict(list) for _ in range(5)]
    # 63位拆成5段；距离<=4的任意两哈希必有至少一段完全相同。
    shifts, widths = [0,13,26,39,51], [13,13,13,12,12]
    candidates = []
    for n,(h,group) in enumerate(groups.items(),1):
        possible = set()
        for b,shift,width in zip(buckets,shifts,widths):
            possible.update(b[(h>>shift)&((1<<width)-1)])
        for other in possible:
            distance = (h^other).bit_count()
            if distance<=4:
                candidates.append(dict(left=[r['sample_id'] for r in groups[other]],
                                       right=[r['sample_id'] for r in group],distance=distance,
                                       cross_split=len({r['split'] for r in groups[other]+group})>1))
        if len({r['images']['cloth']['pixel_sha256'] for r in group})>1:
            candidates.append(dict(left=[group[0]['sample_id']],right=[r['sample_id'] for r in group[1:]],
                                   distance=0,cross_split=len({r['split'] for r in group})>1))
        for b,shift,width in zip(buckets,shifts,widths):
            b[(h>>shift)&((1<<width)-1)].append(h)
        if n%10000==0:
            print('near hashes',n,flush=True)
    candidates.sort(key=lambda r:(not r['cross_split'],r['distance'],r['left'][0],r['right'][0]))
    dump(out/'near_duplicate_pairs_h4.jsonl',candidates)
    index = {r['sample_id']:r for r in rows}
    preview = out/'near_preview'
    preview.mkdir(exist_ok=True)
    for page in range(min(4,(len(candidates)+11)//12)):
        sheet = Image.new('RGB',(1200,1080),'white')
        draw = ImageDraw.Draw(sheet)
        for j,pair in enumerate(candidates[page*12:page*12+12]):
            x,y=(j%3)*400,(j//3)*270
            draw.text((x,y),f"pHash distance={pair['distance']}; candidate only",fill='black')
            for side,key in enumerate(('left','right')):
                r=index[pair[key][0]]
                with Image.open(root/r['paths']['cloth']) as im:
                    im=im.convert('RGB'); im.thumbnail((195,220)); sheet.paste(im,(x+side*200,y+20))
                draw.text((x+side*200,y+245),r['sample_id'],fill='black')
        sheet.save(preview/f'page_{page:02d}.png')
    clean={s:[json.loads(x) for x in (out/f'{s}_clean.jsonl').read_text(encoding='utf-8').splitlines()] for s in ('training','validation','test')}
    hashes={s:{index[r['sample_id']]['images']['cloth']['pixel_sha256'] for r in rs} for s,rs in clean.items()}
    assert not hashes['training'] & (hashes['validation']|hashes['test'])
    excluded=[json.loads(x) for x in (out/'excluded.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(map(len,clean.values()))+len(excluded)==len(rows)
    orphan_rows=[json.loads(x) for x in (out/'orphans.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in orphan_rows:
        r['classification']='auxiliary_metadata' if all(x.endswith('mask_metadata.json') for x in r['paths']) else 'unpaired_file'
    dump(out/'orphans.jsonl',orphan_rows)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary.update(near_h4_candidate_groups=len(candidates),near_h4_cross_split_groups=sum(r['cross_split'] for r in candidates),
                   clean_train_eval_pixel_overlap=0, clean_validation_test_shared_pixel_hashes=len(hashes['validation']&hashes['test']),
                   validation='清单数量守恒及训练-评测像素去重检查通过',
                   near_limits='pHash距离<=4全量候选；不是人工重复结论，不覆盖所有裁剪/旋转/视角变化',
                   auxiliary_metadata_files=sum(r['classification']=='auxiliary_metadata' for r in orphan_rows))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
