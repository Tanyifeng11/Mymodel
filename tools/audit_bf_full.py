"""全量 BF 文件审计与非破坏性清单清洗；不判定语义标注正确性。"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def dump(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def inspect(job):
    root, row = job
    row['errors'], row['warnings'], row['images'] = [], [], {}
    for key, rel in row['paths'].items():
        path = root/rel
        try:
            if key == 'text':
                row['caption'] = path.read_text(encoding='utf-8-sig').strip()
                if not row['caption']:
                    row['errors'].append('text:empty')
                continue
            with Image.open(path) as im:
                im.load()
                rgb = np.asarray(im.convert('RGB'))
                gray = np.asarray(im.convert('L'))
                info = dict(size=list(im.size), mode=im.mode, std=float(gray.std()),
                            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if key == 'cloth':
                    info['pixel_sha256'] = hashlib.sha256(str(im.size).encode()+rgb.tobytes()).hexdigest()
                    small = cv2.resize(gray.astype('float32'), (32, 32), interpolation=cv2.INTER_AREA)
                    freq = cv2.dct(small)[:8, :8].flatten()[1:]
                    info['phash'] = format(sum(int(v > np.median(freq)) << i for i, v in enumerate(freq)), '016x')
                if key == 'mask':
                    info['unique_values'] = int(len(np.unique(gray)))
                    info['foreground_fraction'] = float((gray > 127).mean())
                    if info['unique_values'] > 2:
                        row['warnings'].append('mask:not_binary')
                if info['std'] < 1:
                    row['warnings'].append(key+':nearly_uniform')
                row['images'][key] = info
        except Exception as exc:
            row['errors'].append(key+':'+type(exc).__name__+':'+str(exc))
    target = row['images'].get('cloth', {}).get('size')
    for key in ('mask', 'sketch'):
        if key in row['images'] and row['images'][key]['size'] != target:
            row['warnings'].append(key+':size_mismatch')
    return row


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    rows, inventory, orphans = [], [], []
    bases = [('training', a.root/'training'), ('validation', a.root/'validation')]
    bases += [('test', p) for p in sorted((a.root/'test').iterdir()) if p.is_dir()]
    for split, base in bases:
        maps = {}
        for key, folder in [('cloth', 'cloth' if split == 'training' else 'gt'),
                            ('texture', 'texture'), ('sketch', 'sketch'), ('text', 'text'), ('mask', 'mask')]:
            directory = base/folder
            if key == 'mask' and not directory.exists():
                continue
            index = defaultdict(list)
            for path in sorted(directory.glob('*')):
                if path.is_file():
                    index[path.stem].append(path.relative_to(a.root).as_posix())
            maps[key] = index
            inventory.append(dict(directory=directory.relative_to(a.root).as_posix(), files=sum(map(len,index.values()))))
        for stem, targets in maps['cloth'].items():
            paths, index_errors = {}, []
            for key, index in maps.items():
                values = index.get(stem, [])
                if len(values) != 1:
                    index_errors.append(f'{key}:file_count={len(values)}')
                if values:
                    paths[key] = values[0]
            rows.append(dict(sample_id=base.relative_to(a.root).as_posix()+'/'+stem,
                             split=split, paths=paths, index_errors=index_errors))
        for key, index in maps.items():
            for stem in index.keys()-maps['cloth'].keys():
                orphans.append(dict(modality=key, paths=index[stem], reason='no_target_with_same_stem'))
    dump(a.output/'inventory.jsonl', inventory)
    dump(a.output/'orphans.jsonl', orphans)
    print('indexed', len(rows), flush=True)
    audited = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.output/'audit.jsonl').open('w',encoding='utf-8') as f:
        for i, row in enumerate(pool.map(inspect, ((a.root,r) for r in rows)), 1):
            audited.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if i % 2000 == 0:
                print('decoded', i, '/', len(rows), flush=True)
    exact, near = defaultdict(list), defaultdict(list)
    for r in audited:
        im = r['images'].get('cloth', {})
        if 'pixel_sha256' in im:
            exact[im['pixel_sha256']].append(r)
            near[im['phash']].append(r)
    exact_groups = [dict(pixel_sha256=h, samples=[r['sample_id'] for r in group],
                         splits=sorted({r['split'] for r in group})) for h,group in exact.items() if len(group)>1]
    dump(a.output/'exact_duplicates.jsonl', exact_groups)
    # 相同pHash仅为候选，不是重复结论；未穷举非零汉明距离或裁剪/旋转近重复。
    near_groups = [dict(phash=h,samples=[r['sample_id'] for r in group],
                        cross_split=len({r['split'] for r in group})>1)
                   for h,group in near.items() if len({r['images']['cloth']['pixel_sha256'] for r in group})>1]
    dump(a.output/'near_duplicate_candidates.jsonl', near_groups)
    exclusions, clean = [], defaultdict(list)
    for r in audited:
        reasons = r['index_errors']+r['errors']
        h = r['images'].get('cloth',{}).get('pixel_sha256')
        if r['split']=='training' and h and any(x['split']!='training' for x in exact[h]):
            reasons = reasons+['exact_target_duplicate_in_validation_or_test']
        if reasons:
            exclusions.append(dict(sample_id=r['sample_id'],reasons=reasons,paths=r['paths']))
        else:
            clean[r['split']].append(dict(sample_id=r['sample_id'],caption=r['caption'],
                                         **{k:v for k,v in r['paths'].items() if k!='text'},
                                         warnings=r['warnings']))
    for split in ('training','validation','test'):
        dump(a.output/f'{split}_clean.jsonl',clean[split])
        # 与项目训练JSON字段相容；路径相对BF根，而非training根。
        (a.output/f'{split}_clean.json').write_text(json.dumps(clean[split],ensure_ascii=False,indent=2),encoding='utf-8')
    dump(a.output/'excluded.jsonl',exclusions)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    disk = {r['paths'].get('cloth'):r for r in audited if r['split']=='training'}
    manifest_issues = []
    seen = set()
    for i,r in enumerate(manifest):
        rel = 'training/'+r['cloth'].replace('\\','/')
        seen.add(rel)
        match = disk.get(rel)
        if not match:
            manifest_issues.append(dict(index=i,reason='target_not_in_disk_index',path=rel))
            continue
        for k in ('texture','sketch'):
            if 'training/'+r[k].replace('\\','/') != match['paths'].get(k):
                manifest_issues.append(dict(index=i,reason=k+':path_mismatch'))
        if r['caption'].strip()!=match.get('caption'):
            manifest_issues.append(dict(index=i,reason='caption_differs_from_text_file'))
    dump(a.output/'manifest_issues.jsonl',manifest_issues)
    summary = dict(total=len(audited), splits=dict(Counter(r['split'] for r in audited)),
                   clean_counts={s:len(v) for s,v in clean.items()}, excluded=len(exclusions),
                   exclusion_reasons=dict(Counter(x for r in exclusions for x in r['reasons'])),
                   warning_counts=dict(Counter(x for r in audited for x in r['warnings'])),
                   exact_duplicate_groups=len(exact_groups), cross_split_exact_groups=sum(len(g['splits'])>1 for g in exact_groups),
                   near_candidate_groups=len(near_groups),orphans=len(orphans),
                   manifest_rows=len(manifest),manifest_issues=len(manifest_issues),
                   disk_targets_not_in_manifest=len(disk.keys()-seen),
                   image_sizes={k:dict(Counter(str(r['images'][k]['size']) for r in audited if k in r['images'])) for k in ('cloth','texture','sketch','mask')},
                   policy=['原文件不变；不移动划分；不删除同划分重复','缺失/解码错误/空文本/多文件歧义排除','跨划分像素精确重复仅排除训练样本；验证测试互相重复仍需报告','尺寸异常及近重复仅警告','配对检查仅同名和清单路径一致，不代表语义或几何正确','清洗清单不是人工标注质量认证；路径相对BF根目录'])
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('exclusion_reasons','policy')},ensure_ascii=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('F:/fuxian/dataset/datasets/BF'))
    p.add_argument('--output',type=Path,default=Path('data/processed/bf_full_audit_v1'))
    p.add_argument('--manifest',type=Path,default=Path('data/train_bf_texture.json'))
    p.add_argument('--workers',type=int,default=8)
    main(p.parse_args())
