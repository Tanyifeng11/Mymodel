"""从已有全量审计建立可追溯验证候选；文本标签不当作人工纹样真值。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    repo=Path(args.metadata_project)
    source=repo/'data/processed/bf_full_audit_v1'
    validation=json.loads((source/'validation_clean.json').read_text(encoding='utf-8-sig'))
    groups={r['sample_id']:r['sample_id'] for r in validation}
    def find(x):
        while groups[x]!=x:
            groups[x]=groups[groups[x]];x=groups[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        groups[max(a,b)]=min(a,b)
    training=json.loads((repo/'data/train_bf_texture.json').read_text(encoding='utf-8-sig'))
    stems={Path(r['cloth']).stem.split('_')[0] for r in training}
    training_ids={'training/'+s for s in stems}
    training_pixels=set();validation_pixels={};validation_hashes={}
    with (source/'audit.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['sample_id'] in training_ids:
                value=row['images'].get('cloth',{}).get('pixel_sha256')
                if value: training_pixels.add(value)
            elif row['split']=='validation':
                validation_pixels[row['sample_id']]=row['images'].get('cloth',{}).get('pixel_sha256')
                validation_hashes[row['sample_id']]=row['images'].get('cloth',{}).get('file_sha256')
    near=set()
    with (source/'near_duplicate_pairs_h4.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if set(row['left']) & training_ids:
                near.update(x for x in row['right'] if x.startswith('validation/'))
            if set(row['right']) & training_ids:
                near.update(x for x in row['left'] if x.startswith('validation/'))
            members=[x for x in row['left']+row['right'] if x in groups]
            for member in members[1:]:union(members[0],member)
    pixels={}
    for identity,pixel in validation_pixels.items():
        if pixel and identity in groups:
            if pixel in pixels:union(identity,pixels[pixel])
            else:pixels[pixel]=identity
    exclusions={'identity_overlap':[],'exact_pixel_overlap':[],'near_duplicate_candidate':[]}
    records=[]
    for index,row in enumerate(validation):
        caption=row['caption'].lower()
        pattern=next((k for k,words in [('stripes',['stripe']),('checks',['plaid','checkered','gingham']),
            ('dots',['polka','dotted'])] if any(w in caption for w in words)),None)
        if pattern is None: continue
        identity=row['sample_id']
        reason=('identity_overlap' if Path(row['cloth']).stem.split('_')[0] in stems else
                'exact_pixel_overlap' if validation_pixels.get(identity) in training_pixels else
                'near_duplicate_candidate' if identity in near else None)
        if reason:
            exclusions[reason].append(identity);continue
        records.append(dict(index=index,sample_id=identity,caption=row['caption'],pattern_candidate=pattern,
            identity_group=find(identity),
            sketch=row['sketch'],gt=row['cloth'],texture=row['texture'],
            audited_gt_file_sha256=validation_hashes[identity],audited_gt_pixel_sha256=validation_pixels[identity]))
    value=dict(version=1,metadata_source='existing official validation clean manifest and full pixel/pHash audit',
        source_hashes={n:sha(p) for n,p in [('validation_clean',source/'validation_clean.json'),
            ('training_manifest',repo/'data/train_bf_texture.json'),('near_pairs',source/'near_duplicate_pairs_h4.jsonl')]},
        exclusions=exclusions,records=records,
        near_duplicates='conservative pHash distance<=4 exclusion; not human duplicate certification',
        upstream_pretraining_exposure='unknown')
    Path(args.out).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('CANDIDATES',len(records),'EXCLUSIONS',{k:len(v) for k,v in exclusions.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata-project',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())
