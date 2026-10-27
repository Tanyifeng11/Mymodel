"""独立读取持久化结果，不导入项目指标实现；核查patch像素、分母和配对算术。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

R=Path('F:/fuxian/Mymodel/output_eval/e2a_texture_eval_01')
def csv_read(name):return list(csv.DictReader((R/name).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    config=json.loads((R/'metric_config.json').read_text())
    assert sha(R/'input_manifest.csv')==config['input_manifest_sha256']
    assert sha(R/'reference_pattern_labels.csv')==config['labels_sha256']
    assert sha(R/'mask_manifest.csv')==config['mask_manifest_sha256']
    source={}
    for row in csv_read('input_manifest.csv'):
        for p,h in [('output_path','output_sha256'),('reference_path','reference_sha256'),('sketch_path','sketch_sha256'),('mask_path','mask_sha256')]:
            source[row[h]]=row[p]
    for digest,path in source.items():assert sha(path)==digest,path
    coords={};current=None;array=None;patch_count=0
    for p in csv_read('patch_manifest.csv'):
        key=(p['kind'],p['output_sha256'] if p['kind']=='generated' else p['reference_sha256'])
        if key!=current:
            image=Image.open(source[key[1]]).convert('RGB')
            if p['kind']=='reference':image=image.resize((384,512),Image.Resampling.LANCZOS)
            array=np.asarray(image);current=key
        x,y=int(p['x']),int(p['y']);patch=array[y:y+64,x:x+64]
        assert patch.shape==(64,64,3) and hashlib.sha256(patch.tobytes()).hexdigest()==p['pixel_sha256']
        if p['kind']=='generated':
            assert float(p['mask_occupancy'])>=.9
            coords.setdefault((p['identity'],p['stage']),{}).setdefault(p['arm'],[]).append((x,y))
        patch_count+=1
    for identity,arms in coords.items():assert len(arms)==6 and len({str(v) for v in arms.values()})==1,identity
    stats=json.loads((R/'statistics/paired_bootstrap.json').read_text());checks=[]
    for stage,folder,planned,valid in [('original','original_reference',96,87),('swap','swapped_reference',32,27)]:
        data=json.loads((R/f'{folder}/per_identity_metrics.json').read_text());assert len(data)==planned*6
        for arm in ['B','T1','T2','C','T3','T3M']:
            rows=[x for x in data if x['arm']==arm]
            assert len(rows)==planned and sum(x['dists'] is not None for x in rows)==valid
            for x in rows:
                for key,val in x.items():
                    if isinstance(val,float):assert math.isfinite(val),(x['identity'],key)
                if x['dists'] is not None:assert abs(x['dists']-(x['gen_to_ref']+x['ref_to_gen'])/2)<1e-12
                if x.get('response_margin') is not None:assert abs(x['response_margin']-x['dists_old']+x['dists'])<1e-12
        for comp in [x for x in stats['comparisons'] if x['stage']==stage]:
            metric=comp['metric'];arm=comp['comparison'].split('-')[1]
            a={x['identity']:x[metric] for x in data if x['arm']=='C' and x.get(metric) is not None}
            b={x['identity']:x[metric] for x in data if x['arm']==arm and x.get(metric) is not None}
            ids=sorted(a.keys()&b.keys());values=[a[i]-b[i] for i in ids]
            assert len(ids)==comp['n']
            if values:assert abs(float(np.mean(values))-comp['mean'])<1e-13
            checks.append(dict(stage=stage,comparison=comp['comparison'],metric=metric,paired_n=len(ids),mean_verified=True))
    out=dict(passed=True,independent_code_did_not_import_metric_implementation=True,source_files_rehashed=len(source),patch_pixel_SHAs_verified=patch_count,all_six_generated_patch_coordinates_equal=True,comparisons=checks)
    (R/'final_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='comparisons'}),flush=True)

if __name__=='__main__':main()
