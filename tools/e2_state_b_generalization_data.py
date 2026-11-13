"""新身份与实拍纹理验证：先排除既有曝光，再审阅输入，最后冻结配对。"""
import copy
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.e2_fft_experiment import SOURCE, read, write, sha
from tools.e2_followup_data import color_distance

COUNTS={'low':8,'multi':3}


def stable(value):
    return hashlib.sha256(('B_NEW_REAL_20261113/'+str(value)).encode()).hexdigest()


def product(value):
    name = str(value).replace('\\','/').rsplit('/',1)[-1].split('.')[0].split('_')[0]
    return name if re.fullmatch(r'\d{6,}',name) else None


def exposure(args, registry):
    by_index = {r['index']:r for r in registry}
    roots, hashes = set(), set()
    files = []
    for pattern in ('*/strict_pairs.json','*/evaluation.json','*/protocol.json','*/split.json',
                    '*/protocol/input_manifest.json','*/inputs/dev64.json','*/inputs/reference_pool.json'):
        files += list((args.project/'output_eval').glob(pattern))
    files += [SOURCE/'configs/e2_b_input_review.json', SOURCE/'configs/e2_followup_review.json',
              SOURCE/'configs/garment_test128_20261031.json']
    files = sorted(set(p for p in files if p.exists()))
    def walk(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('identity_group','sample_id','id','sketch','gt','cloth','texture',
                           'reference','original_reference','swapped_reference') and isinstance(item,str):
                    root = product(item)
                    if root:roots.add(root)
                walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
        elif isinstance(value,str):
            if re.fullmatch(r'[0-9a-f]{64}',value):hashes.add(value)
            match = re.fullmatch(r'validation_(\d+)',value)
            if match and int(match[1]) in by_index:
                roots.add(product(by_index[int(match[1])]['identity_group']))
    source_audit = []
    for path in files:
        data = read(path)
        if path.name == 'e2_b_input_review.json':
            for index in data['records']:
                if int(index) in by_index:roots.add(product(by_index[int(index)]['identity_group']))
        walk(data)
        source_audit.append(dict(path=str(path),sha256=sha(path)))
    return roots, hashes, source_audit


def stratum(row):
    quantiles = np.asarray(row['color']['lab_quantiles'])
    spread = float(np.linalg.norm(quantiles[2,1:]-quantiles[0,1:]))
    return ('low' if spread<=6 else 'multi' if spread>=12 else 'middle'), spread


def perceptual_hash(path):
    pixels=np.asarray(Image.open(path).convert('L').resize((32,32),Image.Resampling.LANCZOS),np.float32)
    coefficients=cv2.dct(pixels)[:8,:8].ravel()[1:]
    return sum(int(v>np.median(coefficients))<<i for i,v in enumerate(coefficients))


def pair_qualification(a,b):
    delta,palette = color_distance(a['color'],b['color'])
    if delta>5 or palette>10:return None
    pa,pb = a['spectral_proxy'],b['spectral_proxy']
    if not(pa['valid'] and pb['valid']):return None
    ratio = max(pa['period'],pb['period'])/min(pa['period'],pb['period'])
    angle = abs(pa['line_angle']-pb['line_angle']);angle=min(angle,180-angle)
    family = a['pattern_candidate']!=b['pattern_candidate']
    if not family and ratio<1.5 and not(a['pattern_candidate']=='stripes' and angle>=45):return None
    return dict(mean_lab_distance=delta,palette_quantile_distance=palette,
                family_change=family,period_ratio=ratio,orientation_difference=angle)


def pool(args,by_stratum):
    rows=[dict(row,stratum=group) for group,items in by_stratum.items() for row in items]
    folder=args.out/'input_pool';folder.mkdir(exist_ok=True)
    write(folder/'pool.json',dict(records=rows,model_outputs_seen=False))
    for page in range((len(rows)+24)//25):
        canvas=Image.new('RGB',(1280,1000),'white');draw=ImageDraw.Draw(canvas)
        for slot,r in enumerate(rows[page*25:(page+1)*25]):
            x,y=slot%5*256,slot//5*200
            draw.text((x+2,y+2),str(r['index'])+' '+r['stratum'],fill='black')
            for col,key in enumerate(('gt','texture')):
                canvas.paste(Image.open(args.data/r[key]).convert('RGB').resize((126,174)),(x+128*col,y+22))
        canvas.save(folder/f'pool_{page:02d}.jpg',quality=96)
    archive=args.out/'input_pool_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in folder.iterdir():z.write(file,file.relative_to(args.out))
    print('INPUT_POOL_ARCHIVE',len(rows),archive.stat().st_size,sha(archive),flush=True)


def prepare(args):
    registry_path = args.project/'output_eval/e2_b_screen_20261108/inputs/registry.json'
    registry = read(registry_path)['records']
    audit = args.project/'output_eval/e2_b_screen_20261108/audit/training_inputs_final.json'
    assert read(audit)['passed']
    candidates = read(SOURCE/'configs/e2_b_candidates.json')
    forbidden = set(candidates['exclusions']['actual_training_input_overlap'])
    assert not forbidden.intersection(r['sample_id'] for r in registry)
    roots, hashes, sources = exposure(args,registry)
    available = [r for r in registry if product(r['identity_group']) not in roots
                 and not set(r['hashes'].values()).intersection(hashes)
                 and r['mask_valid'] and len(r['patch_coordinates'])>=4]
    history = [(r['sample_id'],perceptual_hash(args.data/r['gt'])) for r in registry
               if product(r['identity_group']) in roots]
    near_history=[];distinct=[]
    for row in available:
        fingerprint=perceptual_hash(args.data/row['gt'])
        distance,name=min((bin(fingerprint^v).count('1'),name) for name,v in history)
        if distance<=4:near_history.append(dict(sample_id=row['sample_id'],near=name,hamming=distance))
        else:distinct.append(row)
    available=distinct
    if getattr(args,'pool_review',None):
        reviewed=read(args.pool_review)
        assert reviewed['model_outputs_seen'] is False
        available=[r for r in available if reviewed['records'][str(r['index'])]['eligible']]
    by_stratum = {s:sorted([r for r in available if stratum(r)[0]==s],key=lambda r:stable(r['sample_id']))
                  for s in ('low','multi')}
    if getattr(args,'pool_only',False):
        pool(args,by_stratum);return
    used = set(); pairs = []
    for group in ('low','multi'):
        count = 0
        for target in by_stratum[group]:
            if product(target['identity_group']) in used:continue
            choices = []
            for donor in by_stratum[group]:
                if product(donor['identity_group']) in used or donor['identity_group']==target['identity_group']:continue
                qualification = pair_qualification(target,donor)
                if qualification is not None:
                    choices.append((not qualification['family_change'],qualification['mean_lab_distance'],
                                    stable(donor['sample_id']),donor,qualification))
            if not choices:continue
            _,_,_,donor,qualification = min(choices,key=lambda x:x[:3])
            pair = dict(candidate_id=f'{group}_{count:02d}',stratum=group,
                        target=copy.deepcopy(target),donor=copy.deepcopy(donor),qualification=qualification)
            for row in (target,donor):
                for key in ('sketch','gt','texture'):
                    assert sha(args.data/row[key])==row['hashes'][key],(row['sample_id'],key)
                assert row['hashes']['gt']==row['audited_gt_file_sha256']
            pairs.append(pair);count+=1
            used.update(product(r['identity_group']) for r in (target,donor))
            if count==16:break
    assert all(sum(p['stratum']==s for p in pairs)>=n for s,n in COUNTS.items())
    folder = args.out/'inputs';folder.mkdir(parents=True,exist_ok=True)
    for page in range((len(pairs)+7)//8):
        canvas=Image.new('RGB',(1200,888),'white');draw=ImageDraw.Draw(canvas)
        for slot,p in enumerate(pairs[page*8:(page+1)*8]):
            x,y=(slot%2)*600,(slot//2)*222
            label=f"{p['candidate_id']} {p['target']['index']} / {p['donor']['index']} "
            label+=p['target']['pattern_candidate']+' -> '+p['donor']['pattern_candidate']
            draw.text((x+3,y+3),label,fill='black')
            paths=[p['target']['sketch'],p['target']['gt'],p['target']['texture'],p['donor']['texture']]
            for col,path in enumerate(paths):
                canvas.paste(Image.open(args.data/path).convert('RGB').resize((148,196)),(x+150*col,y+23))
        canvas.save(folder/f'candidate_{page:02d}.jpg',quality=96)
    draft=dict(pairs=pairs,registry_sha256=sha(registry_path),training_audit_sha256=sha(audit),
        exclusions=copy.deepcopy(candidates['exclusions']),exposure_sources=sources,
        excluded_product_roots=len(roots),available=len(available),
        history_near_duplicates=near_history,history_phash_threshold=4,
        strata_available={s:len(v) for s,v in by_stratum.items()},source_has_training_overlap_removed=True,
        selection='input-only stable hash; 16 candidate pairs per colour stratum, before generated outputs',
        stratum='reference Lab ab 10%-90% spread <=6 low, >=12 multicolour; middle excluded by design',
        pair_rule='original references, mean Lab <=5, quantile palette <=10; distinct pattern family preferred',
        pool_review_sha256=sha(args.pool_review) if getattr(args,'pool_review',None) else None,
        upstream_pretraining_exposure='unknown; candidate audit covers current training, not all upstream pretraining')
    write(folder/'candidates.json',draft)
    archive=args.out/'input_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in folder.iterdir():
            if file.is_file():z.write(file,file.relative_to(args.out))
    print('NEW_REAL_INPUTS',len(available),{s:len(v) for s,v in by_stratum.items()},len(pairs),flush=True)
    print('INPUT_REVIEW_ARCHIVE',archive.stat().st_size,sha(archive),flush=True)


def freeze(args):
    from eval.eval_utils import prepare_evaluation_masks
    from models.e2_texture_diagnostics import reference_control
    from tools.e2_fft_experiment import environment
    draft = read(args.out/'inputs/candidates.json')
    review = read(args.review)
    assert review['model_outputs_seen'] is False and review['human_reviewers']==0
    chosen = []
    for group in ('low','multi'):
        eligible = [p for p in draft['pairs'] if p['stratum']==group and review['records'][p['candidate_id']]['eligible']]
        assert len(eligible)>=COUNTS[group],(group,len(eligible))
        chosen += eligible[:COUNTS[group]]
    rows=[]
    for item in chosen:
        p,d=item['target'],item['donor']
        folder=args.out/'cases'/('validation_'+str(p['index']));folder.mkdir(parents=True,exist_ok=True)
        mask=prepare_evaluation_masks((384,512),sketch_path=str(args.data/p['sketch']),mask_policy='sketch_only')['garment']
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
        shutil.copyfile(args.data/p['sketch'],folder/'sketch.jpg')
        a,b=(np.asarray(Image.open(args.data/r['texture']).convert('RGB').resize((384,512),Image.Resampling.BICUBIC)) for r in (p,d))
        for name,pixels in [('real_a',a),('real_b',b)]:Image.fromarray(pixels).save(folder/(name+'_ref.png'))
        seed=20261113+int(stable(p['sample_id'])[:6],16)
        for name in ('v16','h16','v32','h32'):
            Image.fromarray(reference_control(a,name,seed)).save(folder/(name+'_ref.png'))
        rows.append(dict(case_id=folder.name,index=p['index'],identity_group=p['identity_group'],
            donor_identity_group=d['identity_group'],stratum=item['stratum'],caption=p['neutral_prompt'],
            sketch=p['sketch'],gt=p['gt'],original_reference=p['texture'],swapped_reference=d['texture'],
            seed=seed,patch_coordinates=p['patch_coordinates'],qualification=item['qualification'],
            input_review=review['records'][item['candidate_id']],
            source_hashes=dict(target=p['hashes'],donor=d['hashes']),
            reference_hashes={c:sha(folder/(c+'_ref.png')) for c in ('v16','h16','v32','h32','real_a','real_b')}))
    assert len(rows)==sum(COUNTS.values()) and len({product(r['identity_group']) for r in rows}|{product(r['donor_identity_group']) for r in rows})==2*len(rows)
    return dict(pairs=rows,input_audit=draft,review_sha256=sha(args.review),source=environment(),
                training_updates=0,selection_uses_generated_outputs=False)
