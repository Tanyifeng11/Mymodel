"""纹样评价入口：先准备/冻结输入，后计算；全部新增结果写独立根。"""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from tools.t3_p0p1 import read,write,sha,csv_write
from tools.pattern_fidelity_core import PROTOCOL,descriptors,compare,stable_period,gray_rgb,js

SOURCE=Path(__file__).resolve().parents[1]
SERVER=Path('/share/home/u2515283058/Mymodel')
NAME='pattern_fidelity_eval_20261029'


def resolve(project,path):
    path=Path(path)
    if path.is_file():return path
    relative=path.as_posix().split('/Mymodel/',1)[-1]
    candidate=project/relative
    if candidate.exists():return candidate
    aliases=[('output_eval/t3_p0_p1_20261028/','output_eval/t3_p0_p1_20261028/server_results/'),
        ('output_eval/e2a_sup_comp_01_20261021/','output_eval/e2a_sup_comp_01_20261021/review_results_20261026/'),
        ('output_eval/t3_reference_response_20261027/exploratory_extension_20261027/','output_eval/t3_reference_response_20261027/exploratory_extension_20261027/ablation_results/'),
        ('output_eval/t3_reference_response_20261027/','output_eval/t3_reference_response_20261027/review_results/')]
    for prefix,target in aliases:
        if relative.startswith(prefix):
            candidate=project/(target+relative[len(prefix):])
            if candidate.exists():return candidate
    raise FileNotFoundError(str(path))


def frozen(path,value):
    if path.exists():assert read(path)==value,('FROZEN_CHANGED',str(path))
    else:write(path,value)


def asset(root,path):
    path=Path(path);digest=sha(path);dest=root/'inputs'/(digest+path.suffix.lower())
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(path,dest)
    assert sha(dest)==digest
    return dest.relative_to(root).as_posix(),digest


def patches(mask):
    body=cv2.erode(mask.astype('uint8'),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    coords=[(x,y) for y in range(0,449,32) for x in range(0,321,32) if body[y:y+64,x:x+64].all()]
    count=len(coords)
    if count>8:coords=[coords[i] for i in np.linspace(0,count-1,8).round().astype(int)]
    return body,coords,count


def reference_images(path):
    rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    array=np.asarray(Image.fromarray(rgb).resize((384,512),Image.Resampling.LANCZOS))
    return [array[y:y+64,x:x+64] for y in np.linspace(0,448,4).round().astype(int) for x in np.linspace(0,320,4).round().astype(int)]


def prepare(project,local_main=False):
    from eval.eval_utils import prepare_evaluation_masks
    root=project/'output_eval'/NAME;root.mkdir(parents=True,exist_ok=True)
    frozen(root/'protocol/config.json',PROTOCOL)
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    frozen(root/'protocol/initial_code.json',dict(commit=commit,files={name:sha(SOURCE/name) for name in
        ['tools/pattern_fidelity_core.py','tools/pattern_fidelity_run.py','eval/eval_utils.py','garment_mask_utils.py']}))
    pp=project/'output_eval/t3_p0_p1_20261028'
    source=read(resolve(project,SERVER/'output_eval/t3_p0_p1_20261028/protocol/input_manifest.json'))
    originals={r['id']:r for r in source if r['condition']=='original'}
    swaps={r['id']:r for r in source if r['condition']=='swapped'}
    cases=[];outputs=[]
    for identity,new in swaps.items():
        old=originals[identity]
        cases.append(dict(split='P2b_swap32',id=identity,sketch=old['sketch'],old=old['reference'],new=new['reference']))
        for model in ['B','T1','T2','T3','C','BF-Fashion','ControlNet+IP-Adapter']:
            paths={}
            for condition in ['original','swapped']:
                if model in ['B','T1','T2','T3','C']:
                    stage='P2b' if condition=='original' else 'reference_swap'
                    original=SERVER/f'output_eval/e2a_sup_comp_01_20261021/evaluation/{stage}/{model}/images/{identity}.png'
                else:original=SERVER/f'output_eval/t3_p0_p1_20261028/outputs/{model}/{condition}/{identity}.png'
                paths[condition]=str(resolve(project,original))
            outputs.append(dict(split='P2b_swap32',id=identity,model=model,**paths))
    if not local_main:
        conf=read(project/'output_eval/t3_reference_response_20261027/splits/confirmation.json')
        for row in conf:
            cases.append(dict(split='viewed128',id=row['id'],sketch=row['sketch'],old=row['reference'],new=row['reference_alternative']))
            for model,arm in [('C','C'),('T3','T3'),('T3-no-body','T3_no_body_s42'),('T3-no-rest','T3_no_rest_s42')]:
                folder='t3_reference_response_20261027'+('/exploratory_extension_20261027' if 'no-' in model else '')
                outputs.append(dict(split='viewed128',id=row['id'],model=model,**{condition:str(project/f'output_eval/{folder}/evaluation/{condition}/{arm}/images/{row["id"]}.png') for condition in ['original','swapped']}))
    reference_rows={};identities=[];masks=[];manifest=[]
    for case in cases:
        split=case['split'];identity=case['id'];record=dict(split=split,id=identity)
        for role in ['sketch','old','new']:
            path=resolve(project,case[role]);relative,digest=asset(root,path)
            record[role]=relative;record[role+'_sha256']=digest
            if role!='sketch':reference_rows.setdefault(digest,dict(reference_sha256=digest,token=digest[:10],path=relative,category='unlabeled',annotation='AI reference input only; pending actual inspection'))
        mask=prepare_evaluation_masks((384,512),sketch_path=str(root/record['sketch']),mask_policy='sketch_only')['garment']
        body,coords,candidates=patches(mask);folder=root/'masks'/split;folder.mkdir(parents=True,exist_ok=True)
        for name,value in [('garment',mask),('body',body)]:Image.fromarray(value.astype('uint8')*255).save(folder/(identity+'_'+name+'.png'))
        record.update(mask=str((folder/(identity+'_garment.png')).relative_to(root)),body=str((folder/(identity+'_body.png')).relative_to(root)),coords=coords,
            valid_body=len(coords)>=4,candidate_patch_n=candidates,patch_n=len(coords))
        record['mask_sha256']=sha(root/record['mask']);record['body_sha256']=sha(root/record['body'])
        identities.append(record);masks.append(dict(split=split,id=identity,mask=record['mask'],mask_sha256=record['mask_sha256'],body_sha256=record['body_sha256'],garment_area=int(mask.sum()),body_area=int(body.sum()),patch_n=len(coords),candidate_patch_n=candidates,valid_body=record['valid_body'],reason=None if record['valid_body'] else 'fewer than4 full-body64 patches',coords=json.dumps(coords)))
        for role in ['old','new']:
            parts=reference_images(root/record[role]);strip=Image.fromarray(np.concatenate(parts,axis=1))
            dest=root/'crops/reference_patches'/split/(identity+'_'+role+'.png');dest.parent.mkdir(parents=True,exist_ok=True);strip.save(dest)
    lookup={(r['split'],r['id']):r for r in identities}
    registry={}
    for name in ['model_registry.csv','external_model_registry.csv']:
        path=resolve(project,SERVER/'output_eval/t3_p0_p1_20261028'/name)
        with path.open(encoding='utf-8-sig',newline='') as stream:
            registry.update({r['model_id']:r for r in csv.DictReader(stream)})
    for output in outputs:
        record=lookup[output['split'],output['id']];model=registry.get(output['model'],{})
        item=dict(identity_id=output['id'],model_id=output['model'],split=output['split'],split_status='previously viewed; exploratory',
            checkpoint_sha256=model.get('checkpoint_sha256'),training_seed=model.get('training_seed'),
            sketch_path=record['sketch'],sketch_sha256=record['sketch_sha256'],reference_old_path=record['old'],reference_old_sha256=record['old_sha256'],
            reference_new_path=record['new'],reference_new_sha256=record['new_sha256'],mask_path=record['mask'],mask_sha256=record['mask_sha256'],
            inference_config=model.get('inference_config'),metric_protocol_version=PROTOCOL['version'],notes='no new generation; fixed identity mask and patch positions')
        for condition,role in [('original','old'),('swapped','new')]:
            path=Path(output[condition]);meta=read(path.with_suffix('.json'))
            assert sha(path)==meta['output_sha256']
            # 输入哈希与旧元数据绑定；外部封装的具体字段另保留原件。
            if 'inputs' in meta:
                assert meta['inputs']['sketch']==record['sketch_sha256'] and meta['inputs']['reference']==record[role+'_sha256']
            item['output_'+role+'_path']=str(path);item['output_'+role+'_sha256']=sha(path)
            item['metadata_'+role]=meta
        manifest.append(item)
    frozen(root/'manifests/identities.json',identities);frozen(root/'manifests/outputs.json',manifest)
    csv_write(root/'manifests/existing_outputs.csv',[{k:v for k,v in r.items() if not k.startswith('metadata_')} for r in manifest])
    csv_write(root/'manifests/mask_manifest.csv',masks)
    frozen(root/'pattern_labels/reference_inputs.json',list(reference_rows.values()))
    csv_write(root/'pattern_labels/reference_pattern_labels_draft.csv',list(reference_rows.values()))
    values=list(reference_rows.values());assert len({r['token'] for r in values})==len(values)
    for offset in range(0,len(values),24):
        part=values[offset:offset+24];canvas=Image.new('RGB',(6*192,4*204),'white');draw=ImageDraw.Draw(canvas)
        for index,row in enumerate(part):
            x=index%6*192;y=index//6*204;draw.text((x+3,y+2),row['token'],fill='black')
            im=ImageOps.contain(Image.open(root/row['path']).convert('RGB'),(184,180));canvas.paste(im,(x+(192-im.width)//2,y+20+(180-im.height)//2))
        target=root/'figures/reference_only'/('references_'+str(offset//24+1).zfill(2)+'.png');target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target)
    for filename,value in [('hypotheses.md',dict(primary=PROTOCOL['primary'],scope=PROTOCOL['scope'])),('metric_definitions.md',PROTOCOL),
        ('pattern_category_rules.md',PROTOCOL['category_rules']),('inclusion_exclusion_rules.md',dict(patches=PROTOCOL['patches'],missing=PROTOCOL['statistics']['missing'])),
        ('statistical_plan.md',dict(statistics=PROTOCOL['statistics'],primary_family=PROTOCOL['primary_family'],secondary_family=PROTOCOL['secondary_family']))]:
        p=root/'protocol'/filename;p.write_text('# '+filename+'\n\n```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    counts={split:dict(identities=sum(r['split']==split for r in identities),body_valid=sum(r['split']==split and r['valid_body'] for r in identities)) for split in sorted({r['split'] for r in identities})}
    write(root/'audit/preparation.json',dict(complete=True,local_main_only=local_main,counts=counts,reference_inputs=len(values),output_pairs=len(outputs),model_scores_viewed=False,environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__)))
    print('PREPARED',counts,'REFS',len(values),'OUTPUT_PAIRS',len(outputs),flush=True)


def package(project):
    import tarfile
    root=project/'output_eval'/NAME
    code=root/'code';code.mkdir(exist_ok=True)
    names=[p.relative_to(SOURCE).as_posix() for p in (SOURCE/'tools').glob('pattern_fidelity_*.py')]+['tools/pattern_reference_labels.csv','scripts/pattern_fidelity.sbatch','eval/metrics.py','eval/eval_utils.py','garment_mask_utils.py']
    for name in names:
        dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,dest)
    ids=sorted({m.group(1) for p in (root/'logs').glob('*.log') if (m:=__import__('re').search(r'_(\d+)\.log$',p.name))})
    raw=subprocess.check_output(['sacct','-X','-j',','.join(ids),'--parsable2','--noheader','--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,ExitCode,Start,End,NodeList'],text=True)
    (root/'audit/sacct_raw.txt').write_text(raw,encoding='utf-8')
    ledger=[]
    for line in raw.splitlines():
        job,name,state,seconds,tres,exitcode,start,end,node=line.split('|')[:9]
        match=__import__('re').search(r'gres/gpu=(\d+)',tres);gpus=int(match.group(1)) if match else 0
        ledger.append(dict(job_id=job,name=name,state=state,elapsed_seconds=int(seconds),allocated_gpus=gpus,gpu_hours=int(seconds)*gpus/3600,allocation=tres,exit_code=exitcode,start=start,end=end,node=node))
    csv_write(root/'resource_ledger.csv',ledger)
    write(root/'run_manifest.json',dict(commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),files={name:sha(SOURCE/name) for name in names},
        initial_code=read(root/'protocol/initial_code.json'),evaluation_release=read(root/'protocol/evaluation_release.json') if (root/'protocol/evaluation_release.json').exists() else None,
        new_generation=0,new_training=0,analysis_complete=(root/'report/analysis_summary.json').is_file(),preparation_archive='archives/preparation_review.tar.gz retained separately; not nested in final bundle'))
    paths=[p for p in root.rglob('*') if p.is_file() and p.name not in ['bundle_manifest.json','bundle_info.json'] and not p.name.endswith(('.tar.gz','.tgz')) and 'source' not in p.relative_to(root).parts]
    write(root/'bundle_manifest.json',[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in paths])
    archive=root/'review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in paths+[root/'bundle_manifest.json']:tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    write(root/'bundle_info.json',dict(bytes=archive.stat().st_size,sha256=sha(archive),members=len(paths)+1))
    print('PACKAGE',read(root/'bundle_info.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','package'])
    parser.add_argument('--project',type=Path,default=SERVER);parser.add_argument('--local-main',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.project,args.local_main)
    else:package(args.project)
