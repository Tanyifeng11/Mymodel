"""确认结果按预设门槛判定，失败亦保留全部身份与图像。"""
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.t3_response_common import *
from tools.t3_response_run import statistics


def valid(value):
    return value is not None and np.isfinite(value)


def analyse():
    assert read(ROOT/'protocol/config.json')==CONFIG
    rows=read(ROOT/'splits/confirmation.json');ids=[r['id'] for r in rows]
    lookup={}
    for arm in MODELS:
        assert read(ROOT/f'metrics/{arm}/complete.json')['passed']
        records=read(ROOT/f'metrics/{arm}/per_identity.json')
        assert len(records)==256
        for rec in records:lookup[rec['stage'],arm,rec['identity']]=rec
    pairaudit=[]
    for row in rows:
        identity=row['id'];bindings=[]
        for stage in ['original','swapped']:
            for arm in MODELS:
                path=ROOT/f'evaluation/{stage}/{arm}/images/{identity}.png'
                meta=read(path.with_suffix('.json'))
                assert sha(path)==meta['output_sha256']
                assert not meta['gt_used_in_inference'] and meta['explicit_latent_verified']
                assert meta['inputs']['reference']==sha(row['reference_alternative' if stage=='swapped' else 'reference'])
                bindings.append((meta['seed'],meta['initial_latent_sha256'],meta['inputs']['gt'],meta['inputs']['sketch']))
        assert len(set(bindings))==1
        pairaudit.append(dict(identity=identity,seed=bindings[0][0],latent_sha256=bindings[0][1],
                              six_output_pairing_passed=True))
    csv_write(ROOT/'audit/six_output_pairing.csv',pairaudit)
    summary=[]
    for stage in ['original','swapped']:
        for arm in MODELS:
            for metric in ['margin','dists_old','dists_new','color_margin','iou','edge_f1','leak','edge_l1']:
                values=[lookup[stage,arm,i][metric] for i in ids if valid(lookup[stage,arm,i].get(metric))]
                stats=statistics(values)
                summary.append(dict(stage=stage,arm=arm,metric=metric,planned_n=128,**stats))
    csv_write(ROOT/'statistics/model_summary.csv',summary)
    common=[i for i in ids if all(valid(lookup['swapped',a,i]['margin']) for a in ['C','T3'])]
    paired=[]
    for identity in ids:
        c=lookup['swapped','C',identity];t=lookup['swapped','T3',identity]
        ok=identity in common
        paired.append(dict(identity=identity,valid_pair=ok,C_margin=c['margin'],T3_margin=t['margin'],
            T3_minus_C=t['margin']-c['margin'] if ok else None,
            failure=None if ok else c['failure'] or t['failure'] or 'nonfinite_distance'))
    csv_write(ROOT/'statistics/primary_per_identity.csv',paired)
    effect=statistics([r['T3_minus_C'] for r in paired if r['valid_pair']])
    effect.update(C_mean=float(np.mean([lookup['swapped','C',i]['margin'] for i in common])) if common else None,
                  T3_mean=float(np.mean([lookup['swapped','T3',i]['margin'] for i in common])) if common else None,
                  difference='T3 minus C',planned_n=128,valid_fraction=len(common)/128,
                  p_holm=effect['p'],family_size=1,missing_n=128-len(common))
    structure=[]
    for stage in ['original','swapped']:
        for metric,bound in CONFIG['structure_margins'].items():
            cases=[i for i in ids if all(valid(lookup[stage,a,i].get(metric)) for a in ['C','T3'])]
            orientation=-1 if metric=='leak' else 1
            diff=np.array([orientation*(lookup[stage,'T3',i][metric]-lookup[stage,'C',i][metric]) for i in cases])
            stats=statistics(diff)
            if len(cases):
                rng=np.random.default_rng(CONFIG['statistics_seed'])
                boots=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
                lower=float(np.quantile(boots,.05/6))
            else:lower=None
            stats.update(stage=stage,metric=metric,oriented='positive favors T3',margin=bound,
               simultaneous_one_sided_lower=lower,
               passed=lower is not None and lower>=-bound and len(cases)/128>=.8,
               valid_fraction=len(cases)/128)
            structure.append(stats)
    primary_pass=(effect['n']/128>=.8 and effect['p'] is not None and effect['p']<.05
                  and effect['ci95'][0]>0 and effect['mean']>=.005)
    continued=bool(primary_pass and all(r['passed'] for r in structure))
    result=dict(protocol=CONFIG['protocol'],primary=effect,primary_pass=bool(primary_pass),
       structural_safety=structure,structural_safety_pass=all(r['passed'] for r in structure),
       continue_new_training=continued,stopped_under_prespecified_rule=not continued,
       reason='all predeclared conditions passed' if continued else 'one or more predeclared response/safety/validity conditions failed',
       human_blind_scores=None,human_reviewers=0,training_seeds_evaluated=[42],
       inference_seeds='one identity-specific seed per identity, shared across all six outputs',
       qualitative_review='pending actual AI image inspection; not human review',
       limitations=['DISTS patch-set response is not motif identity accuracy',
           'legacy Sketch IoU/Edge F1 depend on threshold-based output foreground/edges',
           'Leak is fixed exterior background proxy using old GT, not pure geometric spill',
           'single training seed; pretrained exposure unknown',
           'bootstrap inference is conditional on the fixed reference assignment'],commit=commit())
    write(ROOT/'statistics/decision.json',result)
    write(ROOT/'statistics/model_summary.json',summary)
    csv_write(ROOT/'statistics/structural_safety.csv',structure)
    failures=[dict(arm=a,**r) for a in MODELS for r in read(ROOT/f'metrics/{a}/per_identity.json') if r['failure']]
    write(ROOT/'statistics/failures.json',failures)
    # 固定24例在生成前已选定；额外尾例明确标为结果导向，不替代固定样本。
    fixed=read(ROOT/'splits/qualitative_fixed24.json')
    ranked=sorted([r for r in paired if r['valid_pair']],key=lambda r:(r['T3_minus_C'],r['identity']))
    sets=dict(fixed24=fixed,worst6=[r['identity'] for r in ranked[:6]],best6=[r['identity'] for r in ranked[-6:]])
    write(ROOT/'figures/selection.json',sets)
    for name,selected in sets.items():
        for identity in selected:panel(next(r for r in rows if r['id']==identity),lookup,ROOT/f'figures/{name}/{identity}.jpg')
    blind_material(rows,lookup)
    print('CONFIRMATION_DECISION',json.dumps(result,ensure_ascii=False),flush=True)


def place(canvas,path,x,y,w=160,h=213):
    img=Image.open(path).convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    canvas.paste(img,(x,y))


def panel(row,lookup,path):
    identity=row['id'];canvas=Image.new('RGB',(960,520),'white');draw=ImageDraw.Draw(canvas)
    for j,(key,label) in enumerate([('sketch','Sketch'),('reference','Old reference'),('reference_alternative','New reference')]):
        draw.text((j*320+5,5),identity+'  '+label,fill='black')
        place(canvas,row[key],j*320+80,25)
    for j,(arm,stage) in enumerate((a,s) for a in MODELS for s in ['original','swapped']):
        rec=lookup[stage,arm,identity];margin='NA' if rec['margin'] is None else '%.5f'%rec['margin']
        draw.text((j*160+4,247),arm+' '+stage,fill='black')
        draw.text((j*160+4,263),'M='+margin,fill='black')
        place(canvas,ROOT/f'evaluation/{stage}/{arm}/images/{identity}.png',j*160,284)
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=92)


def blind_material(rows,lookup):
    # 48例由输入哈希选定，不使用任何模型指标。解盲表与待发放材料分开存放。
    selected=sorted(rows,key=lambda r:stable('blind/'+r['id']))[:48]
    keys=[];tasks=[]
    root=ROOT/'human_review/material';root.mkdir(parents=True,exist_ok=True)
    for n,row in enumerate(selected,1):
        identity=row['id'];order=['C','T3'] if int(stable('order/'+identity)[:8],16)%2 else ['T3','C']
        task='Q%03d'%n;keys.append(dict(task=task,identity=identity,A=order[0],B=order[1]))
        canvas=Image.new('RGB',(960,755),'white');draw=ImageDraw.Draw(canvas)
        for j,key in enumerate(['sketch','reference','reference_alternative']):
            draw.text((j*320+10,7),['Sketch','Old reference','New reference'][j],fill='black')
            place(canvas,row[key],j*320+80,27)
        for j,arm in enumerate(order):
            for k,stage in enumerate(['original','swapped']):
                x=j*480+k*200+40
                draw.text((x,260),['A','B'][j]+' '+['Old condition','New condition'][k],fill='black')
                place(canvas,ROOT/f'evaluation/{stage}/{arm}/images/{identity}.png',x,285,200,267)
        draw.text((10,575),task+' | response / new-reference match / sketch structure / design usefulness',fill='black')
        draw.text((10,602),'Compare A and B: 1 strongly A, 2 slightly A, 3 equal, 4 slightly B, 5 strongly B; U unable',fill='black')
        draw.text((10,628),'No method names or automatic scores. Record prior exposure and design background.',fill='black')
        canvas.save(root/(task+'.jpg'),quality=94)
        tasks.append(dict(task=task,image=task+'.jpg',reviewer_id='',design_background='',prior_exposure='',
            response='',reference_match='',structure='',usefulness='',notes=''))
    csv_write(root/'blank_ratings.csv',tasks)
    write(root/'instructions.json',dict(questions=['Which method responds more clearly to the new reference?',
       'Which new output better matches overall new-reference appearance?',
       'Which method preserves sketch garment structure better?',
       'Which result is more useful for a clothing design workflow?'],
       scale={'1':'strong A','2':'slight A','3':'tie','4':'slight B','5':'strong B','U':'unable'},
       pilot_cases=10,formal_cases=48,target_reviewers='20-30 to plan; final human sample needs power analysis',
       required_background_fields=['design_background','prior_exposure'],
       no_fabricated_scores=True,analysis='crossed case/reviewer repeated measures; not independent individual ratings'))
    write(ROOT/'human_review/unblinding_key.json',keys)
    write(ROOT/'human_review/status.json',dict(material_prepared=True,human_reviewers=0,human_blind_scores=None,
       actual_review='NOT_RUN',AI_does_not_replace_human=True))
    with zipfile.ZipFile(ROOT/'human_review/blind_material.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.iterdir()):z.write(p,p.name)


def package():
    result=read(ROOT/'statistics/decision.json')
    jobs=read(ROOT/'jobs.json')
    job_ids=[str(j['id']) for j in jobs]
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(job_ids),
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,NodeList'],text=True)
    (ROOT/'audit/sacct.txt').write_text(accounting,encoding='utf-8')
    code=ROOT/'code';code.mkdir(exist_ok=True)
    for rel in ['tools/t3_response_common.py','tools/t3_response_run.py','tools/t3_response_report.py',
                'tools/e2a_attr_eval.py','inference_IMAGGarment-1.py','eval/metrics.py','eval/eval_utils.py',
                'garment_mask_utils.py','scripts/t3_response.sbatch']:
        target=code/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/rel,target)
    dirs=['audit','protocol','splits','masks','metrics','statistics','figures','human_review','code','evaluation','logs']
    files=sorted(p for d in dirs for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(ROOT/'artifact_manifest.json',manifest)
    completion=dict(execution_complete_under_prespecified_rule=not result['continue_new_training'],
       response_confirmed=result['primary_pass'],structural_safety_confirmed=result['structural_safety_pass'],
       scientific_continuation=result['continue_new_training'],human_blind_complete=False,
       human_blind_scores=None,training_started=False,
       conditional_training='not started: confirmation gate failed' if not result['continue_new_training'] else 'authorized, pending execution',
       generated_images=sum(1 for x in manifest if x['path'].startswith('evaluation/') and x['path'].endswith('.png')),
       artifact_count=len(manifest),commit=commit())
    write(ROOT/'completion.json',completion)
    bundle=ROOT/'review_results.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=3) as tar:
        for p in files+[ROOT/'artifact_manifest.json',ROOT/'completion.json',ROOT/'jobs.json']:
            tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    write(ROOT/'bundle_info.json',dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),
        manifest_count=len(manifest),completion=completion))
    print('BUNDLE',read(ROOT/'bundle_info.json'),flush=True)


if __name__=='__main__':analyse()
