"""失败原因、资源账本和结果包；不更改已冻结门槛，不生成服务器叙述文档。"""
import argparse
from collections import Counter
import subprocess
import tarfile
import cv2
import numpy as np
from PIL import Image,ImageDraw
from models.e34_sarr_refiner import safe_mask
from tools.e34_sarr_protocol import *

def masks():
    cv2.setNumThreads(1)
    audit=read(OUT/'masks_audit.json');rows=read(OUT/'splits/dev128.json')
    lookup={r['id']:r for r in rows};failures=[];sensitivity=[]
    for r in audit['records']:
        source=Image.open(lookup[r['id']]['sketch']).convert('RGB')
        m0,safe,line,info=safe_mask(source)
        reason=[]
        if r['mask_source']!='sketch_flood_fill' or r['mask_low_confidence']:reason.append('unreliable_sketch_mask')
        if r['proposed_image_coverage']<=.03:reason.append('safe_area_le_3pct_image')
        if r['proposed_garment_coverage']<=.20:reason.append('safe_area_le_20pct_garment')
        if not r['valid']:failures.append(dict(r,reasons=reason))
        # 仅解释文档歧义；不以这项较弱保护重新准入，也不改变任何实际掩码。
        interior=(cv2.erode(m0.astype(np.uint8),np.ones((17,17),np.uint8))>0)
        interior &= ~(cv2.dilate(line.astype(np.uint8),np.ones((3,3),np.uint8))>0)
        feather=np.clip(cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)/8,0,1)
        reliable=r['mask_source']=='sketch_flood_fill' and not r['mask_low_confidence']
        valid=bool(reliable and feather.sum()>.03*feather.size and feather.sum()>.20*max(m0.sum(),1))
        sensitivity.append(dict(id=r['id'],kernel17_valid=valid,actual_distance17_valid=r['valid'],
            kernel17_coverage=float(feather.mean()),actual_distance17_coverage=r['image_coverage']))
    write(OUT/'mask_failure_analysis.json',dict(n=128,valid_n=audit['valid_n'],valid_rate=audit['valid_rate'],
        required_rate=.80,reason_counts=dict(Counter(k for r in failures for k in r['reasons'])),
        failures=failures,kernel17_sensitivity=dict(valid_n=sum(r['kernel17_valid'] for r in sensitivity),
            deployment_unchanged=True,posthoc_diagnostic_only=True,records=sensitivity)))
    folder=OUT/'visual_audit/mask_failures';folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(failures),8):
        page=Image.new('RGB',(4*384,2*305),'white');draw=ImageDraw.Draw(page)
        for i,r in enumerate(failures[start:start+8]):
            row=lookup[r['id']];x=(i%4)*384;y=(i//4)*305
            sketch=Image.open(row['sketch']).convert('RGB').resize((192,256),Image.Resampling.BILINEAR)
            raw,_,_,_=safe_mask(Image.open(row['sketch']).convert('RGB'))
            gray=np.asarray(Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR))
            overlay=gray.copy();overlay[raw]=(.65*overlay[raw]+.35*np.array([30,180,255])).astype(np.uint8)
            draw.text((x+2,y+2),r['id']+' | '+','.join(r['reasons']),fill='black')
            draw.text((x+2,y+18),'pre-bypass coverage %.3f / garment %.3f'%(r['proposed_image_coverage'],r['proposed_garment_coverage']),fill='black')
            page.paste(sketch,(x,y+40));page.paste(Image.fromarray(overlay).resize((192,256)),(x+192,y+40))
        page.save(folder/('page%02d.png'%(start//8)))
    print('MASK_FAILURES',len(failures),dict(Counter(k for r in failures for k in r['reasons'])),
        'KERNEL17_DIAGNOSTIC_VALID',sum(r['kernel17_valid'] for r in sensitivity),flush=True)

def accounting(job_ids):
    text=subprocess.check_output(['sacct','-X','-j',','.join(map(str,job_ids)),'--noheader','--parsable2',
        '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode'],text=True)
    entries=[]
    for line in text.strip().splitlines():
        values=line.split('|');keys=['job_id','name','partition','state','elapsed_seconds','allocated','exit_code']
        entries.append(dict(zip(keys,values)))
    gpu_seconds=0
    for r in entries:
        if 'gres/gpu=' in r['allocated']:
            n=int(r['allocated'].split('gres/gpu=')[1].split(',')[0]);gpu_seconds+=int(r['elapsed_seconds'])*n
    write(OUT/'resource_accounting.json',dict(jobs=entries,gpu_seconds=gpu_seconds,gpu_hours=gpu_seconds/3600,
        note='Slurm allocated elapsed time, including any failed job; final CPU finish checked after completion'))

def import_review():
    review=read('tools/e34_sarr_s0_review.json')
    rows=read(OUT/'splits/dev128.json')
    assert [r['id'] for r in review['records']]==[r['id'] for r in rows[:24]]
    write(OUT/'s0_AI_visual_review.json',review)
    decision(S0='pass' if review['task_match_pass'] else 'fail',
        stopped=not review['task_match_pass'],next_phase='S2a' if review['task_match_pass'] else 'stop_E34_SARR',
        stop_stage=None if review['task_match_pass'] else 'S0',
        reason=None if review['task_match_pass'] else 'S0 local RGB refinement task mismatch')

def finalize():
    init();verify_sources()
    audit=read(OUT/'masks_audit.json');summary=read(OUT/'s0_summary.json')
    review=read(OUT/'s0_AI_visual_review.json');rows=read(OUT/'splits/dev128.json')
    state=read(OUT/'final_decision.json')
    checks=dict(fixed128_generated=summary['n']==128,
        all_cases_validated=all((OUT/'s0_e5_dev'/r['id']/'audit.json').exists() for r in rows),
        exact_replay=any(read(p).get('first_identity_exact_replay') is True for p in (OUT/'s0_e5_dev').glob('*/audit.json')),
        frozen_modules=all(read(p)['pass_unchanged'] for p in (OUT/'jobs').glob('*/frozen_modules.json')),
        source_unchanged=read(OUT/'frozen_check.json')['pass_unchanged'],
        validation_not_generated=not (OUT/'s4_validation').exists(),
        engineering_checks=all(read(OUT/'engineering_checks.json')[k] for k in ['step0_exact','zero_mask_exact','outside_exact']))
    assert all(checks.values())
    write(OUT/'completion_check.json',dict(checks=checks,pass_complete=True))
    training_updates=sum(len(read(p)) for p in OUT.glob('S2*/training_history.json'))
    failed=[stage for stage in ['S0','S1','S2a','S2b','S3','S3b','S4'] if state[stage]=='fail']
    decision(training_updates=training_updates,training_updates_per_model=training_updates,
        total_optimizer_updates=2*training_updates,
        stop_stage=failed[-1] if state.get('stopped') and failed else None,
        mask_valid_n=audit['valid_n'],mask_total_n=128,mask_valid_rate=audit['valid_rate'],
        AI_visual_task_match=review['task_match_pass'],paper_method_sufficient=state['S4']=='pass',
        git_commit=commit(),result_claim='stage-gated feasibility experiment; preserve user S1 exception and actual negative gates')
    write(OUT/'run_manifest.json',dict(git_commit=commit(),protocol_sha256=sha(OUT/'protocol.json'),
        artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
            and p.name not in ['run_manifest.json'] and p.suffix not in ['.gz','.tmp','.log','.err']},
        server_summary_documents_created=False,final_narrative_location='local docs only'))
    print('FINAL',json.dumps(read(OUT/'final_decision.json')),flush=True)

def bundle(label):
    target=OUT/(label+'_results.tar.gz')
    with tarfile.open(target,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.png','.log','.err']:
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('BUNDLE',target,sha(target),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--masks',action='store_true');p.add_argument('--finalize',action='store_true')
    p.add_argument('--amend',action='store_true')
    p.add_argument('--previews',action='store_true');p.add_argument('--import-review',action='store_true')
    p.add_argument('--bundle');p.add_argument('--jobs',nargs='*',default=[]);args=p.parse_args()
    if args.masks:masks()
    if args.amend:apply_amendments()
    if args.previews:
        from tools.e34_sarr_s0 import panels
        panels(read(OUT/'splits/dev128.json'))
    if args.import_review:import_review()
    if args.jobs:accounting(args.jobs)
    if args.finalize:finalize()
    if args.bundle:bundle(args.bundle)
