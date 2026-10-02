"""E33 Step0完整划分审计；未通过变换完整性时禁止开始训练。"""
import argparse
import hashlib
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw
from garment_mask_utils import estimate_cloth_foreground_mask
from data.e32_target_pseudogt import image_at
from data.e32_field_dataset import cache_path
from data.e33_self_reference import select_reference
from data.e33_interventions import audit_intervention,transform
from tools.e32_common import DATASET,WEIGHTS,read,write,sha,bootstrap,frozen_manifest,finish_frozen,git_commit
from tools.e33_protocol import OUT,E32,PROTOCOL,GROUPS


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def worker(task):
    row,out,e32,dataset=task
    cv2.setNumThreads(1)
    destination=out/'transform_integrity/cases'/(row['id']+'.json')
    if destination.exists():
        v=read(destination)
        assert v['protocol_sha256']==protocol_hash()
        return v
    target=image_at(dataset/row['target']);sketch=image_at(dataset/row['sketch'])
    foreground=np.asarray(estimate_cloth_foreground_mask(target,384,512)[0])>127
    with np.load(cache_path(e32,row['id'])) as f:geometry=np.asarray(f['supervision_geometry'],np.float32)
    strict=select_reference(target,sketch,foreground,geometry,'strict')
    relaxed=strict if strict['controlled_available'] else select_reference(target,sketch,foreground,geometry,'relaxed')
    no_structure=select_reference(target,sketch,foreground,geometry,'strict',False)
    v=dict(id=row['id'],group=row['group'],protocol_sha256=protocol_hash(),
           strict=strict,relaxed=relaxed,no_structural_exclusion=no_structure,interventions={})
    # 审计strict及relaxed候选，不在看到完整性结果后更换patch。
    for mode,candidate in [('strict',strict),('relaxed',relaxed)]:
        if not candidate['controlled_available']:continue
        rgb=np.asarray(target.crop(tuple(candidate['box'])))
        seed=int(hashlib.sha256(('E33/audit/'+row['id']).encode()).hexdigest()[:8],16)
        v['interventions'][mode]={arm:dict(clean=audit_intervention(rgb,arm),
             noisy=[audit_intervention(rgb,arm,(seed+j)%2**32) for j in range(PROTOCOL['integrity']['noisy_trials'])])
             for arm in ('rot90','scale_up','scale_down')}
    write(destination,v)
    return v


def summarize(rows,mode):
    selected=[r for r in rows if r[mode]['controlled_available']]
    report=dict(total_cases=len(rows),available_cases=len(selected),
                controlled_available_rate=len(selected)/len(rows),interventions={})
    for arm in ('rot90','scale_up','scale_down'):
        clean=[r['interventions'][mode][arm]['clean'] for r in selected]
        noisy=[r['interventions'][mode][arm]['noisy'] for r in selected]
        report['interventions'][arm]=dict(denominator=len(selected),
            clean_valid=bootstrap([float(x['valid']) for x in clean]),
            noisy_valid=bootstrap([float(all(x['valid'] for x in xs)) for xs in noisy]),
            clean_orientation_error_deg=bootstrap([x['orientation_response_error_deg'] for x in clean if 'orientation_response_error_deg' in x]),
            clean_period_error_log2=bootstrap([x['period_response_error_log2'] for x in clean if 'period_response_error_log2' in x]))
    return report


def contact_sheets(rows,mode,out,dataset):
    # 固定hash选32个available候选，不按完整性成功挑图。
    chosen=sorted([r for r in rows if r[mode]['controlled_available']],
                  key=lambda r:hashlib.sha256(('E33/visual/'+r['id']).encode()).hexdigest())[:32]
    splits=read(out/'split_manifest.json');lookup={r['id']:r for g in GROUPS for r in splits[g]}
    write(out/'audits/visual_selection.json',dict(ids=[r['id'] for r in chosen],mode=mode,
          selection='fixed hash32 available candidates, no integrity success filtering'))
    for offset in range(0,len(chosen),8):
        canvas=Image.new('RGB',(1280,8*220),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(chosen[offset:offset+8]):
            y=j*220;source=lookup[row['id']];target=image_at(dataset/source['target']);sketch=image_at(dataset/source['sketch'])
            q=row[mode];box=q['box'];marked=target.copy();ImageDraw.Draw(marked).rectangle(tuple(box),outline='red',width=3)
            rgb=np.asarray(target.crop(tuple(box)))
            items=[('target selected',marked),('sketch',sketch),('R0',Image.fromarray(rgb))]
            for arm in ('rot90','scale_up','scale_down'):
                pixels,_=transform(rgb,arm);a=row['interventions'][mode][arm]['clean']
                label='%s %s o%.1f p%.3f'%(arm,'PASS' if a['valid'] else 'FAIL',a.get('orientation_response_error_deg',-1),a.get('period_response_error_log2',-1))
                items.append((label,Image.fromarray(pixels)))
            for col,(label,im) in enumerate(items):
                im.thumbnail((196,170));canvas.paste(im,(col*212,y+35))
                draw.text((col*212,y+15),label,fill='black')
            draw.text((0,y),row['id']+' '+row['group'],fill='black')
        canvas.save(out/'audits'/('self_reference_contact_%02d.png'%(offset//8)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--e32',type=Path,default=E32)
    parser.add_argument('--dataset',type=Path,default=DATASET);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    assert read(args.e32/'completion_check.json')['experiment_complete']
    if (out/'protocol.json').exists():assert read(out/'protocol.json')['protocol']==PROTOCOL
    else:
        write(out/'protocol.json',dict(protocol=PROTOCOL,protocol_sha256=protocol_hash(),git_commit=git_commit(),
              e32_split_sha256=sha(args.e32/'split_manifest.json')))
        shutil.copy2(args.e32/'split_manifest.json',out/'split_manifest.json')
        frozen_manifest(out,WEIGHTS)
    split=read(out/'split_manifest.json')
    assert sha(out/'split_manifest.json')==sha(args.e32/'split_manifest.json')
    assert [len(split[g]) for g in GROUPS]==[45126,256,8,10]
    tasks=[(dict(row,group=g),out,args.e32,args.dataset) for g in GROUPS for row in split[g]]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,v in enumerate(pool.map(worker,tasks,chunksize=4),1):
            rows.append(v)
            if i%128==0 or i==len(tasks):
                write(out/'audits/progress.json',dict(completed=i,total=len(tasks),git_commit=git_commit()))
                print('[E33 integrity]',i,'/',len(tasks),flush=True)
    strict_train=summarize([r for r in rows if r['group']=='train'],'strict')
    # 选一次统一规则，held-out完全沿用；同一个样本strict合法时保留strict top1。
    mode='relaxed' if strict_train['controlled_available_rate']<.50 else 'strict'
    coverage={g:{m:summarize([r for r in rows if r['group']==g],m) for m in ('strict','relaxed')} for g in GROUPS}
    checks={}
    for g,min_n in [('train',32),('dev',8)]:
        group=coverage[g][mode];checks[g+'/minimum_probes']=group['available_cases']>=min_n
        for arm in ('rot90','scale_up','scale_down'):
            threshold=.90 if arm=='rot90' else .85
            for setting in ('clean_valid','noisy_valid'):
                value=group['interventions'][arm][setting]['mean']
                checks[g+'/'+arm+'/'+setting]=value is not None and value>=threshold
    decision=dict(transform_integrity_pass=all(checks.values()),self_reference_coverage={g:coverage[g][mode]['controlled_available_rate'] for g in GROUPS},
        selected_candidate_mode=mode,controlled_geometry_pass=None,real_pair_geometry_pass=None,
        identity_pass=None,scaffold_pass=None,generation_pass=None,controlled_vs_real_diagnosis=None,
        next_route='P0_prior_training' if all(checks.values()) else 'intervention_generator_integrity_failed_no_training')
    write(out/'transform_integrity/summary.json',dict(coverage=coverage,checks=checks,**{'pass':all(checks.values())},
          no_structural_exclusion_available={g:sum(r['no_structural_exclusion']['controlled_available'] for r in rows if r['group']==g) for g in GROUPS},
          relaxation_trigger='strict train coverage only',selected_mode=mode))
    write(out/'decision_summary.json',decision)
    for stage in ('P0_prior','P1_controlled','P2_curriculum','P3_identity','C_scaffold','D_generation'):
        write(out/stage/'status.json',dict(status='pending_gate_pass' if all(checks.values()) else 'not_run_transform_integrity_failed'))
    contact_sheets(rows,mode,out,args.dataset);finish_frozen(out)
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    write(out/'artifact_manifest.json',dict(files=manifest,training_steps=0))
    print('[E33 decision]',json.dumps(decision),flush=True)


if __name__=='__main__':main()
