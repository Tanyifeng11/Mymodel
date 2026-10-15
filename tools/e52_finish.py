"""核对完整性、统计边界和资源；只生成机器数据与审阅包，中文报告留在本地。"""
import csv
import datetime
import re
import subprocess
import tarfile
from tools.e52_protocol import *
from tools.e52_eval import paired,get_delta

def ledger():
    jobs=read(OUT/'resources/jobs.json');ids=','.join(str(r['id']) for r in jobs)
    fields=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','AllocCPUS','AllocTRES','Start','End']
    value=subprocess.check_output(['sacct','-X','-j',ids,'--format='+','.join(fields),'-P','-n'],text=True)
    byid={str(x['id']):x for x in jobs};records=[]
    for line in value.splitlines():
        parts=line.split('|')
        if len(parts)<len(fields):continue
        rec=dict(zip(fields,parts));identity=rec['JobIDRaw']
        if identity not in byid:continue
        seconds=int(rec['ElapsedRaw']);cpus=int(rec['AllocCPUS'])
        match=re.search(r'(?:^|,)gres/gpu=(\d+)',rec['AllocTRES']);gpus=int(match.group(1)) if match else 0
        records.append(dict(id=int(identity),name=byid[identity]['name'],state=rec['State'],exit_code=rec['ExitCode'],
            elapsed_seconds=seconds,allocated_cpus=cpus,allocated_gpus=gpus,
            gpu_hours=gpus*seconds/3600,cpu_core_hours=cpus*seconds/3600,start=rec['Start'],end=rec['End']))
    write(OUT/'resources/accounting.json',records)
    if records:
        with (OUT/'resources/accounting.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(OUT/'failures/failed_cancelled_jobs.json',[r for r in records if r['state'].split()[0] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']])
    return dict(jobs=len(records),gpu_hours=sum(r['gpu_hours'] for r in records),
        cpu_core_hours=sum(r['cpu_core_hours'] for r in records),
        final_accounting_pending=any(r['state'] in ['RUNNING','COMPLETING','PENDING'] for r in records))

def finish():
    import numpy as np
    from PIL import Image,ImageDraw
    assert read(OUT/'audit/p0_complete.json')['passed']
    expected={**read(OUT/'manifest/input_sha256.json'),**read(OUT/'manifest/model_file_sha256.json'),
        **read(OUT/'manifest/natural_input_sha256.json'),**read(OUT/'manifest/synthetic_pilot_sha256.json'),
        **read(OUT/'manifest/synthetic_lock_sha256.json')}
    assert all(sha(p)==h for p,h in expected.items()),'冻结输入或模型文件改变'
    summary=read(OUT/'metrics/lock/summary.json');natural=read(OUT/'metrics/natural/summary.json')
    records=read(OUT/'metrics/lock/per_image.json');provenance=read(OUT/'audit/token_provenance.json')
    g=summary['primary']['G'];d=summary['primary']['D'];direction=summary['direction_amplitude']
    sham=summary['sham'];holm=summary['holm_p']
    g_supported=(g['ci95'][1]<0 and holm.get('G',1)<.05 and direction['ci95'][0]>0
        and summary['direction_random_label']['p_upper']<.05 and abs(g['mean'])>abs(sham['mean'])
        and summary['geometry_minus_sham']['ci95'][1]<0)
    hg='SUPPORTED_IN_SYNTHETIC_DOMAIN' if g_supported else 'NOT_SUPPORTED_IN_TESTED_GEOMETRY_RANGE' if g['ci95'][0]>=0 else 'INCONCLUSIVE'
    hc='NOT_IDENTIFIABLE' if not provenance['oracle_valid'] else 'INCONCLUSIVE'
    correspondence=None
    if provenance['oracle_valid']:
        byid={r['id']:{x['arm']:x for x in records if x['id']==r['id']} for r in read(OUT/'splits/lock.json')}
        comparisons={control:{key:paired([get_delta(index,'ORACLE',control,key) for index in byid.values()])
            for key in ['f1_3px','texture_retrieval_correct','texture_margin']} for control in ['G0T0','WRONG','RANDOM']}
        c=summary['primary']['C'];q=comparisons['G0T0']['texture_retrieval_correct']
        correspondence=dict(comparisons=comparisons,energy_exclusion='requires inspection of actual injection RMS, coverage and entropies; no automatic strong causal promotion')
        if c['ci95'][1]<=0:hc='NOT_SUPPORTED_WITH_VALIDATED_ORACLE'
        elif c['ci95'][0]>0 and holm.get('C',1)<.05 and q['ci95'][0]>=-.05:
            hc='STRUCTURE_TEXTURE_SIGNATURE_PENDING_ENERGY_REVIEW'
    # 时间交互与反馈分解是证据；不能单凭 RMS 差自动宣布独立动力学根因。
    hd='INCONCLUSIVE'
    d_signature=bool(d['ci95'] is not None and not(d['ci95'][0]<=0<=d['ci95'][1]) and holm.get('D',1)<.05)
    any_go=any(v['passed'] for v in natural['model_gates'].values())
    status=dict(H_G=hg,H_C=hc,H_D=hd,MODEL='NOT_EVALUATED_FOR_FULL_PROMOTION' if any_go else 'NO_GO',
        time_interaction_supported=d_signature,correspondence=correspondence,
        inference='hypotheses can overlap; no forced unique root cause; manual causal review still needed',
        limitations=['synthetic reference domain shift','independent human reference/texture review not_run',
            'synthetic texture retrieval is geometry sensitive','Writer pretraining and some historical consumption logs unknown',
            'natural geometric/UV oracle not available; natural96 confirms fixed time/state contrasts only'],
        confidence='natural96 is a historically audited cleaner validation pool, not proven full-model never-seen')
    write(OUT/'mechanism_decision.json',status)
    decision(P0='pass',P1='objective_QC_pass_human_review_not_run',P2='complete',
        P3='complete' if provenance['oracle_valid'] else 'NOT_IDENTIFIABLE_WITH_CURRENT_WRITER_TOKENS',
        P4='complete',P5='complete',**{k:status[k] for k in ['H_G','H_C','H_D','MODEL']})
    write(OUT/'audit/final_frozen_inputs.json',dict(unchanged=True,n=len(expected),commit=commit()))
    resources=ledger()
    # 全量固定顺序图板，保存输入和输出；此步骤不等于独立人工盲评。
    for stage in ['pilot','lock','natural','probe']:
        rows=read(OUT/'splits'/(stage+'.json'))
        arms=list(read(OUT/'metrics'/stage/'summary.json')['means']) if (OUT/'metrics'/stage/'summary.json').exists() else ['E5','B1']
        root=OUT/'figures'/stage/'outputs';root.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(rows),4):
            page=rows[start:start+4];canvas=Image.new('RGB',((2+len(arms))*128,len(page)*196),'white');draw=ImageDraw.Draw(canvas)
            for y,row in enumerate(page):
                items=[('Sketch',row['sketch']),('BaseRef',row['reference'])]+[(a,image_path(stage,a,row['id'])) for a in arms]
                for x,(label,path) in enumerate(items):
                    draw.text((x*128+2,y*196+2),row['id']+' '+label,fill='black')
                    canvas.paste(Image.open(path).convert('RGB').resize((128,171)),(x*128,y*196+23))
            canvas.save(root/('page%03d.png'%start))
        if stage in ['natural','probe']:
            for row in rows:
                for key in ['gt','sketch','reference']:
                    path=OUT/'review_inputs'/row['id']/(key+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                    Image.open(row[key]).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR).save(path)
    write(OUT/'complete.json',dict(complete=True,scientific_training_updates=0,resources=resources,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit(),
        manual_final_interpretation='pending local audit',raw_pt_tensors='retained on server, excluded from portable review archive'))
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.pt','.gz'] and not p.name.startswith('e52_final_review.')}
    write(OUT/'manifest/review_sha256.json',files)
    archive=OUT/'e52_final_review.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=4) as tar:
        for name in sorted(files):tar.add(OUT/name,arcname=name)
        tar.add(OUT/'manifest/review_sha256.json',arcname='manifest/review_sha256.json')
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print('E52 FINAL ARCHIVE',archive,sha(archive),archive.stat().st_size,flush=True)

if __name__=='__main__':finish()
