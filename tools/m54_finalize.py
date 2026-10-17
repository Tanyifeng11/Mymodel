"""服务器只生成可验证数据包；中文科研结论及科学图在本地交付。"""
import csv
import json
import re
import subprocess
import tarfile
from datetime import datetime,timezone
from tools.m54_protocol import *


def package(path,files):
    manifest={name:dict(sha256=sha(source),bytes=source.stat().st_size) for name,source in files.items()}
    manifest_path=path.with_name(path.name+'.manifest.json');write(manifest_path,manifest)
    with tarfile.open(path,'w:gz') as archive:
        for name,source in files.items():archive.add(source,arcname=name,recursive=False)
        archive.add(manifest_path,arcname='portable_manifest.json')
    path.with_name(path.name+'.sha256').write_text(sha(path)+'  '+path.name+'\n',encoding='utf-8')


def input_review_package(stage):
    from tools.m54_prepare import panels
    rows=read(split_path(stage));panels(rows,stage+'_active')
    files={'splits/'+stage+'.json':split_path(stage)}
    for source in (OUT/'audit/input_review'/(stage+'_active')).glob('*.png'):
        files['input_review/'+source.name]=source
    if (OUT/'active_inputs.json').exists():files['active_inputs.json']=OUT/'active_inputs.json'
    package(OUT/('m54_'+stage+'_input_review.tar.gz'),files)


def seed_aggregate(best):
    from tools.m54_eval import assess,KEYS
    stages=[result_stage('natural',2400,seed) for seed in [42,123,456]]
    sources=[read(OUT/stage/'per_image.json') for stage in stages]
    ids=[r['id'] for r in read(split_path('natural'))];records=[]
    for arm in ['E5','B1',best]:
        indexes=[{r['id']:r for r in source if r['variant']==arm} for source in sources]
        assert all(set(index)==set(ids) for index in indexes)
        for identity in ids:
            entries=[index[identity] for index in indexes];record=dict(entries[0])
            for key in KEYS:
                values=[r[key] for r in entries]
                record[key]=sum(values)/3 if all(v is not None for v in values) else None
            records.append(record)
    write(OUT/'seed_aggregate/per_image.json',records);gates=assess(records,'seed_aggregate')
    per_seed={str(seed):read(OUT/stage/'gates.json')[best] for seed,stage in zip([42,123,456],stages)}
    value=dict(best1=best,seeds=[42,123,456],identities=len(ids),independent_statistical_units=len(ids),
        estimator='average paired metrics across three seeds within identity, then identity paired bootstrap10000/seed32042',
        per_seed_gates=per_seed,aggregate_gates=gates[best],all_three_seeds_pass=all(g['passed'] for g in per_seed.values()),
        independent_human_review='NOT_RUN',independent_blind_review='NOT_RUN')
    write(OUT/'seed_aggregate/summary.json',value);return value


def verify_outputs():
    import torch
    from tools.e44_runtime import digest
    from tools.m54_cleanup import retired_traces
    retired=retired_traces()
    entries=[]
    # 每个完整条件逐身份验证真实50步epsilon与初末latent，失败/部分输出单独保留不算完成。
    for done in sorted(OUT.glob('*/'+'*_complete.json')):
        if done.name in ['evaluation_complete.json','prepare_complete.json','final_complete.json']:continue
        value=read(done)
        if 'arm' not in value or 'n' not in value:continue
        arm=value['arm'];stage=done.parent.name;source=stage.split('_u')[0].split('_s')[0]
        rows=read(split_path(source));assert len(rows)==value['n']
        if 'split_sha256' in value:assert value['split_sha256']==sha(split_path(source))
        for row in rows:
            image=image_path(stage,arm,row['id']);meta=read(image.with_suffix('.json'));trace=image.with_suffix('.pt')
            assert sha(image)==meta['png_sha256']
            assert not meta['gt_used_in_inference'] and meta['seed']==value['seed']
            if trace.exists():
                assert sha(trace)==meta['diagnostics_sha256']
                raw=torch.load(trace,map_location='cpu',weights_only=False)
                assert digest(raw['initial_latent'])==meta['initial_sha256'] and digest(raw['final_latent'])==meta['final_sha256']
                assert raw['conditional_epsilon'].shape[0]==50 and raw['unconditional_epsilon'].shape[0]==50
                assert all(torch.isfinite(raw[key]).all() for key in ['initial_latent','final_latent','conditional_epsilon','unconditional_epsilon'])
                trace_status='PRESENT_REVERIFIED_FULL50'
            else:
                prior=retired[(stage,arm,row['id'])]
                assert prior['trace_sha256']==meta['diagnostics_sha256'] and prior['png_sha256']==sha(image)
                assert prior['frames']==50 and prior['seed']==value['seed']
                trace_status='VERIFIED_FULL50_BEFORE_USER_AUTHORIZED_DELETION'
            effects=read(image.with_name(image.stem+'_effects.json'));assert len(effects)==50
            assert [r['step'] for r in effects]==list(range(50))
            entries.append(dict(stage=stage,arm=arm,id=row['id'],trace_sha256=meta['diagnostics_sha256'],
                trace_present=trace.exists(),trace_status=trace_status,png_sha256=sha(image),frames=50,seed=value['seed']))
    write(OUT/'audit/output_integrity.json',dict(passed=True,conditions=len(entries),entries=entries,
        raw_traces_present=sum(x['trace_present'] for x in entries),
        raw_traces_verified_before_user_cleanup=sum(not x['trace_present'] for x in entries)))
    for manifest in [OUT/'manifests/model_file_hashes.json',OUT/'manifests/input_hashes.json',OUT/'manifests/input_hashes_r1.json']:
        if manifest.exists():assert all(sha(path)==expected for path,expected in read(manifest).items()),('FROZEN_SOURCE_CHANGED',str(manifest))
    for arm in TRAIN_ARMS:
        folder=OUT/'training'/arm
        if not folder.exists():continue
        lines=[]
        for checkpoint in sorted((folder/'checkpoints').glob('*.pt')):
            expected=read(checkpoint.with_suffix('.json'));actual=sha(checkpoint);assert actual==expected['sha256']
            lines.append(actual+'  '+checkpoint.name)
        (folder/'checkpoint.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        history=read(folder/'history.json')
        (folder/'training.log').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in history)+'\n',encoding='utf-8')
    return len(entries)


def finalize():
    state=read(OUT/'decision.json');assert state['natural']=='COMPLETE_GO' and state['seeds']=='COMPLETE'
    best=read(OUT/'selection/best1.json')['selected'][0];replication=seed_aggregate(best)
    count=verify_outputs()
    records=[]
    for path in OUT.glob('*/per_image.json'):
        records.extend(dict(r,stage=path.parent.name) for r in read(path))
    for arm in sorted({r['variant'] for r in records}):
        folder=OUT/'modules'/arm;folder.mkdir(parents=True,exist_ok=True)
        selected=[r for r in records if r['variant']==arm];keys=sorted({k for r in selected for k in r})
        for name in ['metrics.csv','per_image.csv']:
            with (folder/name).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(selected)
        if not (folder/'config.yaml').exists():
            write(folder/'config.yaml',dict(arm=arm,components=recipe_components(arm,2400),protocol_sha256=sha(protocol_path())))
            (folder/'module_card.md').write_text('# '+arm+'\n\n冻结机制适配，未声称完整论文复现。实际参数、更新预算及逐身份指标见同目录文件。\n',encoding='utf-8')
    write(OUT/'audit/final_complete.json',dict(passed=True,verified_outputs=count,best1=best,
        replication_all_three_seeds_pass=replication['all_three_seeds_pass'],
        semantic_oracle_proven=False,independent_human_review='NOT_RUN',independent_blind_review='NOT_RUN',
        final_job_accounting='requires sacct after this FINAL job COMPLETED; separate SHA256-verified resource package',
        raw_tensors='retained on server; portable package excludes .pt',commit=commit()))
    files={}
    for source in OUT.rglob('*'):
        if source.is_file() and source.suffix in ['.json','.yaml','.md','.csv','.jsonl','.png','.log','.err','.psv','.sha256']:
            if '.tar.gz' not in source.name:files[str(source.relative_to(OUT))]=source
    # 审阅者可实际看到所有已执行split的输入，不能仅凭caption判断reference纹样。
    for stage in ['pilot','dev32','lock','natural','oracle_diagnostic']:
        if not any(name.startswith(stage+'/') or name.startswith(stage+'_u') or name.startswith(stage+'_combo') for name in files):continue
        for row in read(split_path(stage)):
            for key in ['sketch','reference','gt']:
                source=Path(row[key]);files['inputs/'+row['id']+'/'+key+source.suffix]=source
    package(OUT/'m54_portable_review.tar.gz',files)
    print('M54 FINAL PACKAGE READY',count,flush=True)


def accounting():
    jobs=read(OUT/'resources/jobs.json');assert any(j['name']=='FINAL' for j in jobs)
    fields='JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'
    raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),'--format='+fields],text=True)
    rows=list(csv.DictReader(raw.splitlines(),delimiter='|'));index={int(r['JobIDRaw']):r for r in rows if r['JobIDRaw'].isdigit()}
    assert set(index)=={j['id'] for j in jobs},'MISSING_ACCOUNTING_JOB'
    terminal={'COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
    ledger=[]
    for job in jobs:
        row=index[job['id']];status=row['State'].split()[0];assert status in terminal,('JOB_STILL_ACTIVE',job['id'],status)
        if job['name']=='FINAL':assert status=='COMPLETED' and row['ExitCode']=='0:0'
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',row['AllocTRES'])
        if match:gpus=int(match.group(1))
        else:
            typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)',row['AllocTRES']);gpus=sum(map(int,typed))
        elapsed=int(row['ElapsedRaw']);cpus=int(row['AllocCPUS'])
        ledger.append(dict(**job,slurm=row,actual_gpu_count=gpus,actual_cpu_count=cpus,
            elapsed_seconds=elapsed,gpu_hours=gpus*elapsed/3600,cpu_hours=cpus*elapsed/3600))
    (OUT/'resources/accounting_final.psv').write_text(raw,encoding='utf-8')
    write(OUT/'resources/accounting_final.json',dict(utc=datetime.now(timezone.utc).isoformat(),records=ledger,
        includes_all_success_failure_cancelled_allocations=True,steps_excluded_to_avoid_double_count=True,
        gpu_hours=sum(r['gpu_hours'] for r in ledger),cpu_hours=sum(r['cpu_hours'] for r in ledger),
        submission_failures_file='submission_failures.json if present',source='sacct allocation records after FINAL completed'))
    package(OUT/'m54_resource_ledger.tar.gz',{name:OUT/'resources'/name for name in ['jobs.json','accounting_final.psv','accounting_final.json']})
    print('M54 FINAL ACCOUNTING READY',len(ledger),flush=True)
