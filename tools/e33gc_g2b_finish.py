"""汇总既有产物和Slurm用量；不训练、不补选病例、不修改原结果。"""
import csv,json,re,subprocess
from tools.e33gc_g2b_protocol import *

def accounting():
    ids=sorted({re.search(r'job_(\d+)',p.name)[1] for p in OUT.glob('job_*.log')})
    command=['sacct','-j',','.join(ids),'--parsable2','--noheader','--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,NodeList,ExitCode']
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    (OUT/'protocol/slurm_accounting.txt').write_text(result.stdout,encoding='utf-8')
    jobs=[]
    for line in result.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<7 or not fields[0].isdigit():continue
        jid,name,state,seconds,tres,nodes,exitcode=fields[:7]
        gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        count=int(gpu[1]) if gpu else 0
        jobs.append(dict(job_id=jid,state=state,elapsed_seconds=int(seconds),allocated_GPUs=count,
            GPU_hours=int(seconds)*count/3600,node=nodes,exit_code=exitcode,allocated_TRES=tres))
    value=dict(jobs=jobs,GPU_hours=sum(v['GPU_hours'] for v in jobs),cap=6,
        code_commit=commit(),scope='all Slurm jobs with logs in this new experiment directory; allocated GPU wall time',
        submission_scripts=['submit/e33gc_g2b_cpu.sh','submit/e33gc_g2b_gpu.sh'])
    submission_log=OUT/'protocol/gpu_submissions.jsonl'
    value['GPU_submissions']=[json.loads(line) for line in submission_log.read_text().splitlines()] if submission_log.exists() else []
    write(OUT/'protocol/commands_and_gpu_budget.json',value)
    return value

def run():
    init();d=read(OUT/'decision_summary.json');budget=accounting()
    fit=read(OUT/'protocol/g2b_fit_probe_ids.json')['fit']
    metrics=read(OUT/'G2b_eval/identity_metrics.json')
    rows=[]
    for case in metrics:
        for method,value in case['methods'].items():
            row=dict(id=case['id'],label=case['label'],group=case['group'],method=method)
            row.update({k:v for k,v in value['DFT'].items() if k not in ['blocks','fixed_boxes']})
            for key in sorted({k for m in value['arms'].values() for k,v in m.items() if isinstance(v,(float,int))}):
                values=[m[key] for m in value['arms'].values() if isinstance(m.get(key),(int,float))]
                row[key+'_arm_mean']=sum(values)/len(values)
            row.update({'E26_'+k:v for k,v in value['E26'].items()})
            rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'result_table.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    proof=[read(OUT/'G2b_smoke/cache'/r['id']/'proof.json') for r in fit]
    smoke=read(OUT/'G2b_smoke/smoke_audit.json')
    training=read(OUT/'G2b_train/training_complete.json')
    updates=[json.loads(v)['update'] for v in (OUT/'G2b_train/updates.jsonl').read_text().splitlines()]
    upper_needed=d.get('g2b_AI_amended_fit_pass') is False
    checks=dict(original_results_unchanged=read(OUT/'frozen_check.json')['pass_unchanged'],
        independent_split_frozen=d['gc_controlled_dev_new_frozen'],
        real64_input_only_two_AI_reviews=(OUT/'G0a_real_input_annotations/real_input_eligibility.json').exists(),
        G1a_hard_stop_or_frozen_contract=d.get('main_reference_derived_G2b')=='not_run' or d['g2b_route']=='A',
        fixed4_identities=len(fit)==4,
        exact12_zero_arms=all(a['exact'] for p in proof for a in p['arms']) and sum(len(p['arms']) for p in proof)==12,
        all_three_losses_autograd=smoke['pass_autograd'] and set(smoke['component_gradients'])=={'rgb','orientation_pair','outside'},
        frozen_backward_precision=sha(OUT/'protocol/backward_precision.json')==smoke['backward_precision_sha256'],
        smoke_fully_completed=read(OUT/'G2b_smoke/smoke_complete.json')['smoke_audit_sha256']==sha(OUT/'G2b_smoke/smoke_audit.json'),
        frozen_input_target_hashes=all(sha(p)==h for p,h in read(OUT/'protocol/input_target_contract.json')['files'].items()),
        fixed160_updates=training['updates']==160 and updates==list(range(1,161)),
        endpoint_checkpoint_hash=sha(OUT/'G2b_train/checkpoint_step160.pt')==training['checkpoint_sha256'],
        frozen_model_modules=(OUT/'G2b_smoke/frozen_modules.json').exists() and (OUT/'G2b_train/frozen_modules.json').exists(),
        independent_DFT_and_E26=(OUT/'G2b_eval/identity_metrics.json').exists(),
        blind_AI_endpoint_review=(OUT/'G2b_eval/two_AI_blind_review.json').exists(),
        required_upper_complete=(not upper_needed) or ((OUT/'G2b_upper_bound_if_needed/two_AI_blind_review.json').exists() and
            read(OUT/'G2b_upper_bound_if_needed/diagnosis.json')['updates']==80),
        geometry_comparison=(OUT/'G0a_simple_geometry/comparison.json').exists(),
        GPU_budget_pass=budget['GPU_hours']<=6,
        no_later_stages=not any(d[k] for k in ['g3_authorized','g4_authorized','g5_authorized']))
    write(OUT/'completion_check.json',dict(checks=checks,protocol_execution_complete=all(checks.values()),
        scientific_success=d.get('g2b_fit_pass'),human_certification=None,code_commit=commit(),
        AI_review_amendment=CONFIG['review'],local_report_transferred_to_server=False))
    verify_frozen()
    code_files=list(Path('tools').glob('e33gc_g2b_*.py'))+list(Path('tools').glob('e33gc_g2b_*reviews.json'))+[
        Path('data/e33gc_g2b_renderer.py'),Path('submit/e33gc_g2b_cpu.sh'),Path('submit/e33gc_g2b_gpu.sh')]
    write(OUT/'protocol/code_hashes.json',{str(p):sha(p) for p in code_files})
    write(OUT/'artifact_manifest.json',{str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ['.gz'] and p.name!='artifact_manifest.json'})
    bundle('final')
    assert all(checks.values()),checks

if __name__=='__main__':run()
