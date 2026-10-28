"""汇总统一评价、保留原确认/探索证据，打包新增结果；不生成人类评分。"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from tools.t3_p0p1 import PROJECT,ROOT,SOURCE,SUP,CONF,read,write,sha,csv_write
from tools.t3_p0p1_review import METRICS,paired_stats,holm

ARMS=['E0','E1','B','T1','T2','C','T3','BF-Fashion','ControlNet+IP-Adapter']


def records(arm):
    folder=ROOT/'metrics'/arm
    assert read(folder/'complete.json')['complete'],('INCOMPLETE_METRICS',arm)
    return [read(p) for condition in ['original','swapped'] for p in sorted((folder/condition).glob('*.json'))]


def aggregate():
    history_audit()
    data={arm:records(arm) for arm in ARMS}
    means=[];response=[];contrasts=[];replay=[]
    for arm,rows in data.items():
        original=[r for r in rows if r['condition']=='original']
        assert len(original)==96 and len({r['id'] for r in original})==96
        for metric,direction in METRICS.items():
            values=[r[metric] for r in original if r.get(metric) is not None]
            means.append(dict(model=arm,input_modalities='sketch+reference' if arm=='BF-Fashion' else 'text+sketch+reference',
                condition='original',metric=metric,mean=float(np.mean(values)),n=len(values),planned_n=96,
                direction='up' if direction>0 else 'down',protocol='uniform CUDA float32 metrics, TF32 disabled',
                output_native_size='256x256 then directBILINEAR384x512' if arm=='BF-Fashion' else '384x512'))
        if arm in ['B','T1','T2','C','T3']:
            legacy={r['id']:r for r in read(SUP/'evaluation/P2b'/arm/'per_identity_metrics.json')}
            for metric in METRICS:
                difference=[abs(r[metric]-legacy[r['id']][metric]) for r in original if r.get(metric) is not None]
                replay.append(dict(model=arm,metric=metric,n=len(difference),max_abs_difference=max(difference),
                    strict_2e_6_passed=max(difference)<=2e-6,
                    policy='external table uses uniformly recomputed values for all methods; historical P0 values preserved'))
        swapped=[r for r in rows if r['condition']=='swapped']
        for metric in ['dists_margin','color_margin','iou','edge_f1','leak']:
            values=[r[metric] for r in swapped if r.get(metric) is not None]
            if not values:continue
            stat=paired_stats(values,'external summary/'+arm+'/'+metric)
            response.append(dict(model=arm,condition='swapped',metric=metric,mean=stat['delta'],
                ci_low=stat['ci_low'],ci_high=stat['ci_high'],n=len(values),planned_n=32,
                interpretation='patch-set response, not motif accuracy' if metric=='dists_margin' else 'legacy proxy',
                CI='descriptive unadjusted identity bootstrap95%'))
    t3={r['id']:r for r in data['T3'] if r['condition']=='swapped'}
    for baseline in ['B','T1','T2','C','BF-Fashion','ControlNet+IP-Adapter']:
        other={r['id']:r for r in data[baseline] if r['condition']=='swapped'}
        valid=sorted(i for i in t3 if i in other and t3[i].get('dists_margin') is not None and other[i].get('dists_margin') is not None)
        values=[t3[i]['dists_margin']-other[i]['dists_margin'] for i in valid]
        contrasts.append(dict(baseline=baseline,comparison='T3 minus '+baseline,metric='dists_margin',
            evidence='retrospective P2b32 development comparison; not new independent confirmation',
            **paired_stats(values,'external contrast/'+baseline)))
    for family in [[r for r in contrasts if r['baseline'] in ['B','T1','T2']],
                   [r for r in contrasts if r['baseline']=='C'],
                   [r for r in contrasts if r['baseline'] in ['BF-Fashion','ControlNet+IP-Adapter']]]:
        holm(family)
    csv_write(ROOT/'external_comparison_table.csv',means)
    csv_write(ROOT/'tables/reference_response_summary.csv',response)
    csv_write(ROOT/'tables/reference_response_paired.csv',contrasts)
    csv_write(ROOT/'audit/legacy_six_metric_replay.csv',replay)
    csv_write(ROOT/'external_metrics.csv',[dict(row,source_model=arm) for arm,rows in data.items() for row in rows])
    generation=[]
    for arm in ['E0','E1','BF-Fashion','ControlNet+IP-Adapter']:
        for path in sorted((ROOT/'outputs'/arm).glob('*/*.json')):
            meta=read(path);image=path.with_suffix('.png')
            assert sha(image)==meta['output_sha256'] and meta['GT_used_in_inference'] is False
            generation.append(dict(model=arm,output=str(image),metadata=str(path),**meta))
    assert len(generation)==448
    csv_write(ROOT/'external_generation_manifest.csv',generation)
    registry(ROOT,generation)
    old_evidence(ROOT)
    plots(ROOT,response,contrasts)
    panels(ROOT,data)
    ledger(ROOT)
    failures=[dict(model=arm,**row) for arm,rows in data.items() for row in rows
              if not row.get('valid_generation') or row.get('dists_invalid_reason')]
    write(ROOT/'audit/metric_failures.json',failures)
    write(ROOT/'automatic_results_status.json',dict(P0='COMPLETE',P1_external_BF_CN='COMPLETE',
        original_per_model=96,swapped_per_reference_eligible_model=32,new_images=448,new_training=0,
        human_evaluation='PILOT_MATERIAL_READY_AWAITING_REAL_RESPONSES',human_reviewers=0,
        original_confirmation_decision_unchanged=True,LOTS='see sibling LOTS project and candidate status',
        statistical_scope='single training replicate for new internal methods; historical E0/E1 training unseeded; retrospective development comparisons'))
    print('AGGREGATION_COMPLETE',len(generation),'new images',flush=True)


def history_audit():
    import yaml
    info={}
    for arm,run in [('E0','run-20260614_124544-nwm97z4x'),('E1','run-20260616_055520-lvoy6ojq')]:
        candidates=[PROJECT/'wandb'/run/'files/config.yaml']
        candidates+=list((PROJECT/'output').glob('*/logs/wandb/'+run+'/files/config.yaml'))
        candidates+=list((PROJECT/'output').glob('*/wandb/'+run+'/files/config.yaml'))
        present=next((p for p in candidates if p.is_file()),None)
        value=dict(historical_run=run,training_seed=None,initialization='common Texture Adapter; independent E0/E1 training',
                   global_step_endpoint=28365,source_report='docs/E0_E1_E2_E5_训练方法_文献依据与性能详解_20261018.md',
                   successful_update_count_newly_audited=False,config_rechecked=present is not None)
        if present:
            config=yaml.safe_load(present.read_text())
            selected={key:config[key].get('value') for key in ['seed','gam_init_ckpt','texture_adapter_ckpt',
                'dataset_json_path','layer_group_enabled','num_train_epochs','gradient_accumulation_steps'] if key in config}
            value.update(config_path=str(present),config_sha256=sha(present),actual_selected_fields=selected)
            assert selected.get('seed') is None
        info[arm]=value
    ids={r['id'] for r in read(SUP/'splits/P2b.json')}
    datasets=[]
    for name in ['bf_fashion_train.json','train_bf_texture.json']:
        path=PROJECT/'data'/name
        if path.is_file():
            rows=read(path);training_ids={Path(r[k]).stem for r in rows for k in ['cloth','sketch','texture']}
            overlap=sorted(ids&training_ids);assert not overlap
            datasets.append(dict(path=str(path),sha256=sha(path),rows=len(rows),P2b_identity_overlap=overlap))
    write(ROOT/'audit/E0_E1_history.json',dict(models=info,training_identity_audits=datasets,
          pretrained_exposure='unknown; no claim that pretrained data is disjoint'))


def registry(root,generation):
    inventory=read(root/'audit/server_inventory.json')
    with (root/'model_registry.csv').open(encoding='utf-8-sig',newline='') as stream:internal=list(csv.DictReader(stream))
    internal=[row for row in internal if row['model_id'] not in ['T3-no-body','T3-no-rest']]
    for row in internal:
        if row['model_id'] in ['E0','E1']:
            row.update(training_seed='None (historical W&B configuration)',dataset_split='same P2b96 as B/T1/T2/T3',
                status='same-protocol96 inference and six-metric evaluation completed',
                comparability_notes='historical global-step endpoint28365; successful update count not newly audited; historical500 scores excluded',
                metric_protocol_version='uniform CUDA recomputation, TF32 disabled')
    extension=CONF/'exploratory_extension_20261027'
    for key,label,body,rest in [('T3_no_body_s42','T3-no-body',0.,.03816244801608925),
                              ('T3_no_rest_s42','T3-no-rest',.1,0.)]:
        checkpoint=extension/'training'/key/'checkpoint-final/joint_model.pt'
        verification=read(checkpoint.parent.parent/'checkpoint_verification.json')
        internal.append(dict(model_id=label,role='exploratory_internal_ablation',model_definition=f'body_rgb={body};rest={rest}',
            input_modalities='text+sketch+reference',checkpoint_sha256=verification['sha256'],training_seed=42,
            training_steps=28365,initialization_checkpoint=inventory['models']['E1']['sha256'],
            dataset_split='parent confirmation128 reused after results were seen',inference_config='frozen parent DDIM50 CFG7',
            metric_protocol_version='original T3 confirmation frozen DISTS',result_source=str(extension/'ablation_closeout/statistics/results.json'),
            status='read-only reuse; exploratory seed42',comparability_notes='not independent confirmation; no new training this round'))
    csv_write(root/'model_registry.csv',internal)
    bf=read(root/'audit/BF_provenance.json');cn=read(root/'audit/CNIPA_provenance.json')
    external=[dict(model_id='BF-Fashion',input_modalities='sketch+reference',checkpoint_sha256=bf['checkpoint_sha256'],
        training_seed='historical; not established',training_steps=bf['global_step'],dataset_split='P2b96+swap32',
        inference_config='native256 DDIM200 eta0 scales1.1/1.5 float32; bilinear384x512 for metrics',
        code_version='existing trained BF source; exact SHA in inventory and per-output metadata',
        status='128 outputs and uniform evaluation complete',limitations=bf['limitations']+'; no text condition'),
        dict(model_id='ControlNet+IP-Adapter',input_modalities='text+sketch+reference',
        checkpoint_sha256=json.dumps(cn['choice'],sort_keys=True),training_seed=42,training_steps='CN5000 trained/3000 selected;IPA3000',
        dataset_split='P2b96+swap32',inference_config='DDIM50 CFG7 384x512 CN1.25 IPA0.9; previously frozen selection',
        code_version=cn['source_commit']['output'].strip(),status='128 outputs and uniform evaluation complete',
        limitations='fine-tuned generic composition; background/segmentation proxies can be confounded')]
    csv_write(root/'external_model_registry.csv',external)


def old_evidence(root):
    evidence=root/'historical_evidence';evidence.mkdir(exist_ok=True)
    files=[CONF/'statistics/decision.json',CONF/'statistics/model_summary.json',
           CONF/'exploratory_extension_20261027/ablation_closeout/statistics/results.json',
           CONF/'exploratory_extension_20261027/ablation_closeout/statistics/contrasts.csv',
           CONF/'exploratory_extension_20261027/ablation_closeout/statistics/structure_descriptive.csv']
    manifest=[]
    for index,path in enumerate(files):
        dest=evidence/(f'{index}_'+path.name);shutil.copy2(path,dest)
        manifest.append(dict(source=str(path),source_sha256=sha(path),copy=str(dest),copy_sha256=sha(dest)))
    csv_write(evidence/'provenance.csv',manifest)
    assert read(CONF/'statistics/decision.json')['continue_new_training'] is False
    results=read(files[2]);pairs=[r for r in results['contrasts'] if r['b'].startswith('T3_no_')]
    csv_write(root/'tables/ablation_original_contrasts.csv',pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(8,3.6))
    for index,row in enumerate(pairs):
        low,high=row['ci95'];center=row['mean']
        axis.errorbar(center,index,xerr=[[center-low],[high-center]],fmt='o',capsize=4,color='#167278')
    axis.axvline(0,ls='--',color='gray');axis.set_yticks(range(len(pairs)),['Full minus no-body','Full minus no-rest'])
    axis.set_xlabel('DISTS Margin difference; identity95% CI')
    axis.set_title('Exploratory ablations | seed42 |107/128 valid\nOriginal planned-five Holm correction preserved')
    fig.tight_layout();fig.savefig(root/'figures/ablation_effects.png',dpi=180);plt.close(fig)
    # 原固定消融面板只读复制；保留其原始输入与图像来源。
    source=CONF/'exploratory_extension_20261027/ablation_closeout/figures'
    for path in source.rglob('*'):
        if path.is_file() and path.suffix.lower() in ['.png','.jpg','.json']:
            dest=root/'figures/ablation_reused'/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)


def plots(root,response,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in response if r['metric']=='dists_margin']
    fig,axis=plt.subplots(figsize=(9,4.8))
    for index,row in enumerate(rows):
        mean=row['mean'];axis.errorbar(mean,index,xerr=[[mean-row['ci_low']],[row['ci_high']-mean]],fmt='o',capsize=3)
    axis.axvline(0,color='gray',ls='--');axis.set_yticks(range(len(rows)),[r['model'] for r in rows])
    axis.set_xlabel('DISTS Margin on new-reference output (higher = stronger response proxy)')
    axis.set_title('P2b32 development reference replacement\nShared masks and inputs; native architecture/sampler differences disclosed')
    fig.tight_layout();fig.savefig(root/'figures/reference_response_models.png',dpi=180);plt.close(fig)


def panels(root,data):
    manifest=read(root/'protocol/input_manifest.json')
    originals={r['id']:r for r in manifest if r['condition']=='original'}
    swaps={r['id']:r for r in manifest if r['condition']=='swapped'}
    regular=sorted(originals,key=lambda identity:hashlib.sha256(('T3PP_GRID/'+identity).encode()).hexdigest())[:6]
    chosen=[(i,'fixed input hash') for i in regular]
    cn={r['id']:r for r in data['ControlNet+IP-Adapter'] if r['condition']=='original'}
    tail=sorted(cn,key=lambda i:cn[i].get('leak',-1),reverse=True)[:3]
    chosen.extend((i,'CN highest legacy background proxy; outcome selected') for i in tail)
    index=[]
    def render(identity,entries,folder,reason):
        cell=(160,214);canvas=Image.new('RGB',(len(entries)*160,248),'white');draw=ImageDraw.Draw(canvas)
        for column,(label,path) in enumerate(entries):
            draw.text((column*160+4,4),identity+' '+label,fill='black')
            im=ImageOps.contain(Image.open(path).convert('RGB'),(156,214));canvas.paste(im,(column*160+(160-im.width)//2,30+(214-im.height)//2))
        target=root/folder/(identity+'.png');target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target)
        index.append(dict(id=identity,selection=reason,panel=str(target),panel_sha256=sha(target),
                          source_sha256=json.dumps({label:sha(path) for label,path in entries},sort_keys=True)))
    for identity,reason in chosen:
        row=originals[identity];base=[('Sketch',Path(row['sketch'])),('Ref',Path(row['reference']))]
        render(identity,base+[(a,root/'outputs'/a/'original'/(identity+'.png')) if a in ['E0','E1'] else
            (a,SUP/'evaluation/P2b'/a/'images'/(identity+'.png')) for a in ['E0','E1','B','T3']],
            'figures/stage_comparison',reason)
        render(identity,base+[('T3',SUP/'evaluation/P2b/T3/images'/(identity+'.png'))]+
            [(a,root/'outputs'/a/'original'/(identity+'.png')) for a in ['BF-Fashion','ControlNet+IP-Adapter']],
            'external_failure_cases' if identity in tail else 'external_qualitative_grid',reason)
    t3={r['id']:r for r in data['T3'] if r['condition']=='swapped'}
    bf={r['id']:r for r in data['BF-Fashion'] if r['condition']=='swapped'}
    valid=[i for i in swaps if t3[i].get('dists_margin') is not None and bf[i].get('dists_margin') is not None]
    order=sorted(valid,key=lambda i:t3[i]['dists_margin']-bf[i]['dists_margin'])
    selected=sorted(swaps,key=lambda i:hashlib.sha256(('T3PP_SWAP/'+i).encode()).hexdigest())[:4]+order[:2]+order[-2:]
    for identity in dict.fromkeys(selected):
        old,new=originals[identity],swaps[identity]
        entries=[('Sketch',Path(old['sketch'])),('Old ref',Path(old['reference'])),('New ref',Path(new['reference']))]
        for arm in ['T3','BF-Fashion','ControlNet+IP-Adapter']:
            for condition in ['original','swapped']:
                path=(SUP/'evaluation'/('P2b' if condition=='original' else 'reference_swap')/arm/'images'/(identity+'.png')) if arm=='T3' else root/'outputs'/arm/condition/(identity+'.png')
                entries.append((arm+' '+condition,path))
        render(identity,entries,'figures/reference_swap','fixed input hash or labelled post-result T3-BF tail')
    csv_write(root/'provenance/final_panel_manifest.csv',index)


def ledger(root):
    paths=list((root/'logs').glob('*.log'))+list((PROJECT.parent/'LOTS/logs').glob('*.log'))
    ids=sorted({match.group(1) for path in paths if (match:=re.search(r'_(\d+)\.log$',path.name))})
    # Python3.8兼容；sacct原始字段同时保存，GPUh按实际已分配GPU数与ElapsedRaw计算。
    text=subprocess.check_output(['sacct','-X','-j',','.join(ids),'--parsable2','--noheader',
        '--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,ExitCode,Start,End,NodeList'],text=True)
    (root/'audit/sacct_raw.txt').write_text(text,encoding='utf-8')
    rows=[]
    for line in text.splitlines():
        values=line.split('|');job,name,state,seconds,tres,exitcode,start,end,node=values[:9]
        match=re.search(r'gres/gpu=(\d+)',tres);gpus=int(match.group(1)) if match else 0
        rows.append(dict(job_id=job,name=name,state=state,elapsed_seconds=int(seconds),allocated_gpus=gpus,
            gpu_hours=int(seconds)*gpus/3600,allocation=tres,exit_code=exitcode,start=start,end=end,node=node))
    csv_write(root/'resource_ledger.csv',rows)


def package():
    for path in list((SOURCE/'tools').glob('t3_p0p1*.py'))+[SOURCE/'scripts/t3_p0p1.sbatch']:
        dest=ROOT/'code'/path.relative_to(SOURCE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(ROOT/'code/source_version.json',dict(commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        repository='https://github.com/Tanyifeng11/Mymodel',branch='codex/t3-p0-p1-20261028',
        LOTS_sibling='/share/home/u2515283058/LOTS; branch codex/lots-repro-20261028'))
    paths=[]
    for path in ROOT.rglob('*'):
        relative=path.relative_to(ROOT)
        if not path.is_file() or relative.parts[0] in ['source','lots_env','runtime']:
            continue
        if path.suffix in ['.gz','.tgz'] or path.name in ['bundle_manifest.json','bundle_info.json']:
            continue
        if 'assets' in relative.parts or '.venv' in relative.parts or 'cache' in relative.parts:
            continue
        paths.append(path)
    write(ROOT/'bundle_manifest.json',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in paths])
    target=ROOT/'review_results.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for path in paths+[ROOT/'bundle_manifest.json']:archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    write(ROOT/'bundle_info.json',dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,members=len(paths)+1,
        no_large_model_weights=True,human_response_rows=0))
    print('BUNDLE_COMPLETE',read(ROOT/'bundle_info.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['aggregate','package']);args=parser.parse_args()
    globals()[args.action]()
