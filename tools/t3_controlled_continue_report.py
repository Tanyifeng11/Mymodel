"""在本地按训练前固定的规则判断是否追加训练，不启动任何作业。"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from tools.t3_training_diagnosis import read, write, csv_read, csv_write, sha, REST_WEIGHT
from tools.t3_training_diagnosis_report import paired

STAGES=['T3_final','C0250','C0500','C1000']
KEYS=['dists_rgb','dists_gray','color_js','dists_rgb_margin','dists_gray_margin','color_margin','iou','edge_f1','leak']


def run(root):
    server=root/'server_results'
    protocol=read(server/'protocol/controlled_protocol.json')
    assert protocol['lr']==5e-6 and protocol['maximum_successful_updates']==1000
    metrics=[]; losses=[]; fingerprints={}
    for stage in STAGES:
        group=[read(p) for p in sorted((server/'metrics'/stage).glob('*/*.json'))]
        assert len(group)==128 and len({(r['id'],r['condition']) for r in group})==128
        assert all(r['stage']==stage for r in group)
        for r in group:
            p=server/'outputs'/stage/r['condition']/(r['id']+'.png')
            assert sha(p)==r['output_sha256']
        metrics.extend(group)
        parts=[read(p) for p in sorted((server/'loss_probe'/stage).glob('*.json'))]
        assert len(parts)==64
        for part in parts:
            assert len(part)==4 and sorted(r['timestep'] for r in part)==[64,256,512,768]
            for row in part:
                k=(row['split'],row['id'],row['timestep'])
                v=tuple(row[n] for n in ['noise_sha256','latent_sha256','reference_latent_sha256'])
                if k in fingerprints: assert fingerprints[k]==v
                fingerprints[k]=v
            losses.append(dict(stage=stage,id=part[0]['id'],split=part[0]['split'],
                **{k:float(np.mean([r[k] for r in part])) for k in ['base','body_rgb','rest','total']}))
    assert len(fingerprints)==256
    manifest=read(server/'protocol/input_manifest.json')
    expected={(r['id'],r['condition']) for r in manifest}
    for stage in STAGES:
        assert {(r['id'],r['condition']) for r in metrics if r['stage']==stage}==expected
    for row in manifest:
        for stage in STAGES:
            m=read(server/'outputs'/stage/row['condition']/(row['id']+'.json'))
            assert m['initial_latent_sha256']==row['historical_initial_latent_sha256']
    summaries=[]; contrasts=[]; lsummaries=[]; lcontrasts=[]
    for stage in STAGES:
        for condition in ['original','swapped']:
            group=[r for r in metrics if r['stage']==stage and r['condition']==condition]
            for key in KEYS:
                v=[r[key] for r in group if r[key] is not None]
                summaries.append(dict(stage=stage,condition=condition,metric=key,n=len(v),mean=float(np.mean(v)) if v else None))
            if stage!='T3_final':
                a={r['id']:r for r in group}
                b={r['id']:r for r in metrics if r['stage']=='T3_final' and r['condition']==condition}
                for key in KEYS: contrasts.append(dict(stage=stage,condition=condition,metric=key,**paired(a,b,key)))
        for split in ['train_probe32','dev_probe32']:
            group=[r for r in losses if r['stage']==stage and r['split']==split]
            assert len(group)==32 and len({r['id'] for r in group})==32
            for key in ['base','body_rgb','rest','total']:
                lsummaries.append(dict(stage=stage,split=split,metric=key,n=32,mean=float(np.mean([r[key] for r in group]))))
            if stage!='T3_final':
                a={r['id']:r for r in group}
                b={r['id']:r for r in losses if r['stage']=='T3_final' and r['split']==split}
                for key in ['base','body_rgb','rest','total']:
                    lcontrasts.append(dict(stage=stage,split=split,metric=key,**paired(a,b,key)))
    csv_write(root/'metrics/generation_summary.csv',summaries)
    csv_write(root/'metrics/fixed_loss_summary.csv',lsummaries)
    csv_write(root/'metrics/fixed_loss_per_identity.csv',losses)
    csv_write(root/'statistics/generation_paired_effects.csv',contrasts)
    csv_write(root/'statistics/loss_paired_effects.csv',lcontrasts)
    def lc(stage):return next(r for r in lcontrasts if r['stage']==stage and r['split']=='dev_probe32' and r['metric']=='total')
    def gc(stage,condition,key):return next(r for r in contrasts if r['stage']==stage and r['condition']==condition and r['metric']==key)
    baseline=next(r['mean'] for r in lsummaries if r['stage']=='T3_final' and r['split']=='dev_probe32' and r['metric']=='total')
    final=lc('C1000'); middle=lc('C0500')
    reduction=-final['delta']/baseline
    rule=protocol['decision']
    loss_gate=middle['delta']<rule['dev_total_delta_at500_below'] and reduction>=rule['dev_total_reduction_at1000_at_least'] and final['ci_high']<rule['dev_total_CI_high_below']
    protections=[]
    for condition in rule['guardrail_conditions']:
        for key in rule['guardrail_metrics']:
            effect=gc('C1000',condition,key)
            degradation=effect['delta']*(1 if key=='leak' else -1)
            protections.append(dict(effect,degradation=degradation,
                passed=degradation<=rule['guardrail_mean_absolute_degradation_max']))
    guardrail_gate=all(r['passed'] for r in protections)
    texture_gate=gc('C0500','swapped','dists_gray')['delta']<0 and gc('C1000','swapped','dists_gray')['ci_high']<0
    decision=dict(loss_gate=loss_gate,guardrail_gate=guardrail_gate,texture_gate=texture_gate,
        dev_total_relative_reduction=float(reduction),dev_total_delta500=middle,dev_total_delta1000=final,
        guardrails=protections,recommend_more_for_objective=loss_gate and guardrail_gate,
        recommend_more_for_texture=loss_gate and guardrail_gate and texture_gate,
        uncertainty='one seed, fresh optimizer, one LR, fixed viewed development data; local texture metrics do not measure full motif fidelity')
    training=server/'training'
    first,second=csv_read(training/'training_log.csv'),csv_read(training/'auxiliary_log.csv')
    assert [int(r['update']) for r in first]==list(range(1,1001))
    assert [int(r['update']) for r in second]==list(range(1,1001))
    aligned=[]
    for a,b in zip(first,second):
        assert abs(float(a['lr'])-5e-6)<1e-15
        base=float(a['denoise'])+float(a['gram'])+.05*float(a['edge'])+.1*float(a['rgb_global'])
        total=base+.1*float(b['body_rgb'])+REST_WEIGHT*float(b['rest'])
        assert abs(total-float(a['total']))<1e-6
        aligned.append(dict(update=int(a['update']),base=base,body_rgb=float(b['body_rgb']),rest=float(b['rest']),total=float(a['total']),lr=float(a['lr'])))
    csv_write(root/'training/aligned_losses.csv',aligned)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False})
    (root/'figures').mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,key in zip(axes,['base','body_rgb','rest','total']):
        for split,label in [('train_probe32','训练32'),('dev_probe32','开发32')]:
            values=[next(r['mean'] for r in lsummaries if r['stage']==s and r['split']==split and r['metric']==key) for s in STAGES]
            ax.plot([0,250,500,1000],values,'o-',label=label)
        ax.set_title(key);ax.set_xlabel('新增成功更新');ax.legend(fontsize=8)
    fig.suptitle('同一固定协议；基线为原T3最终权重，续训采用新AdamW、LR=5e-6')
    fig.tight_layout();fig.savefig(root/'figures/fixed_loss_checkpoints.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for ax,key in zip(axes,['base','body_rgb','rest','total']):
        ax.plot([r['update'] for r in aligned],[r[key] for r in aligned],alpha=.2,lw=.6)
        bins=[aligned[i:i+50] for i in range(0,1000,50)]
        ax.plot([np.mean([r['update'] for r in g]) for g in bins],[np.mean([r[key] for r in g]) for g in bins])
        ax.set_ylabel(key)
        for step in [250,500,1000]: ax.axvline(step,color='gray',ls=':',lw=.6)
    axes[-1].set_xlabel('新增成功更新');fig.suptitle('续训随机训练日志：原始值与固定50步分箱；不与固定probe绝对值混比')
    fig.tight_layout();fig.savefig(root/'figures/training_losses.png',dpi=150);plt.close(fig)
    from PIL import Image,ImageDraw
    old={r['id']:r for r in manifest if r['condition']=='original'}
    new={r['id']:r for r in manifest if r['condition']=='swapped'}
    ids=sorted(new,key=lambda i:hashlib.sha256(('T3_DIAG_301042/figure/'+i).encode()).hexdigest())[:4]
    for condition in ['original','swapped']:
        page=Image.new('RGB',(7*160,4*240),'white');draw=ImageDraw.Draw(page)
        for j,identity in enumerate(ids):
            paths=[('Sketch',server/'inputs'/Path(old[identity]['sketch']).name),
                   ('Old ref',server/'inputs'/Path(old[identity]['reference']).name),
                   ('New ref',server/'inputs'/Path(new[identity]['reference']).name)]
            paths += [(s,server/'outputs'/s/condition/(identity+'.png')) for s in STAGES]
            for i,(label,path) in enumerate(paths):
                page.paste(Image.open(path).convert('RGB').resize((160,213)),(160*i,240*j+26))
                draw.text((160*i+3,240*j+2),identity+'\n'+label,fill='black')
        page.save(root/'figures'/('fixed_hash4_'+condition+'.png'))
    write(root/'decision.json',decision)
    write(root/'audit/analysis_checks.json',dict(generation_records=len(metrics),fixed_loss_raw_records=len(losses)*4,
        unique_fixed_noise_checks=len(fingerprints),all_input_latents_match=True,training_updates=1000,
        all_learning_rates_nonzero_and_constant=True,case_ids=ids,protocol_sha256=sha(server/'protocol/controlled_protocol.json')))
    print(decision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('F:/fuxian/Mymodel/output_eval/t3_controlled_continue_20261030'))
    run(p.parse_args().root)
