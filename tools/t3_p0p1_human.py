"""由真实PNG制作匿名人评草案；绝不填充或模拟真人评分。"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from tools.t3_p0p1 import read,write,sha,csv_write
from tools.t3_p0p1_review import sources,input_image


def build(project):
    root=project/'output_eval/t3_p0_p1_20261028'
    sup,_,_=sources(project)
    swaps=read(sup/'splits/reference_swap_v2.json')
    metrics={a:{r['id']:r for r in read(sup/'evaluation/P2b'/a/'per_identity_metrics.json')} for a in ['B','T1','T2','T3','C']}
    # 两个所有方法均较难的结构案例 + 八个输入哈希案例，不按T3是否获胜选题。
    hard=sorted(swaps,key=lambda r:min(metrics[a][r['id']]['iou'] for a in metrics))[:2]
    ids={r['id'] for r in hard}
    ordinary=sorted([r for r in swaps if r['id'] not in ids],
        key=lambda r:hashlib.sha256(('T3PP_HUMAN/'+r['id']).encode()).hexdigest())[:8]
    selected=hard+ordinary
    public=root/'human_eval/blind_material'
    private=root/'human_eval/private'
    public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    images={};stimuli=[];trials=[];keys=[];randomization=[]
    def asset(path):
        digest=sha(path)
        dest=public/'images'/(digest+path.suffix.lower())
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(path,dest)
        assert sha(dest)==digest
        images[digest]=dict(source=str(path),sha256=digest,public=str(dest.relative_to(public)))
        return str(dest.relative_to(public)).replace('\\','/')
    for row in selected:
        identity=row['id'];base=sup/'evaluation/P2b';swap=sup/'evaluation/reference_swap'
        oldmeta=read(base/'T3/images'/(identity+'.json'))
        newmeta=read(swap/'T3/images'/(identity+'.json'))
        inputs=[asset(input_image(sup,oldmeta,'sketch')),asset(input_image(sup,oldmeta,'reference')),
                asset(input_image(sup,newmeta,'reference'))]
        for baseline in ['B','T1','T2','C']:
            trial_id=hashlib.sha256(('trial/'+identity+'/'+baseline).encode()).hexdigest()[:12]
            variants=[]
            for arm in ['T3',baseline]:
                pair=[]
                for stage,folder in [('original',base),('swapped',swap)]:
                    path=folder/arm/'images'/(identity+'.png')
                    meta=read(path.with_suffix('.json'))
                    assert sha(path)==meta['output_sha256']
                    assert meta['gt_used_in_inference'] is False
                    pair.append(asset(path))
                    stimuli.append(dict(trial_id=trial_id,id=identity,model=arm,condition=stage,
                        output_sha256=sha(path),seed=meta['seed'],latent_sha256=meta['initial_latent_sha256'],
                        output_source=str(path),selection='global low IoU hard case' if row in hard else 'fixed input hash'))
                variants.append(pair)
            trials.append(dict(trial_id=trial_id,inputs=inputs,variants=variants))
            keys.append(dict(trial_id=trial_id,id=identity,variant0='T3',variant1=baseline))
    orders={}
    for index in range(1,31):
        reviewer=f'E{index:03}'
        rng=random.Random(281042+index)
        order=list(range(len(trials)));rng.shuffle(order)
        sides=[0]*20+[1]*20;rng.shuffle(sides)
        orders[reviewer]=[dict(trial=i,flip=side) for i,side in zip(order,sides)]
        for position,(i,side) in enumerate(zip(order,sides),1):
            key=keys[i]
            randomization.append(dict(reviewer=reviewer,position=position,trial_id=key['trial_id'],id=key['id'],
                A=key['variant1'] if side else key['variant0'],B=key['variant0'] if side else key['variant1']))
    payload=json.dumps(dict(trials=trials,orders=orders),ensure_ascii=False).replace('</','<\\/')
    (public/'index.html').write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    csv_write(private/'stimuli_manifest.csv',stimuli)
    csv_write(private/'randomization_manifest.csv',randomization)
    csv_write(private/'unblinding_key.csv',keys)
    csv_write(root/'human_eval/questionnaire.csv',[
        dict(question='q1',role='primary',text='哪种方法的输出更合理地响应了新参考？',options='A更好|B更好|两者相当|无法判断'),
        dict(question='q2',role='secondary',text='哪种方法更好地保持草图轮廓与基本服装结构？',options='A更好|B更好|两者相当|无法判断'),
        dict(question='q3',role='secondary',text='哪种结果更适合作为服装设计候选方案？',options='A更好|B更好|两者相当|无法判断')])
    protocol=dict(status='PILOT_READY_FORMAL_DRAFT',human_reviewers=0,responses=0,
        pilot_reviewers='5-8 actual participants',pilot_trials=8,formal_draft_trials=40,formal_reviewer_planning='20-30, revise from real pilot variance and repeated measures',
        source_split='previously viewed development32, no untouched-test claim',selection_ids=[r['id'] for r in selected],
        selection='2 lowest across-method IoU hard cases plus8 input hashes; not an estimate of population preference prevalence',
        baseline_comparisons=['B','T1','T2','C'],primary_question='reference-response rationality',secondary_questions=['structure','design utility'],
        randomization_seed_base=281042,randomization='participant-specific balanced side randomization and independent trial order; three endpoint questions fixed',
        formal_freeze='must be finalized after real pilot; do not treat this draft as completed formal study',
        missing='cannot judge excluded for that endpoint; ties retained as zero signed preference; retain missing and all raw files',
        exclusions='no automatic speed-based exclusion; flag median dwell<3sec for declared review; duplicate reviewer/trial requires resolution before formal analysis',
        planned_analysis='crossed participant and case random-intercept model for signed preference (-1,0,1); sensitivity two-way participant/case cluster bootstrap; final protocol fixed after pilot',
        multiplicity='Holm3 for q1 T3 vs B/T1/T2; historical C separately secondary; q2/q3 secondary families disclosed',
        AI_policy='no AI ratings, no inferred human wins, no synthetic rows',
        private_key='private directory excluded from participant zip')
    write(root/'human_eval/human_eval_protocol.json',protocol)
    write(root/'human_eval/private/image_provenance.json',images)
    write(root/'human_eval/status.json',dict(status='AWAITING_REAL_PILOT',human_reviewers=0,actual_response_rows=0,formal_complete=False))
    for name,fields in [('responses_raw.csv',['reviewer','phase','trial_id','q1','q2','q3','seconds','background','exported_at']),
                        ('responses_clean.csv',['reviewer','id','baseline','question','score','status']),
                        ('human_eval_statistics.csv',['baseline','question','n_reviewers','n_cases','effect','ci_low','ci_high','p','p_holm'])]:
        path=root/'human_eval'/name
        if not path.exists():
            with path.open('w',encoding='utf-8-sig',newline='') as stream:
                csv.writer(stream).writerow(fields)
    archive=root/'human_eval/blind_material.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(public.rglob('*')):
            if path.is_file():zipped.write(path,str(path.relative_to(public)))
    write(root/'human_eval/material_verification.json',dict(trials=len(trials),orders=len(orders),
        all_orders_unique_trials=all(len({r['trial'] for r in order})==40 for order in orders.values()),
        balanced_side_positions=all(sum(r['flip'] for r in order)==20 for order in orders.values()),
        participant_zip_sha256=sha(archive),shared_files=len(images)+1,raw_response_rows=0,
        unblinding_not_in_zip=True,GT_not_in_stimuli=True))
    print('HUMAN_PILOT_MATERIAL_READY',len(trials),'trials, human responses=0')


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>参考响应匿名评价 · 试评草案</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f7;color:#24313b;margin:0}main{max-width:1200px;margin:auto;padding:24px}
section{background:white;border-radius:12px;padding:20px;margin-bottom:18px}h1{font-size:24px;margin:0 0 10px}h2{font-size:19px}
.inputs,.methods,.pair{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}.inputs img{height:150px;max-width:200px;object-fit:contain}
.pair img{width:min(20vw,210px);height:auto}figure{margin:0;text-align:center}figcaption{font-size:14px;color:#647583}
.methods>section{background:#f7f9fa;border:1px solid #dce4e8;flex:1;min-width:250px}.question{padding:12px 0;border-bottom:1px solid #e7ecef}
label{display:inline-block;margin:8px 20px 8px 0}button,select,input{font:inherit;padding:8px 14px;border:1px solid #b7c8ce;border-radius:6px}button{background:#126b72;color:white;cursor:pointer}
.bar{display:flex;justify-content:space-between;gap:12px;align-items:center}.muted{color:#6a7883;font-size:14px}#study{display:none}
</style><main><section><h1>参考面料变化与服装结构评价</h1><div class="muted">匿名试评材料 · 请独立判断 · 没有标准答案</div></section>
<section id="welcome"><p>每题展示同一草图、旧参考、新参考，以及两种匿名方法在旧/新参考下的结果。请比较参考变化是否被合理表现，同时考虑基本服装结构。</p>
<p>请使用研究者分配的匿名编号。不收集姓名或联系方式；答卷仅下载到你的设备，由你交给研究者。你可以退出、导出部分答卷或选择“无法判断”。</p>
<label>匿名编号 <select id="reviewer"></select></label><label>背景 <select id="background"><option>无服装相关经验</option><option>服装设计学习者</option><option>服装或视觉设计从业者</option><option>相关研究者</option></select></label>
<label>阶段 <select id="phase"><option value="pilot">试评：8组</option><option value="formal_draft">正式草案：40组（需研究者批准后使用）</option></select></label>
<p><button id="start">自愿参加并开始</button></p><p class="muted">预计试评约5–10分钟。研究者须先收集真实试评，再确定正式问卷与规模。</p></section>
<div id="study"><section><div class="bar"><b id="progress"></b><button id="export">导出当前答卷 CSV</button></div><div id="inputs" class="inputs"></div></section>
<div id="outputs" class="methods"></div><section id="questions"></section><section><div class="bar"><button id="prev">上一题</button><span class="muted">允许“相当”和“无法判断”，无需猜测。</span><button id="next">下一题</button></div></section></div></main>
<script>const data=__DATA__;let session,position=0,began=0,answers={};const $=id=>document.getElementById(id);
Object.keys(data.orders).forEach(id=>$('reviewer').add(new Option(id,id)));
const questions=['哪种方法的输出更合理地响应了新参考？','哪种方法更好地保持草图轮廓与基本服装结构？','哪种结果更适合作为服装设计候选方案？'];
const labels=[['A','A更好'],['B','B更好'],['tie','两者相当'],['cannot','无法判断']];
function key(){return 't3pp-human-'+session.reviewer+'-'+session.phase}
function save(){const item=session.order[position];if(!item)return;const trial=data.trials[item.trial];const old=answers[trial.trial_id]||{};
const row={...old,trial_id:trial.trial_id,seconds:(old.seconds||0)+Math.max(0,(Date.now()-began)/1000)};
for(let q=1;q<=3;q++){const node=document.querySelector('input[name=q'+q+']:checked');row['q'+q]=node?node.value:''}answers[trial.trial_id]=row;began=Date.now();localStorage.setItem(key(),JSON.stringify(answers))}
function show(){const item=session.order[position],trial=data.trials[item.trial],a=answers[trial.trial_id]||{};const order=item.flip?[1,0]:[0,1];
$('progress').textContent=(position+1)+' / '+session.order.length;
$('inputs').innerHTML=trial.inputs.map((src,i)=>'<figure><img src="'+src+'"><figcaption>'+['草图','旧参考','新参考'][i]+'</figcaption></figure>').join('');
$('outputs').innerHTML=order.map((v,i)=>'<section><h2>方法 '+['A','B'][i]+'</h2><div class="pair">'+trial.variants[v].map((src,j)=>'<figure><img src="'+src+'"><figcaption>'+['旧参考下输出','新参考下输出'][j]+'</figcaption></figure>').join('')+'</div></section>').join('');
$('questions').innerHTML=questions.map((q,i)=>'<div class="question"><b>'+(i+1)+'. '+q+'</b><div>'+labels.map(([v,l])=>'<label><input type="radio" name="q'+(i+1)+'" value="'+v+'" '+(a['q'+(i+1)]===v?'checked':'')+'>'+l+'</label>').join('')+'</div></div>').join('');
$('prev').disabled=position===0;$('next').textContent=position===session.order.length-1?'保存并导出':'下一题';began=Date.now();window.scrollTo(0,0)}
$('start').onclick=()=>{session={reviewer:$('reviewer').value,phase:$('phase').value,background:$('background').value};session.order=data.orders[session.reviewer].slice(0,session.phase==='pilot'?8:40);
answers=JSON.parse(localStorage.getItem(key())||'{}');$('welcome').style.display='none';$('study').style.display='block';show()};
$('prev').onclick=()=>{save();if(position>0){position--;show()}};$('next').onclick=()=>{save();if(position<session.order.length-1){position++;show()}else exportCSV()};
function exportCSV(){save();const fields=['reviewer','phase','trial_id','q1','q2','q3','seconds','background','exported_at'];const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const now=new Date().toISOString();
const rows=Object.values(answers).map(a=>({...a,reviewer:session.reviewer,phase:session.phase,background:session.background,exported_at:now}));
const csv='\\ufeff'+[fields.join(','),...rows.map(row=>fields.map(k=>quote(row[k])).join(','))].join('\\r\\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));link.download='anonymous_'+session.reviewer+'_'+session.phase+'.csv';link.click();URL.revokeObjectURL(link.href)}
$('export').onclick=exportCSV;
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();build(args.project)
