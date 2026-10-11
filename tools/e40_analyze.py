"""E40 固定分析：来源分组 OOF，配对来源 bootstrap；不训练生成模型。"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tools.e40_protocol import OUT, OLD, CLASSES, VARIANTS, read, write, sha
from tools.e39_visual_review import LABELS

REPS = ['fft', 'clip', 'local', 'color']


def load(path):
    with np.load(path) as f:
        return {k: f[k].copy() for k in f.files}


def valid(f, rep):
    return (rep!='color' or bool(f['mask_valid'])) and bool(f.get(rep+'_valid', True)) and np.isfinite(f[rep]).all()


def distance(a, b, rep):
    if not valid(a, rep) or not valid(b, rep): return None
    x, y = a[rep].ravel(), b[rep].ravel()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-10 or ny < 1e-10: return None
    return float(1-np.clip(np.dot(x, y)/(nx*ny), -1, 1))


def lab_distance(a, b):
    if not bool(a['mask_valid']) or not bool(b['mask_valid']): return None
    return float(np.linalg.norm(a['lab_mean']-b['lab_mean']))


def stats(values, groups):
    buckets = defaultdict(list)
    for value, group in zip(values, groups):
        if value is not None and np.isfinite(value): buckets[group].append(value)
    x = np.array([np.mean(v) for v in buckets.values()])
    if not len(x): return dict(mean=None, ci95=None, source_groups=0, observations=0)
    rng = np.random.default_rng(40044)
    means = x[rng.integers(len(x), size=(2000, len(x)))].mean(1)
    return dict(mean=float(x.mean()), ci95=np.quantile(means, [.025,.975]).tolist() if len(x)>1 else None,
                source_groups=len(x), observations=sum(map(len, buckets.values())))


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    cm = confusion_matrix(y, p, labels=CLASSES)
    denom = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*cm.diagonal(), denom, out=np.zeros(4, float), where=denom>0)
    return dict(n=len(y), accuracy=float((y==p).mean()) if len(y) else None,
                macro_f1=float(f1.mean()) if len(y) else None, confusion_matrix=cm.tolist(),
                per_class_f1=dict(zip(CLASSES, f1.tolist())))


def patterned_scores(y, p):
    indices=[i for i,label in enumerate(y) if label!='solid']
    result=scores([y[i] for i in indices],[p[i] for i in indices])
    result['macro_f1']=float(np.mean([result['per_class_f1'][c] for c in CLASSES[:3]])) if indices else None
    result['averaged_classes']=CLASSES[:3]
    return result


def paired_f1_drop(rows, predictions, variant):
    selected = [r for r in rows if r['id'] in predictions['base'] and r['id'] in predictions[variant]]
    if not selected: return dict(drop=None, ci95=None, source_groups=0)
    buckets = defaultdict(list)
    for i, r in enumerate(selected): buckets[r['family']].append(i)
    by_class = [[v for k,v in buckets.items() if selected[v[0]]['pattern']==c] for c in CLASSES]
    y = np.array([r['pattern'] for r in selected]); ids = [r['id'] for r in selected]
    a = np.array([predictions['base'][i] for i in ids]); b = np.array([predictions[variant][i] for i in ids])
    rng = np.random.default_rng(40045); drops=[]
    for _ in range(1000):
        indices = np.concatenate([v[j] for v in by_class if v for j in rng.integers(len(v),size=len(v))])
        drops.append(scores(y[indices],a[indices])['macro_f1']-scores(y[indices],b[indices])['macro_f1'])
    return dict(drop=scores(y,a)['macro_f1']-scores(y,b)['macro_f1'],
                ci95=np.quantile(drops,[.025,.975]).tolist(),source_groups=len(buckets), n=len(selected))


class Probe:
    def __init__(self, x, y):
        self.scale=StandardScaler().fit(x); a=self.scale.transform(x)
        self.pca=None
        if np.max(np.abs(a))>1e-12:
            self.pca=PCA(n_components=min(32,x.shape[1],len(x)-1),svd_solver='randomized',random_state=42).fit(a)
            a=self.pca.transform(a)
        self.model=LogisticRegression(C=1.,class_weight='balanced',max_iter=4000,random_state=42).fit(a,y)

    def proba(self, x):
        a=self.scale.transform(x)
        if self.pca is not None: a=self.pca.transform(a)
        return self.model.predict_proba(a)

    def predict(self, x):
        return self.model.classes_[np.argmax(self.proba(x),axis=1)]


def classify(rows, features):
    results={}; models={}
    for domain in ['controlled','real']:
        base=[r for r in rows if r['domain']==domain and r['variant']=='base']
        results[domain]={};models[domain]={}
        for rep in REPS:
            predictions={v:{} for v in VARIANTS}; folds={}; fitted={}
            for fold in range(4):
                train=[r for r in base if r['fold']!=fold and valid(features[(r['id'],'base')],rep)]
                test=[r for r in base if r['fold']==fold]
                assert not {r['family'] for r in train}&{r['family'] for r in test}
                if {r['pattern'] for r in train}!=set(CLASSES):
                    folds[str(fold)]=dict(status='missing train class');continue
                model=Probe(np.stack([features[(r['id'],'base')][rep] for r in train]),[r['pattern'] for r in train])
                fitted[fold]=model;folds[str(fold)]=dict(train=len(train),test=len(test),C=1.)
                for variant in VARIANTS:
                    eligible=[r for r in test if valid(features[(r['id'],variant)],rep)]
                    if eligible:
                        pred=model.predict(np.stack([features[(r['id'],variant)][rep] for r in eligible]))
                        predictions[variant].update({r['id']:str(p) for r,p in zip(eligible,pred)})
            metrics={}
            for variant in VARIANTS:
                eligible=[r for r in base if r['id'] in predictions[variant]]
                metrics[variant]=scores([r['pattern'] for r in eligible],[predictions[variant][r['id']] for r in eligible])
                metrics[variant]['patterned_only']=patterned_scores([r['pattern'] for r in eligible],[predictions[variant][r['id']] for r in eligible])
            results[domain][rep]=dict(metrics=metrics, folds=folds, predictions=predictions,
                paired_f1_drop={v:paired_f1_drop(base,predictions,v) for v in VARIANTS if v!='base'})
            models[domain][rep]=fitted
    return results,models


def reference_tests(rows, features):
    result={}
    for domain in ['controlled','real']:
        base=[r for r in rows if r['variant']=='base' and r['domain']==domain]
        result[domain]={}
        for rep in REPS[:3]:
            stable={}
            for variant in VARIANTS[1:]:
                vals=[distance(features[(r['id'],'base')],features[(r['id'],variant)],rep) for r in base]
                stable[variant]=stats(vals,[r['family'] for r in base])
            # 受控 B 的三类有纹参考严格共享直方图；素色另有亮度/方差混杂，不纳入主 gap。
            if domain=='controlled':
                pairs=[(a,b) for a,b in combinations(base,2) if a['pattern']!=b['pattern']
                       and 'solid' not in [a['pattern'],b['pattern']] and a['id'].split('_',1)[1]==b['id'].split('_',1)[1]]
            else:
                index={r['id']:r for r in base}
                pairs=[(index[p['left']],index[p['right']]) for p in read(OUT/'real_color_pairs.json')]
            # 同一参考可能出现在多个配对端点；真实配对按连通分量 bootstrap。
            parents={r['id']:r['id'] for r in base}
            def root(key):
                while parents[key]!=key:key=parents[key]
                return key
            for a,b in pairs: parents[root(b['id'])]=root(a['id'])
            records=[]
            for a,b in pairs:
                dp=distance(features[(a['id'],'base')],features[(b['id'],'base')],rep)
                dc=distance(features[(a['id'],'color1')],features[(a['id'],'color2')],rep)
                group=a['family'].split('_',1)[1] if domain=='controlled' else root(a['id'])
                records.append(dict(left=a['id'],right=b['id'],group=group,D_pattern=dp,D_color=dc,
                                    gap=None if dp is None or dc is None else dp-dc))
            sensitivity={k:stats([r[k] for r in records],[r['group'] for r in records]) for k in ['D_pattern','D_color','gap']}
            # 检索仅验证已知来源的图像增强，不能充当真实跨布料或完整重复单元身份验证。
            patterned=[r for r in base if r['pattern']!='solid']; retrieval={}
            for variant in VARIANTS[1:]:
                values=[];groups=[];details=[]
                for r in patterned:
                    candidates=[(q,distance(features[(r['id'],variant)],features[(q['id'],'base')],rep)) for q in patterned]
                    candidates=[(q,d) for q,d in candidates if d is not None]
                    if not candidates: continue
                    best=min(candidates,key=lambda p:p[1])[0]; hit=float(best['family']==r['family'])
                    values.append(hit);groups.append(r['family']);details.append(dict(query=r['id'],nearest=best['id'],same_family=bool(hit)))
                retrieval[variant]=dict(**stats(values,groups),candidate_families=len({r['family'] for r in patterned}),details=details)
            result[domain][rep]=dict(stability=stable,sensitivity=sensitivity,pair_records=records,retrieval=retrieval)
    return result


def probe_record(f, ref, models, rep, domain, target, fold=None):
    if not valid(f,rep) or not valid(ref,rep):return None
    pool=models[domain][rep]
    selected=[pool[fold]] if fold in pool else list(pool.values())
    if not selected:return None
    p=np.mean([m.proba(f[rep][None])[0] for m in selected],0)
    q=np.mean([m.proba(ref[rep][None])[0] for m in selected],0)
    classes=selected[0].model.classes_; idx=list(classes).index(target)
    return dict(prediction=str(classes[p.argmax()]),reference_prediction=str(classes[q.argmax()]),
                target=target,target_probability=float(p[idx]),reference_target_probability=float(q[idx]),
                correct=bool(classes[p.argmax()]==target),reference_correct=bool(classes[q.argmax()]==target))


def generated_tests(models):
    rows=read(OUT/'generation.json'); records=[]; fs={}
    for r in rows:
        folder=OUT/'generated'/r['id'];e=read(folder/'evaluation.json')
        f=load(folder/'features.npz');ref=load(e['reference_features']);fs[r['id']]=f
        records.append(dict(**r,mask_valid=bool(f['mask_valid']),patch_count=int(f['patch_count']),
            color_distance=lab_distance(f,ref),distances={rep:distance(f,ref,rep) for rep in REPS[:3]},
            classifier={domain:{rep:probe_record(f,ref,models,rep,domain,r['pattern']) for rep in REPS}
                        for domain in ['controlled','real']}))
    summaries={}
    for domain in ['controlled','real']:
        summaries[domain]={}
        for rep in REPS:
            selected=[r for r in records if r['classifier'][domain][rep] is not None]
            summaries[domain][rep]={}
            for color in [0,1]:
                rs=[r for r in selected if r['color']==color]
                clf=[r['classifier'][domain][rep] for r in rs]
                calibrated=[r for r in rs if r['classifier'][domain][rep]['reference_correct']]
                summaries[domain][rep][str(color)]=dict(
                    metrics=scores([r['pattern'] for r in rs],[c['prediction'] for c in clf]),
                    correct=stats([float(c['correct']) for c in clf],[r['carrier'] for r in rs]),
                    reference_correct=stats([float(c['reference_correct']) for c in clf],[r['carrier'] for r in rs]),
                    reference_valid_subset=stats([float(r['classifier'][domain][rep]['correct']) for r in calibrated],[r['carrier'] for r in calibrated]),
                    probability_drop=stats([c['reference_target_probability']-c['target_probability'] for c in clf],[r['carrier'] for r in rs]))
    responses={}
    for intervention in ['color','pattern']:
        pairs=[]
        for a,b in combinations(rows,2):
            if a['carrier']!=b['carrier']:continue
            if intervention=='color' and not(a['pattern']==b['pattern'] and a['color']!=b['color']):continue
            if intervention=='pattern' and not(a['pattern']!=b['pattern'] and a['color']==b['color']):continue
            x,y=fs[a['id']],fs[b['id']]
            pairs.append(dict(left=a['id'],right=b['id'],carrier=a['carrier'],
                lab=lab_distance(x,y),**{rep:distance(x,y,rep) for rep in REPS[:3]}))
        responses[intervention]=dict(summary={k:stats([p[k] for p in pairs],[p['carrier'] for p in pairs]) for k in ['lab']+REPS[:3]},pairs=pairs)
        if intervention=='pattern':
            patterned=[p for p in pairs if '_solid_' not in p['left'] and '_solid_' not in p['right']]
            responses[intervention]['matched_histogram_three_pattern_subset']={k:stats([p[k] for p in patterned],[p['carrier'] for p in patterned]) for k in ['lab']+REPS[:3]}
    preservation=dict(color=stats([r['color_distance'] for r in records],[r['carrier'] for r in records]),
        **{rep:stats([r['distances'][rep] for r in records],[r['carrier'] for r in records]) for rep in REPS[:3]})
    for carrier in sorted({r['carrier'] for r in rows}):
        canvas=Image.new('RGB',(8*144,212),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate([r for r in rows if r['carrier']==carrier]):
            x=j*144
            canvas.paste(Image.open(r['reference']).resize((64,64)),(x,0))
            canvas.paste(Image.open(OUT/'generated'/r['id']/'image.png').resize((108,144)),(x,64))
            draw.text((x+65,0),r['pattern'],fill='black');draw.text((x+65,15),'c%d'%r['color'],fill='black')
        path=OUT/'review'/('generated_'+carrier+'.jpg');canvas.save(path)
    return dict(records=records,classifier=summaries,reference_preservation=preservation,
                intervention_responses=responses,images=len(records),valid_masks=sum(r['mask_valid'] for r in records),
                valid_patch_images=sum(r['patch_count']>0 for r in records))


def reused_tests(models, refs):
    records=[]; interventions=[]; ref_index={r['id']:r for r in refs if r['variant']=='base'}
    for row in read(OLD/'pairs.json'):
        folder=OUT/'e39_reuse'/row['id'];audit=read(folder/'audit.json');arms=read(OLD/'cases'/row['id']/'complete.json')['arms']
        features={arm:load(folder/(arm+'_generated.npz')) for arm in arms}
        for arm in arms:
            assert sha(OLD/'cases'/row['id']/(arm+'.png'))==audit['original_png_sha256'][arm]
            if arm=='Rzero':continue
            f=features[arm];ref=load(folder/(arm+'_reference.npz'));target=LABELS[row['id']] if arm in ['Rplus','R90'] else None
            inputrow=ref_index.get('real_'+row['id']);fold=inputrow['fold'] if inputrow else None
            classifier={rep:probe_record(f,ref,models,rep,'real',target,fold) for rep in REPS} if target in CLASSES else None
            records.append(dict(id=row['id'],arm=arm,target=target,mask_valid=audit['mask_valid'],
                color_distance=lab_distance(f,ref),distances={rep:distance(f,ref,rep) for rep in REPS[:3]},classifier=classifier))
        for arm in arms:
            if arm=='Rplus':continue
            interventions.append(dict(id=row['id'],arm=arm,lab=lab_distance(features['Rplus'],features[arm]),
                **{rep:distance(features['Rplus'],features[arm],rep) for rep in REPS[:3]}))
    return dict(original_png_hashes_unchanged=True,reused_images=sum(len(read(OLD/'cases'/r['id']/'complete.json')['arms']) for r in read(OLD/'pairs.json')),
        records=records,interventions=interventions,
        summary={arm:{k:stats([r[k] for r in interventions if r['arm']==arm],[r['id'] for r in interventions if r['arm']==arm])
                     for k in ['lab']+REPS[:3]} for arm in ['Rminus','Rzero','R90','Rcolor']},
        note='Original E39 captions retained; donor labels unknown and out-of-four classes remain NA; no generated-image ground truth.')


def input_drift(rows):
    base={r['id']:r for r in rows if r['variant']=='base'}; records=[]
    for r in rows:
        if r['variant'] not in ['color1','color2']:continue
        a=np.asarray(Image.open(base[r['id']]['path']).convert('RGB'));b=np.asarray(Image.open(r['path']).convert('RGB'))
        gray=float(np.mean(abs(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(float)-cv2.cvtColor(b,cv2.COLOR_RGB2GRAY))))
        la=cv2.cvtColor(a.astype(np.float32)/255,cv2.COLOR_RGB2LAB);lb=cv2.cvtColor(b.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        records.append(dict(id=r['id'],domain=r['domain'],variant=r['variant'],family=r['family'],gray_mae=gray,
                            lab_delta=float(np.linalg.norm(la.mean((0,1))-lb.mean((0,1))))))
    return dict(records=records,summary={d:{k:stats([r[k] for r in records if r['domain']==d],[r['family'] for r in records if r['domain']==d])
                                           for k in ['gray_mae','lab_delta']} for d in ['real','controlled']})


def run():
    for shard in [0,1]:
        assert read(OUT/('shard%d_complete.json'%shard))['complete']
        audit=read(OUT/('audit/frozen_%d.json'%shard));assert audit['before']==audit['after']
    inputs=read(OUT/'input_hashes.json');assert all(sha(p)==v for p,v in inputs.items())
    rows=read(OUT/'references.json');assert all(sha(r['path'])==r['sha256'] for r in rows)
    features={(r['id'],r['variant']):load(OUT/'features'/('ref_'+r['id']+'_'+r['variant']+'.npz')) for r in rows}
    classification,models=classify(rows,features)
    reference=reference_tests(rows,features);generated=generated_tests(models)
    assert generated['images']==128
    reused=reused_tests(models,rows);drift=input_drift(rows);protocol=read(OUT/'protocol.json')
    gates={}
    for domain in ['controlled','real']:
        gates[domain]={}
        for rep in REPS[:3]:
            c=classification[domain][rep];gap=reference[domain][rep]['sensitivity']['gap']['ci95']
            drops=[c['paired_f1_drop'][v]['drop'] for v in ['color1','color2']]
            gates[domain][rep]=dict(classification=c['metrics']['base']['macro_f1']>=.70,
                color_stability=all(d is not None and d<=.05 for d in drops),
                positive_pattern_color_gap=gap is not None and gap[0]>0)
    passed=[r for r in REPS[:3] if all(gates['controlled'][r].values()) and all(gates['real'][r].values())]
    decision=dict(operational_candidates=passed,full_go=False,H2='unverified',D_real_material='unverified',
        status='inconclusive_full_identity',
        reason='No independently verified real fabric identities or cross-material pairs; category probes and augmented-source retrieval do not establish complete identity. Generation classifiers have unvalidated reference-to-garment domain transfer.',
        gates=gates,thresholds=protocol['gates'])
    result=dict(protocol=protocol,analysis=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        estimator='Train-only StandardScaler and PCA<=32, fixed balanced logistic C=1; no hyperparameter selection',
        uncertainty='2000 source-group bootstrap mean CIs; 1000 stratified paired source-group F1-drop bootstraps; generated means cluster on 16 carriers',
        cosine='Undefined zero-norm or missing patch descriptors are NA; distances compared only within representation',
        audits=dict(input_files_unchanged=len(inputs),reference_views_unchanged=len(rows),E5_frozen=True)),
        classification=classification,reference_tests=reference,input_color_drift=drift,
        controlled_generation=generated,e39_reuse=reused,decision=decision)
    write(OUT/'E40_pattern_representation_comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,domain in zip(axs,['controlled','real']):
        x=np.arange(4)
        for i,variant in enumerate(['base','color1','color2']):
            ax.bar(x+(i-1)*.24,[classification[domain][r]['metrics'][variant]['macro_f1'] for r in REPS],.24,label=variant)
        ax.set_xticks(x,REPS);ax.set_ylim(0,1.05);ax.set_ylabel('Grouped OOF macro F1');ax.set_title(domain);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'classification.png',dpi=160);plt.close(fig)
    write(OUT/'audit/artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and not any(s in p.name for s in ['.tar.gz','job_','artifact_hashes'])})
    print('E40 ANALYSIS COMPLETE',generated['images'],reused['reused_images'],decision,flush=True)


if __name__=='__main__':run()
