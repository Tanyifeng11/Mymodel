"""按身份汇总 A-G；只生成数据与图，不向服务器写最终 Markdown 报告。"""
import argparse
import collections
import numpy as np
from PIL import Image,ImageDraw
from tools.e39_protocol import *
from tools.e39_visual_review import LABELS, COLOR, INPUT_LIMITS


def stats(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(a):return dict(n=0,mean=None,median=None,ci95=None)
    means=np.random.default_rng(39042).choice(a,(10000,len(a)),replace=True).mean(1)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),ci95=np.percentile(means,[2.5,97.5]).tolist())


def readout(x,mode):
    x=x.astype(np.float32)
    if mode=='flat':return x.ravel()
    if x.ndim==4:x=x[0].reshape(x.shape[1],-1).T
    elif x.ndim==3:x=x[0]
    else:return x.ravel()
    return np.concatenate([x.mean(0),x.std(0)]).ravel()


def normalized(x):
    return x/max(float(np.linalg.norm(x)),1e-12)


def ridge(xtrain,ytrain,xtest):
    # 训练折中心化与归一化，双对偶线性岭回归；固定 alpha=1，不按结果选参。
    mean=xtrain.mean(0);xtrain=xtrain-mean;xtest=xtest-mean
    scale=np.sqrt(np.mean(xtrain*xtrain,axis=0));scale=np.maximum(scale,1e-4)
    xtrain=xtrain/scale;xtest=xtest/scale
    gram=xtrain@xtrain.T/xtrain.shape[1]
    return xtest@xtrain.T/xtrain.shape[1]@np.linalg.solve(gram+np.eye(len(xtrain)),ytrain)


def feature_analysis(rows):
    layers=['clip_patches','cnn3_native','encoder_fused','token_resampler','token_bf','token_tcpm']
    samples={};sensitivity={};probes={}
    for layer in layers:
        samples[layer]={mode:[] for mode in ['meanstd','flat']}
        effects={arm:[] for arm in ['Rminus','Rzero','R90','Rcolor']}
        for r in rows:
            folder=OUT/'cases'/r['id'];values={}
            for arm in read(folder/'complete.json')['arms']:
                with np.load(folder/(arm+'_features.npz')) as f:values[arm]=f[layer].astype(np.float32)
            baseline=values['Rplus'].ravel()
            for arm in values:
                if arm=='Rplus':continue
                other=values[arm].ravel()
                effects[arm].append(dict(id=r['id'],cosine_distance=float(1-normalized(baseline)@normalized(other)),
                    relative_l2=float(np.linalg.norm(baseline-other)/max(np.linalg.norm(baseline),1e-12))))
            for mode in samples[layer]:
                # 原生 CNN 的 flat 读出用保存的 8x8 网格，避免把空间平均误称原生 flatten。
                if mode=='flat' and layer=='cnn3_native':
                    vals={}
                    for arm in ['Rplus','R90']:
                        with np.load(folder/(arm+'_features.npz')) as f:vals[arm]=f['cnn3_pool8']
                else:vals=values
                samples[layer][mode].append({arm:readout(vals[arm],mode) for arm in ['Rplus','R90']})
        sensitivity[layer]={arm:dict(relative_l2=stats([r['relative_l2'] for r in v]),
            cosine_distance=stats([r['cosine_distance'] for r in v]),identities=v) for arm,v in effects.items()}
        probes[layer]={}
        for mode,items in samples[layer].items():
            x=np.stack([s[arm] for s in items for arm in ['Rplus','R90']]);groups=np.repeat(np.arange(len(rows)),2)
            labels=np.repeat([r['pattern'] for r in rows],2)
            counts=collections.Counter(r['pattern'] for r in rows)
            classes=sorted(k for k,n in counts.items() if n>=2 and k!='unspecified')
            keep=np.array([v in classes for v in labels]);pred=np.full(len(x),-1,int);folds={}
            rng=np.random.default_rng(39043)
            for label in classes:
                ids=rng.permutation([i for i,r in enumerate(rows) if r['pattern']==label])
                for j,i in enumerate(ids):folds[i]=j%3
            if len(classes)>=2:
                for fold in range(3):
                    test=keep&np.array([folds.get(g,-1)==fold for g in groups]);train=keep&~test
                    if not test.any() or not train.any():continue
                    y=np.array([[float(v==c) for c in classes] for v in labels[train]])
                    pred[test]=ridge(x[train],y,x[test]).argmax(1)
            valid=keep&(pred>=0);correct=np.array([classes[p]==v if p>=0 else False for p,v in zip(pred,labels)])
            balanced=[float(correct[valid&(labels==c)].mean()) for c in classes if (valid&(labels==c)).any()]
            original=np.stack([normalized(s['Rplus']) for s in items]);rotated=np.stack([normalized(s['R90']) for s in items])
            retrieval=(rotated@original.T).argmax(1)==np.arange(len(items))
            # FFT 伪标签方向回归：两视图属于同一身份，按身份分折，绝不跨折泄漏。
            direction=[];di=[]
            for i,r in enumerate(rows):
                ref=read(OUT/'cases'/r['id']/'semantics.json')['fft_reference']
                if all(ref[a]['confidence'] is not None and ref[a]['confidence']>=.15 for a in ['Rplus','R90']):
                    for arm in ['Rplus','R90']:
                        angle=ref[arm]['angle']*np.pi/90;direction.append([np.cos(angle),np.sin(angle)])
                        di.append(i*2+(arm=='R90'))
            errors=[];constant=[]
            if len(set(groups[di]))>=6:
                di=np.asarray(di,int);y=np.asarray(direction);fold_assignment=groups[di]%3
                for fold in range(3):
                    tr=fold_assignment!=fold;te=~tr
                    if not te.any():continue
                    out=ridge(x[di[tr]],y[tr],x[di[te]])
                    def err(pred,true):return np.abs((np.arctan2(pred[:,1],pred[:,0])-np.arctan2(true[:,1],true[:,0])+np.pi)%(2*np.pi)-np.pi)*90/np.pi
                    errors.extend(err(out,y[te]).tolist());constant.extend(err(np.tile(y[tr].mean(0),(te.sum(),1)),y[te]).tolist())
            probes[layer][mode]=dict(pattern_accuracy=float(correct[valid].mean()) if valid.any() else None,
                balanced_pattern_accuracy=float(np.mean(balanced)) if balanced else None,classes=classes,class_sources=dict(counts),
                pattern_views=int(valid.sum()),pattern_sources=int(len(set(groups[valid]))),majority_chance=max([counts[c] for c in classes],default=0)/max(sum(counts[c] for c in classes),1),
                grouped_identity_folds=True,single_ai_visual_labels=True,rotation_identity_linear_retrieval=stats(retrieval.astype(float)),
                retrieval_chance=1/len(items),direction_axial_mae=float(np.mean(errors)) if errors else None,
                direction_constant_mae=float(np.mean(constant)) if constant else None,direction_views=len(errors),
                direction_sources=len(set(groups[di])) if len(di) else 0,
                flat_note='CNN native uses saved 8x8 pool; others full saved flatten')
        del samples[layer]
    write(OUT/'representation_summary.json',dict(sensitivity=sensitivity,linear_probes=probes,
        limits=['single AI reference-surface labels, not human ground truth; low N and correlated original/rotation views',
                'identity retrieval uses original identity as class prototype; tests rotation robustness, not unseen-identity classification',
                'FFT pseudo labels test direction readout only; not motif ground truth',
                'representational distance scales cannot be compared directly across dimensions']))
    return sensitivity,probes


def analyze():
    rows=read(OUT/'pairs.json')
    assert all((OUT/'cases'/r['id']/'semantics.json').exists() for r in rows)
    assert set(LABELS)=={r['id'] for r in rows}
    write(OUT/'reference_visual_review.json',dict(assessor='single AI, not human or independent validation',
        label_lock='input contact sheets reviewed before aggregate representation/semantic statistics',
        rows=[dict(id=r['id'],reference_sha256=sha(r['reference']),surface_label=LABELS[r['id']],
            caption_pattern=r['pattern'],input_limit=INPUT_LIMITS.get(r['id']),color_review=COLOR.get(r['id'])) for r in rows]))
    rows=[dict(r,caption_pattern=r['pattern'],pattern=LABELS[r['id']]) for r in rows]
    sensitivity,probes=feature_analysis(rows)
    common=[read(OUT/'cases'/r['id']/'common_trajectory.json') for r in rows]
    trajectory={str(t):{branch:stats([v[i][branch]['relative'] for v in common]) for branch in ['conditional','guided']}
                for i,t in enumerate([r['t'] for r in common[0]])}
    exact={};final={};semantic={};attention={}
    for arm in ['Rminus','Rzero','R90','Rcolor','Rtoken0']:
        exact[arm]={str(t):stats([v['conditional']['relative'] for r in rows
            for v in read(OUT/'cases'/r['id']/'exact_probes.json') if v['t']==t and v['arm']==arm]) for t in TIMES}
        if arm!='Rtoken0':
            comparisons=[read(OUT/'cases'/r['id']/'free_trajectory_comparison.json').get(arm) for r in rows]
            final[arm]={key:stats([c[key]['relative'] for c in comparisons if c]) for key in ['final_latent','rgb_unclamped','rgb_clamped']}
        if arm in ['Rminus','R90','Rcolor']:
            effects=[read(OUT/'cases'/r['id']/'semantics.json')['reference_preference_effects'].get(arm,{}) for r in rows]
            semantic[arm]={key:{measure:stats([e[key][measure] for e in effects if e.get(key)])
                for measure in ['donor_gain','donor_vs_original_preference_shift']}
                for key in ['clip_texture','tpf_patch','lab_delta','fft_angular_l1','fft_radial_l1','fft_orientation_error']}
    for t in TIMES:
        attention[str(t)]={}
        for r in rows:
            vals=[v['residual_norm_fraction'] for v in read(OUT/'cases'/r['id']/'attention.json')
                  if v['t']==t and v['arm']=='Rplus' and v['branch']=='conditional' and v['kind']=='post_gate_residual']
            attention[str(t)][r['id']]=float(np.mean(vals)) if vals else None
    replay=[read(OUT/'cases'/r['id']/'Rplus_generation.json')['original_e5_png_equal'] for r in rows]
    verified_effects=[read(OUT/'cases'/r['id']/'semantics.json')['reference_preference_effects'].get('Rcolor',{})
                     for r in rows if COLOR.get(r['id'],{}).get('verified')]
    verified_color={key:{measure:stats([e[key][measure] for e in verified_effects if e.get(key)])
                    for measure in ['donor_gain','donor_vs_original_preference_shift']}
                    for key in ['clip_texture','tpf_patch','lab_delta','fft_angular_l1','fft_radial_l1','fft_orientation_error']}
    summary=dict(identities=len(rows),steps=50,training_updates=0,
        color_candidates=sum(r['color_valid'] for r in rows),mask_valid=sum(read(OUT/'cases'/r['id']/'semantics.json')['mask_valid'] for r in rows),
        color_single_ai_verified=sum(v['verified'] for v in COLOR.values()),verified_color_semantics=verified_color,
        original_e5_replay=dict(available=sum(v is not None for v in replay),exact=sum(v is True for v in replay)),
        trajectory_same_latent=trajectory,exact_t_same_latent=exact,latent_to_rgb=final,semantics=semantic,
        attention_residual_norm_fraction={t:stats(list(v.values())) for t,v in attention.items()})
    write(OUT/'summary.json',summary)
    frozen=read(OUT/'audit/input_hashes.json');assert all(sha(p)==h for p,h in frozen.items())
    write(OUT/'audit/inputs_unchanged.json',dict(unchanged=True,files=len(frozen)))
    panels(rows);plots(summary,sensitivity,probes)
    write(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name!='artifact_hashes.json' and not p.name.startswith('job_')
        and '.tar.gz' not in p.name})
    print('E39 ANALYSIS COMPLETE',len(rows),flush=True)


def panels(rows):
    folder=OUT/'panels';folder.mkdir(exist_ok=True)
    for i,r in enumerate(rows):
        case=OUT/'cases'/r['id'];arms=read(case/'complete.json')['arms']
        canvas=Image.new('RGB',(160*(2+len(arms)),248),'white');draw=ImageDraw.Draw(canvas)
        views=[('Sketch',r['sketch']),('Reference',r['reference'])]+[(a,case/(a+'.png')) for a in arms]
        for j,(label,path) in enumerate(views):
            canvas.paste(Image.open(path).convert('RGB').resize((160,216)),(160*j,32))
            draw.text((160*j+3,4),label,fill='black')
        draw.text((3,19),r['id'],fill='black');canvas.save(folder/('%02d_%s.png'%(i,r['id'])))


def plots(summary,sensitivity,probes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    t=sorted(map(int,summary['trajectory_same_latent']),reverse=True)
    for branch in ['conditional','guided']:
        vals=[summary['trajectory_same_latent'][str(k)][branch] for k in t]
        axes[0].plot(t,[v['mean'] for v in vals],label=branch)
        axes[0].fill_between(t,[v['ci95'][0] for v in vals],[v['ci95'][1] for v in vals],alpha=.15)
    axes[0].invert_xaxis();axes[0].set(title='Same-latent reference swap',xlabel='DDIM timestep',ylabel='Relative epsilon L2');axes[0].legend()
    layers=list(sensitivity)
    for arm in ['Rminus','R90']:
        axes[1].plot(range(len(layers)),[sensitivity[k][arm]['cosine_distance']['mean'] for k in layers],'o-',label=arm)
    axes[1].set_xticks(range(len(layers)));axes[1].set_xticklabels(layers,rotation=45,ha='right');axes[1].set(title='Representation sensitivity',ylabel='Cosine distance');axes[1].legend()
    keys=['clip_texture','tpf_patch','fft_angular_l1','fft_radial_l1'];values=[summary['semantics']['Rminus'][k]['donor_vs_original_preference_shift'] for k in keys]
    axes[2].bar(range(len(keys)),[v['mean'] if v['mean'] is not None else 0 for v in values])
    axes[2].axhline(0,color='black',linewidth=.7);axes[2].set_xticks(range(len(keys)));axes[2].set_xticklabels(keys,rotation=40,ha='right');axes[2].set(title='Final RGB preference shift (+ follows donor)')
    fig.tight_layout();fig.savefig(OUT/'trajectory_overview.png',dpi=150);plt.close(fig)


if __name__=='__main__':analyze()
