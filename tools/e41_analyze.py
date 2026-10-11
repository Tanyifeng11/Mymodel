"""E41 分组属性probe、七点轨迹及探索性瓶颈评分；不训练生成模型。"""
import csv
import subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tools.e41_protocol import *
from tools.e41_attributes import error,angular_error


def load(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}


def stats(values,groups):
    buckets=defaultdict(list)
    for v,g in zip(values,groups):
        if v is not None and np.isfinite(v):buckets[g].append(float(v))
    a=np.array([np.mean(v) for v in buckets.values()])
    if not len(a):return dict(mean=None,ci95=None,groups=0,n=0)
    sampled=a[np.random.default_rng(41043).integers(len(a),size=(2000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(sampled,[.025,.975]).tolist() if len(a)>1 else None,
        groups=len(a),n=sum(map(len,buckets.values())))


def target(value,attribute):
    if value is None:return None
    if attribute=='orientation':return np.array([np.cos(value*np.pi/90),np.sin(value*np.pi/90)])
    if attribute in ['period','scale']:return np.array([np.log(max(value,1e-8))])
    return np.array([value])


def recover(value,attribute):
    value=np.atleast_1d(value)
    if attribute=='orientation':return float(np.arctan2(value[1],value[0])*90/np.pi%180)
    if attribute in ['period','scale']:return float(np.exp(np.clip(value[0],-10,10)))
    return float(value[0])


class Transform:
    def __init__(self,x):
        self.scaler=StandardScaler().fit(x);a=self.scaler.transform(x)
        self.pca=PCA(n_components=min(32,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=42).fit(a)
    def __call__(self,x):return self.pca.transform(self.scaler.transform(x))


def probes(rows,x,full=False):
    summaries={};predictions={a:[] for a in ATTRS};models={}
    for fold in range(4):
        tr=np.array([r['fold']!=fold for r in rows]);te=~tr
        if not te.any() or tr.sum()<3:continue
        tx=Transform(x[tr]);ztr=tx(x[tr]);zte=tx(x[te]);trrows=[r for r,keep in zip(rows,tr) if keep]
        terows=[r for r,keep in zip(rows,te) if keep]
        for a in ATTRS:
            keep=np.array([r['params'].get(a) is not None for r in trrows]);valid=np.array([r['params'].get(a) is not None for r in terows])
            if keep.sum()<3 or not valid.any():continue
            y=np.stack([target(r['params'][a],a) for r,k in zip(trrows,keep) if k]);model=Ridge(alpha=1.).fit(ztr[keep],y)
            groups=sorted(set(r['family'] for r,k in zip(trrows,keep) if k));permuted=np.random.default_rng(41044+fold).permutation(groups)
            truth_by_group={g:np.mean([target(r['params'][a],a) for r,k in zip(trrows,keep) if k and r['family']==g],0) for g in groups}
            mapped=dict(zip(groups,permuted));shuffled=np.stack([truth_by_group[mapped[r['family']]] for r,k in zip(trrows,keep) if k])
            smodel=Ridge(alpha=1.).fit(ztr[keep],shuffled)
            yp=model.predict(zte[valid]);sp=smodel.predict(zte[valid]);null=recover(y.mean(0),a)
            for row,pred,shuf in zip([r for r,k in zip(terows,valid) if k],yp,sp):
                predicted=recover(pred,a);shuffled_value=recover(shuf,a);truth=row['params'][a]
                predictions[a].append(dict(id=row['id'],family=row['family'],fold=fold,truth=truth,prediction=predicted,
                    error=error(predicted,truth,a),null_error=error(null,truth,a),shuffled_error=error(shuffled_value,truth,a)))
    for a,records in predictions.items():
        groups=[r['family'] for r in records];mae=stats([r['error'] for r in records],groups)
        null=stats([r['null_error'] for r in records],groups);shuffled=stats([r['shuffled_error'] for r in records],groups)
        gain=stats([r['null_error']-r['error'] for r in records],groups)
        ratio=mae['mean']/null['mean'] if mae['mean'] is not None and null['mean'] and null['mean']>1e-8 else None
        summaries[a]=dict(mae=mae,null_mae=null,shuffled_mae=shuffled,gain=gain,
            normalized_mae=ratio,valid_readout=ratio is not None and ratio<.8 and gain['ci95'] is not None and gain['ci95'][0]>0,
            records=records)
    if full:
        tx=Transform(x);z=tx(x)
        for a in ATTRS:
            keep=np.array([r['params'].get(a) is not None for r in rows])
            if keep.sum()<3:continue
            y=np.stack([target(r['params'][a],a) for r,k in zip(rows,keep) if k]);models[a]=(tx,Ridge(alpha=1.).fit(z[keep],y))
    return summaries,models


def representation(refs):
    result={}
    for domain in ['controlled','real']:
        rows=[r for r in refs if r['domain']==domain];result[domain]={}
        features=[load(OUT/'features'/(r['id']+'.npz')) for r in rows]
        for layer in LAYERS:
            result[domain][layer]={}
            for mode in ['spatial','meanstd']:
                x=np.stack([f[layer+'_'+mode] for f in features])
                summary,_=probes(rows,x);result[domain][layer][mode]=summary
                print('E41 PROBE',domain,layer,mode,flush=True)
        del features
    comparison={}
    for a in ATTRS:
        encoder=result['controlled']['encoder_fused']['spatial'][a]
        token=result['controlled']['token_tcpm']['spatial'][a]
        e={r['id']:r for r in encoder['records']};t={r['id']:r for r in token['records']}
        shared=sorted(set(e)&set(t));gap=stats([t[k]['error']-e[k]['error'] for k in shared],[e[k]['family'] for k in shared])
        comparison[a]=dict(token_minus_encoder_error=gap,encoder_readout_valid=encoder['valid_readout'],
            token_readout_valid=token['valid_readout'],compression_bottleneck_supported=encoder['valid_readout'] and gap['ci95'] is not None and gap['ci95'][0]>0,
            limit='fixed linear probe and matched PCA capacity; comparative readability is not proof of information destruction')
    return result,comparison


def vae_analysis(refs):
    rows=[r for r in refs if r['domain']=='controlled'];features=[];roundtrip=[]
    for r in rows:
        path=OUT/'vae_standards'/(r['id']+'.npz');features.append(load(path)['readout']);roundtrip.append(read(path.with_suffix('.json')))
    summary,models=probes(rows,np.stack(features),full=True);reconstruction={}
    for a in ATTRS:
        vals=[error(r['roundtrip'].get(a),r['reference'].get(a),a) for r in roundtrip]
        reconstruction[a]=stats(vals,[r['family'] for r in rows])
    return dict(latent_probe=summary,roundtrip_error=reconstruction,
        valid_on_generated_domain=False,limit='independent full-field VAE calibration does not validate masked garment latent OOD transfer'),models


def generation_analysis(rows,models):
    records=[];latent_records=[]
    for r in rows:
        folder=OUT/'generated'/r['id'];data=read(folder/'attributes.json');latent=load(folder/'seven_latents.npz')
        for t in data['trajectory']:
            for a in ATTRS:
                reference=data['reference'].get(a);value=t['attributes'].get(a)
                records.append(dict(id=r['id'],carrier=r['carrier'],condition=r['condition'],attribute=a,t=t['t'],
                    reference=reference,generated=value,error=error(value,reference,a),mask_valid=data['mask_valid']))
                if a in models:
                    tx,model=models[a];pred=recover(model.predict(tx(latent['readout_t%d'%t['t']][None]))[0],a)
                    latent_records.append(dict(id=r['id'],carrier=r['carrier'],condition=r['condition'],attribute=a,t=t['t'],
                        prediction=pred,error_to_parameter=error(pred,r['params'][a],a),validity='OOD proxy, not calibrated generated semantic evidence'))
        for a in ATTRS:
            reference=data['reference'].get(a);value=data['final'].get(a)
            records.append(dict(id=r['id'],carrier=r['carrier'],condition=r['condition'],attribute=a,t='RGB',reference=reference,
                generated=value,error=error(value,reference,a),mask_valid=data['mask_valid']))
    summary={};change={}
    for a in ATTRS:
        summary[a]={str(t):stats([r['error'] for r in records if r['attribute']==a and r['t']==t],
            [r['carrier'] for r in records if r['attribute']==a and r['t']==t]) for t in TIMES+['RGB']}
        # 预先固定t801为早期、t1为后期；t981另列，不按最优时刻挑选。
        lookup={(r['id'],r['t']):r for r in records if r['attribute']==a}
        pairs=[];groups=[]
        for r in rows:
            early=lookup.get((r['id'],801));late=lookup.get((r['id'],1))
            if early and late and early['error'] is not None and late['error'] is not None:
                pairs.append(late['error']-early['error']);groups.append(r['carrier'])
        change[a]=stats(pairs,groups)
    return dict(records=records,summary=summary,latent_proxy_records=latent_records,late_minus_early_error=change,
        new_images=160,valid_masks=sum(read(OUT/'generated'/r['id']/'attributes.json')['mask_valid'] for r in rows),
        trajectory_note='guided predicted x0 decoded using original VAE; all RGB measurements mapped to common256 square coordinates; group bootstrap carriers',
        decoder_attribution='unverified: latent readout has unvalidated garment domain transfer')


def conditioning(rows):
    carriers=sorted(set(r['carrier'] for r in rows));records=[]
    for c in carriers:records.extend(read(OUT/'carriers'/c/'sensitivity.json'))
    result={}
    spans=dict(orientation=90.,period=np.log(2),scale=np.log(2),density=2/3)
    for a in ATTRS:
        selected=[r for r in records if r['attribute']==a]
        result[a]=dict(by_timestep={str(t):stats([r['epsilon']['relative'] for r in selected if r['t']==t],
            [r['carrier'] for r in selected if r['t']==t]) for t in TIMES},
            standardized_sensitivity=stats([r['sensitivity_per_parameter_unit']*spans[a] for r in selected],[r['carrier'] for r in selected]),
            sham=stats([r['sham']['relative'] for r in selected],[r['carrier'] for r in selected]))
    return dict(summary=result,records=records,meaning='epsilon response only; neither low nor high sensitivity independently establishes correct attribute utilization')


def reused():
    records=[];count=0
    for row in read(E39/'pairs.json'):
        folder=OUT/'real_reuse'/row['id'];audit=read(folder/'audit.json');assert audit['unchanged'];count+=len(audit['original_png_hashes'])
        for arm,h in audit['original_png_hashes'].items():assert sha(E39/'cases'/row['id']/(arm+'.png'))==h
        records.append(dict(id=row['id'],**read(folder/'attributes.json')))
    summary={}
    for a in ATTRS:
        summary[a]={}
        for t in TIMES+['RGB']:
            vals=[];groups=[]
            for r in records:
                m=r['final'] if t=='RGB' else next(v['attributes'] for v in r['trajectory'] if v['t']==t)
                vals.append(error(m.get(a),r['reference'].get(a),a));groups.append(r['id'])
            summary[a][str(t)]=stats(vals,groups)
    return dict(sources=32,original_images_hash_checked=count,new_generations=0,summary=summary,records=records,
        limit='BF real reference pseudo-attributes and original E39 captions, no independent physical pattern labels')


def bottlenecks(rep,compression,cond,gen,vae,cal):
    result={};maxcond=max(v['standardized_sensitivity']['mean'] or 0 for v in cond['summary'].values())
    for a in ATTRS:
        token=rep['controlled']['token_tcpm']['spatial'][a];null=token['null_mae']['mean']
        er=token['normalized_mae'];er=float(np.clip(er,0,1)) if er is not None else None
        sens=cond['summary'][a]['standardized_sensitivity']['mean']
        ec=1-sens/maxcond if sens is not None and maxcond>1e-12 else None
        delta=gen['late_minus_early_error'][a]['mean'];rgb=gen['summary'][a]['RGB']['mean']
        ed=float(np.clip(delta/null,0,1)) if delta is not None and null else None
        eg=float(np.clip(rgb/null,0,1)) if rgb is not None and null else None
        parts=[er,ec,ed,eg];score=float(np.mean(parts)) if all(v is not None for v in parts) and cal['summary'][a]['pass_calibration'] else None
        result[a]=dict(E_rep=er,E_cond=ec,E_denoise=ed,E_rgb=eg,B_equal_weight=score,
            parts_order=['representation','conditioning_proxy','denoising_proxy','RGB_proxy'],
            calibrated_input=cal['summary'][a]['pass_calibration'],semantic_composite_validated=False,
            compression_evidence=compression[a]['compression_bottleneck_supported'])
    ranked=sorted((a for a in ATTRS if result[a]['B_equal_weight'] is not None),key=lambda a:result[a]['B_equal_weight'],reverse=True)
    weights=[np.ones(4)/4]+[np.array([.4 if i==j else .2 for i in range(4)]) for j in range(4)]
    scenarios=[]
    for w in weights:
        scores={a:float(np.dot(w,[result[a][k] for k in ['E_rep','E_cond','E_denoise','E_rgb']])) for a in ranked}
        scenarios.append(dict(weights=w.tolist(),scores=scores,ranking=sorted(scores,key=scores.get,reverse=True)))
    # 表示以参数族、其余阶段以载体分组；同一次重采样在四属性间配对。
    families=sorted({r['family'] for r in rep['controlled']['token_tcpm']['spatial'][ATTRS[0]]['records']})
    carriers=sorted({r['carrier'] for r in cond['records']});stage_values={}
    for a in ATTRS:
        rr=rep['controlled']['token_tcpm']['spatial'][a]['records']
        er=np.array([np.mean([r['error'] for r in rr if r['family']==g]) for g in families])
        cr=[r for r in cond['records'] if r['attribute']==a]
        spans=dict(orientation=90.,period=np.log(2),scale=np.log(2),density=2/3)
        ec=np.array([np.mean([r['sensitivity_per_parameter_unit']*spans[a] for r in cr if r['carrier']==g]) for g in carriers])
        gr=[r for r in gen['records'] if r['attribute']==a];lookup={(r['id'],r['t']):r for r in gr}
        eg=[];ed=[]
        for g in carriers:
            rgb=[r['error'] for r in gr if r['carrier']==g and r['t']=='RGB' and r['error'] is not None]
            differences=[]
            for r in gr:
                early=lookup.get((r['id'],801))
                if r['carrier']==g and r['t']==1 and early and r['error'] is not None and early['error'] is not None:
                    differences.append(r['error']-early['error'])
            eg.append(np.mean(rgb) if rgb else np.nan);ed.append(np.mean(differences) if differences else np.nan)
        stage_values[a]=(er,ec,np.array(ed),np.array(eg))
    rng=np.random.default_rng(41046);draws={a:[] for a in ranked}
    for _ in range(1000):
        fi=rng.integers(len(families),size=len(families));ci=rng.integers(len(carriers),size=len(carriers))
        sensitivity={a:float(np.mean(stage_values[a][1][ci])) for a in ranked};ceiling=max(sensitivity.values(),default=0)
        for a in ranked:
            er,ec,ed,eg=stage_values[a];null=rep['controlled']['token_tcpm']['spatial'][a]['null_mae']['mean']
            if not null or not ceiling:continue
            den=ed[ci];rgb=eg[ci]
            if not np.isfinite(den).any() or not np.isfinite(rgb).any():continue
            parts=[np.mean(er[fi])/null,1-sensitivity[a]/ceiling,np.nanmean(den)/null,np.nanmean(rgb)/null]
            draws[a].append(float(np.mean(np.clip(parts,0,1))))
    for a in ranked:result[a]['B_ci95']=np.quantile(draws[a],[.025,.975]).tolist() if len(draws[a])>=900 else None
    pairwise={}
    if ranked:
        top=ranked[0]
        for a in ranked[1:]:
            if len(draws[top])==len(draws[a]) and len(draws[top])>=900:
                pairwise[top+'_minus_'+a]=np.quantile(np.asarray(draws[top])-draws[a],[.025,.975]).tolist()
    return dict(attributes=result,exploratory_ranking=ranked,weight_sensitivity=scenarios,
        exploratory_top_difference_ci95=pairwise,
        dominant_attribute=None,dominant_stage=None,status='inconclusive_causal_bottleneck',
        reason='Input calibration and representation probes can be evaluated. E_cond is a normalized response proxy; generated RGB/latent semantic validity and physical scale are not independently established. Composite rank is exploratory, not a causal bottleneck conclusion.',
        stop_condition_test='cannot assert all attributes preserved without calibrated generated-image labels')


def panels(rows):
    folder=OUT/'review';folder.mkdir(exist_ok=True)
    for carrier in sorted(set(r['carrier'] for r in rows)):
        selected=[r for r in rows if r['carrier']==carrier]
        canvas=Image.new('RGB',(10*154,2*236),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(selected):
            canvas.paste(Image.open(r['reference']).resize((150,150)),(i*154,24))
            canvas.paste(Image.open(OUT/'generated'/r['id']/'Rplus.png').resize((150,200)),(i*154,262))
            draw.text((i*154+2,3),r['condition'][:22],fill='black')
        canvas.save(folder/('generated_'+carrier+'.jpg'))


def analyze():
    assert sha(E5)==EXPECTED_E5
    assert all(sha(path)==digest for path,digest in read(OUT/'input_hashes.json').items())
    refs=read(OUT/'references.json');rows=read(OUT/'generation.json')
    assert all(sha(r['path'])==r['sha256'] for r in refs)
    for i in [0,1]:assert read(OUT/'audit'/('frozen_shard%d.json'%i))['unchanged']
    assert len(rows)==160 and all((OUT/'generated'/r['id']/'Rplus.png').exists() for r in rows)
    rep,compression=representation(refs);vae,models=vae_analysis(refs)
    gen=generation_analysis(rows,models);cond=conditioning(rows);real=reused();cal=read(OUT/'calibration.json')
    result=dict(protocol=read(OUT/'protocol.json'),calibration=cal['summary'],representation=rep,
        compression=compression,conditioning=cond,generation=gen,vae_controls=vae,real_reuse=real,
        bottleneck=bottlenecks(rep,compression,cond,gen,vae,cal),
        audit=dict(input_files_unchanged=len(read(OUT/'input_hashes.json')),reference_views_unchanged=len(refs),
            checkpoint_unchanged=True,generation_updates=0,new_images=160,
            analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    write(OUT/'E41_attribute_trajectory.json',result);panels(rows)
    flat=[]
    for a,stages in gen['summary'].items():
        for t,s in stages.items():flat.append(dict(attribute=a,timestep=t,**s))
    write(OUT/'trajectory_summary.json',flat)
    print('E41 ANALYSIS COMPLETE',result['bottleneck']['exploratory_ranking'],flush=True)


if __name__=='__main__':analyze()
