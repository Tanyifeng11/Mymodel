"""本地对下载包独立重算；不依赖训练/评测模块，不产生服务器实验结果。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def statistics(a):
    values=np.asarray(a,float)
    if not len(a):return dict(mean=None,n=0,ci95=None)
    samples=np.random.default_rng(32042).choice(values,(2000,len(a)),replace=True).mean(1)
    return dict(mean=float(values.mean()),n=len(a),ci95=np.percentile(samples,[2.5,97.5]).tolist())
def axial(a,b):return np.arccos(np.clip((a*b).sum(0),-1,1))*90/np.pi
def verify(root):
    checks={};manifest=read(root/'artifact_manifest.json')['files']
    for name,value in manifest.items():checks['sha/'+name]=digest(root/name)==value
    table=read(root/'result_table.json');stages={r['model']:r for r in table}
    for name,entry in stages.items():
        stage=root/name;rr=read(stage/'real/rows.json');real=entry['real'];checks[name+'/real256']=len(rr)==256
        checks[name+'/readable217']=sum(r['readable_cells']>0 for r in rr)==217
        for key,rowkey in [('near_advantage','near_advantage'),('random_advantage','random_advantage'),('rot90_success','rot90_response_success'),('r180_identity','r180_identity_success')]:
            checks[name+'/'+key]=real[key]==statistics([float(r[rowkey]) for r in rr if r[rowkey] is not None])
        for arm in ['matched','rot90','rot180','color_near','random','zero']:
            checks[name+'/'+arm]=real['errors'][arm]==statistics([float(r[arm+'_error']) for r in rr if r[arm+'_error'] is not None])
        cf=entry['controlled'];checks[name+'/controlled128']=cf['denominator']==128 and cf['strict']['denominator']==45
        rows={r['id']:r for r in rr}
        for path in (stage/'real/fields').glob('*.npz'):
            row=rows[path.stem]
            with np.load(path) as z:
                ori=z['orientation'];gt=z['gt'];mask=z['support'];weight=gt[3][mask]
                avg=lambda x:float(np.average(x[mask],weights=weight)) if mask.any() else None
                for i,arm in enumerate(['matched','rot90','rot180','color_near','random','zero']):
                    value=avg(axial(ori[i],gt[:2]));saved=row[arm+'_error']
                    checks[name+'/'+path.stem+'/'+arm]=value is saved if value is None else abs(value-saved)<1e-4
                for key,value in [('rot90_response_error',avg(axial(ori[1],-ori[0]))),('r180_response_error',avg(axial(ori[2],ori[0])))]:
                    checks[name+'/'+path.stem+'/'+key]=value is row[key] if value is None else abs(value-row[key])<1e-4
                checks[name+'/'+path.stem+'/finite']=all(np.isfinite(z[k]).all() for k in z.files)
                checks[name+'/'+path.stem+'/support']=int(mask.sum())==row['readable_cells']
        drift={r['id']:r for r in entry['drift']}
        for path in (stage/'feature_drift').glob('*.npz'):
            with np.load(path) as z:
                before,after,raw=z['before'],z['after'],z['raw_residual'];row=drift[path.stem]
                cosine=lambda a,b:(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8)
                values=dict(cosine_drift=float((1-cosine(before[0],after[0])).mean()),raw_residual_norm=float(np.linalg.norm(raw[0],axis=-1).mean()),
                    scaled_residual_norm=float(np.linalg.norm(.1*raw[0],axis=-1).mean()),rotation_relative_cosine_change=float(abs(cosine(after[0],after[1])-cosine(before[0],before[1])).mean()))
                checks[name+'/'+path.stem+'/drift']=all(abs(value-row[k])<1e-6 for k,value in values.items())
    final=read(root/'decision_summary.json');dual=[]
    for endpoint in final['endpoints']:
        cf,r=endpoint['controlled'],endpoint['real']
        ok=cf['clean_r90_success']['mean']>=.95 and cf['noisy_r90_both_success']['mean']>=.90 and cf['r180_identity_success']['mean']>=.95
        ok=ok and r['errors']['matched']['mean']<=7 and r['near_advantage']['mean']>=5 and r['near_advantage']['ci95'][0]>0 and r['rot90_success']['mean']>=.5
        ok=ok and cf['finite_prediction_rate']==r['finite_prediction_rate']==1
        checks['seed%d/dual_gate'%endpoint['seed']]=ok==endpoint['dual_gate_pass'];dual.append(ok)
    checks['two_of_three']=final['real_rotation_causality_pass']==(final['rf0_reproduction_pass'] and sum(dual)>=2)
    result=dict(checks=checks,**{'pass':all(checks.values())},check_count=len(checks),failed=[k for k,v in checks.items() if not v])
    (root/'independent_local_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}));assert result['pass']
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);verify(parser.parse_args().root)
