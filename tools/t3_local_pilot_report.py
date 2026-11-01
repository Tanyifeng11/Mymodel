"""独立感知评价、受控周期正控和保守停止规则；不把频谱等同纹样身份。"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tools.t3_local_pilot import ROOT, PROJECT, ARMS, read, write, sha, coordinates


def pattern_measure(pixels, row):
    from scipy.signal import find_peaks
    angles, periods = [], []
    for x,y in row['patches']:
        patch = pixels[y:y+128,x:x+128].astype(float).mean(-1)/255
        dy,dx = np.gradient(patch)
        angle = 0 if np.mean(dy**2) > np.mean(dx**2) else 90
        profile = patch.mean(1 if angle==0 else 0)
        profile -= profile.mean()
        ac = np.correlate(profile,profile,'full')[len(profile)-1:]
        ac /= np.arange(len(profile),0,-1)
        ac /= max(ac[0],1e-10)
        peaks,_ = find_peaks(ac[5:42],height=.25,prominence=.1)
        peaks += 5
        if len(peaks):
            best = ac[peaks].max()
            periods.append(int(min(p for p in peaks if ac[p]>=.85*best)))
        angles.append(angle)
    angle = int(np.median(angles)) if angles else None
    period = float(np.median(periods)) if periods else None
    return dict(orientation=angle,period=period,orientation_correct=angle==row['expected_angle'],
        period_relative_error=abs(period-row['expected_period'])/row['expected_period'] if period else 1.,
        valid_patches=len(angles))


def paired(values, seed=110142):
    values = np.asarray(values,float)
    if len(values)==0:
        return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(n=len(values),mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist())


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation
    from tools import t3_response_run as response
    from tools.pattern_fidelity_core import gray_rgb
    from models.t3_local_texture import reference_consistency
    from tools.t3_local_pilot import image_tensor, samples, ref_samples
    protocol=read(ROOT/'protocol.json')
    rows=read(ROOT/'inputs/evaluation.json')
    response.ROOT=ROOT/'audit/dists'
    model=response.load_dists('cuda')
    write(ROOT/'audit/dists_positive_control.json',response.sanity(model))
    tensor=lambda patches:torch.from_numpy(np.stack(patches).transpose(0,3,1,2).copy()).float().cuda()/255
    def feats(patches):
        return model.forward_once(tensor([gray_rgb(p) for p in patches]))
    cache={}
    def ref(path):
        if path not in cache:
            im=np.asarray(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            cache[path]=feats([im[y:y+128,x:x+128] for x,y in [(0,0),(128,0),(0,128),(128,128)]])
        return cache[path]
    originals={r['identity']:r for r in rows if r['condition']=='original'}
    measurements=[]
    with torch.no_grad():
        for row in rows:
            mask=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
            outside=1-np.asarray(Image.open(row['mask']).convert('L'),float)/255
            latents=[]
            for arm in ARMS:
                path=ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
                meta=read(path.with_suffix('.json'))
                assert meta['sha256']==sha(path) and meta['reference_sha256']==sha(row['reference'])
                latents.append(meta['initial_latent_sha256'])
                pixels=np.asarray(Image.open(path).convert('RGB'))
                struct=compute_structure_preservation(str(path),row['sketch'],mask_bundle=mask)
                rec=dict(id=row['id'],identity=row['identity'],condition=row['condition'],arm=arm,
                    gray_dists=None,gray_margin=None,spectrum=None,
                    iou=float(struct['struct_iou']),edge_f1=float(struct['struct_edge_f1']),
                    background_mae=float((abs(pixels/255.-1)*outside[...,None]).sum()/max(3*outside.sum(),1)),
                    patch_n=len(row['patches']))
                if row['patches']:
                    gen=feats([pixels[y:y+128,x:x+128] for x,y in row['patches']])
                    distance=response.distances(model,gen,ref(row['reference']))
                    rec['gray_dists']=float(.5*(distance.min(0).values.mean()+distance.min(1).values.mean()))
                    rec['spectrum']=float(reference_consistency(samples(image_tensor(path).cuda(),row['patches']),
                                                               ref_samples(image_tensor(row['reference'],(256,256)).cuda())))
                    if row['condition']=='swapped':
                        old=response.distances(model,gen,ref(originals[row['identity']]['reference']))
                        rec['gray_margin']=float(.5*(old.min(0).values.mean()+old.min(1).values.mean()))-rec['gray_dists']
                if row['condition']=='synthetic':
                    rec.update(pattern_measure(pixels,row))
                measurements.append(rec)
            assert len(set(latents))==1,'三个模型没有使用相同初始噪声'
    write(ROOT/'metrics/per_image.json',measurements)
    lookup={(r['arm'],r['condition'],r['id']):r for r in measurements}
    comparisons=[]
    for other in ['T3','residual']:
        for condition in ['original','swapped']:
            for metric in ['gray_dists','gray_margin','spectrum','iou','edge_f1','background_mae']:
                values=[]
                for row in rows:
                    if row['condition']!=condition:continue
                    a=lookup['reference',condition,row['id']][metric]
                    b=lookup[other,condition,row['id']][metric]
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                        values.append(a-b)
                comparisons.append(dict(comparison='reference-'+other,condition=condition,metric=metric,**paired(values)))
    write(ROOT/'metrics/paired.json',comparisons)
    quality=[]
    for other in ['T3','residual']:
        rec=next(r for r in comparisons if r['comparison']=='reference-'+other and r['condition']=='swapped' and r['metric']=='gray_dists')
        quality.append(dict(passed=bool(rec['n']>=13 and
            rec['mean']<=-.002 and rec['ci95'][1]<0),**rec))
    structure=[]
    for rec in comparisons:
        if rec['comparison']!='reference-T3' or rec['metric'] not in ['iou','edge_f1','background_mae']:continue
        if not rec['n']: passed=False
        elif rec['metric']=='iou': passed=rec['ci95'][0]>=-.01
        elif rec['metric']=='edge_f1': passed=rec['ci95'][0]>=-.02
        else: passed=rec['ci95'][1]<=.005
        structure.append(dict(passed=bool(passed),**rec))
    synthetic={}
    for arm in ARMS:
        part=[r for r in measurements if r['arm']==arm and r['condition']=='synthetic']
        synthetic[arm]=dict(n=len(part),orientation_accuracy=float(np.mean([r['orientation_correct'] for r in part])),
            period_relative_error=float(np.mean([r['period_relative_error'] for r in part])))
    synthetic_pass=synthetic['reference']['orientation_accuracy']>=.75 and synthetic['reference']['period_relative_error']<=.2
    frozen=all(read(ROOT/'training'/arm/'complete.json')['frozen_unchanged'] for arm in ['residual','reference'])
    same_stream=read(ROOT/'training/residual/complete.json')['stream_sha256']==read(ROOT/'training/reference/complete.json')['stream_sha256']
    decision=dict(result='GO' if all(r['passed'] for r in quality+structure) and synthetic_pass and frozen and same_stream else 'NO_GO',
        quality=quality,structure=structure,synthetic=synthetic,synthetic_pass=synthetic_pass,
        frozen=frozen,same_training_stream=same_stream,
        scope='探索性单seed小验证；真实集已查看，合成集仅两张草图；没有独立确认或纹样身份准确率',
        next_action='通过后再评估独立确认方案；本脚本不自动扩大训练')
    write(ROOT/'decision.json',decision)
    summary=[]
    for arm in ARMS:
        for condition in ['original','swapped','synthetic']:
            part=[r for r in measurements if r['arm']==arm and r['condition']==condition]
            rec=dict(arm=arm,condition=condition,n=len(part))
            for metric in ['gray_dists','gray_margin','spectrum','iou','edge_f1','background_mae']:
                vals=[r[metric] for r in part if r[metric] is not None and np.isfinite(r[metric])]
                rec[metric]=float(np.mean(vals)) if vals else None
                rec[metric+'_n']=len(vals)
            summary.append(rec)
    write(ROOT/'metrics/summary.json',summary)
    # 输入哈希固定案例与全部合成例；不按哪个模型获胜选图。
    chosen=[r for r in rows if r['condition']=='swapped'][:6]+[r for r in rows if r['condition']=='synthetic']
    for page,start in enumerate(range(0,len(chosen),4),1):
        part=chosen[start:start+4]
        panel=Image.new('RGB',(1000,270*len(part)), 'white')
        draw=ImageDraw.Draw(panel)
        for j,row in enumerate(part):
            paths=[row['sketch'],row['reference']]+[ROOT/'outputs'/a/row['condition']/(row['id']+'.png') for a in ARMS]
            for col,(label,path) in enumerate(zip(['Sketch','Reference']+ARMS,paths)):
                image=Image.open(path).convert('RGB')
                image.thumbnail((190,235),Image.Resampling.LANCZOS)
                panel.paste(image,(col*200,j*270+30))
                draw.text((col*200+4,j*270+3),label,fill='black')
            draw.text((4,j*270+251),row['id'],fill='black')
        folder=ROOT/'figures';folder.mkdir(parents=True,exist_ok=True)
        panel.save(folder/('cases_%02d.png'%page))
    print('DECISION',json.dumps(decision,ensure_ascii=False),flush=True)
