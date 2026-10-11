"""E41 固定参数、来源分组和160张生成预算。"""
from itertools import product
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.e39_protocol import read,write,sha,E5,EXPECTED_E5
from tools.e41_attributes import ATTRS,LEVELS,BASE,render,extract,error

OUT=Path('output_eval/e41_attribute_bottleneck_20261011')
E39=Path('output_eval/e39_reference_trajectory_20261011')
E40=Path('output_eval/e40_pattern_identity_20261011')
TIMES=[981,801,601,401,201,41,1]
LAYERS=['clip_patches','cnn3_pool8','encoder_fused','token_resampler','token_bf','token_tcpm']


def conditions():
    result=[dict(name='base',attribute=None,params=dict(BASE))]
    for attribute in ATTRS:
        for value in LEVELS[attribute]:
            if abs(value-BASE[attribute])<1e-6:continue
            params=dict(BASE);params[attribute]=value
            result.append(dict(name=attribute+'_'+str(value).replace('.','p'),attribute=attribute,params=params))
    assert len(result)==10;return result


def controlled_records():
    for values in product(*[LEVELS[k] for k in ATTRS]):
        params=dict(zip(ATTRS,values));family='_'.join('%s%g'%(k[0],v) for k,v in params.items())
        for phase_i,phase in enumerate([0.,.19,.37,.57]):
            for color in [0,1]:
                yield dict(id=family+'_p%d_c%d'%(phase_i,color),family=family,fold=phase_i,
                    phase=phase,color=color,shape='ellipse',params=params,domain='controlled')


def calibration():
    records=[]
    # 仅输入几何校准；阈值预先固定，失败的属性不进入肯定性瓶颈定位。
    for r in controlled_records():
        metrics=extract(render(r['params'],r['phase'],r['color']))
        records.append(dict(**r,metrics=metrics,errors={a:error(metrics[a],r['params'][a],a) for a in ATTRS}))
    summary={}
    limits=dict(orientation=10.,period=.15,scale=.20,density=.15)
    for a in ATTRS:
        vals=[r['errors'][a] for r in records if r['errors'][a] is not None]
        mae=float(np.mean(vals)) if vals else None;coverage=len(vals)/len(records)
        summary[a]=dict(mae=mae,coverage=coverage,limit=limits[a],pass_calibration=mae is not None and mae<=limits[a] and coverage>=.80)
    return dict(records=records,summary=summary,thresholds_frozen=True,
        note='Full factorial parameter truth; scale spectral calibration uses isolated ellipses, natural images remain proxies. Occupancy changes superlattice frequencies and contrast statistics; off-target extractor drift is explicitly retained.')


def prepare():
    assert sha(E5)==EXPECTED_E5
    if (OUT/'protocol.json').exists():return
    OUT.mkdir(parents=True,exist_ok=True);inputs={str(E5):sha(E5)};refs=[]
    cal=calibration();write(OUT/'calibration.json',cal)
    for r in controlled_records():
        path=OUT/'references'/(r['id']+'.png');path.parent.mkdir(exist_ok=True)
        render(r['params'],r['phase'],r['color']).save(path)
        refs.append(dict(**r,path=str(path),sha256=sha(path)))
    # 同一参数族的相位/配色全部同折。108参数组合以固定种子分组4折。
    families=sorted(set(r['family'] for r in refs));order=np.random.default_rng(41042).permutation(len(families))
    folds={families[i]:j%4 for j,i in enumerate(order)}
    for r in refs:r['fold']=folds[r['family']]
    real=[r for r in read(E40/'references.json') if r['domain']=='real' and r['variant']=='base']
    for r in real:
        inputs[r['source']]=sha(r['source'])
        for j,angle in enumerate([0,30,45,90]):
            image=Image.open(r['source']).convert('RGB').resize((256,256))
            if angle:image=image.rotate(angle,resample=Image.Resampling.BILINEAR)
            path=OUT/'references'/('real_'+r['id'].replace('/','_')+'_r%d.png'%angle);image.save(path)
            refs.append(dict(id=path.stem,path=str(path),sha256=sha(path),family=r['family'],fold=r['fold'],
                domain='real',params=extract(image),rotation=angle,labels='input extractor pseudo-labels, not independent physical ground truth'))
    carriers=[];seen=set()
    for row in read(E40/'generation.json'):
        if row['carrier'] in seen:continue
        seen.add(row['carrier']);carriers.append(row)
    assert len(carriers)==16
    generation=[];review=OUT/'review';review.mkdir(exist_ok=True)
    for ci,c in enumerate(conditions()):
        path=OUT/'generation_references'/(c['name']+'.png');path.parent.mkdir(exist_ok=True)
        image=render(c['params'],phase=.23,color=0);image.save(path)
        write(path.with_suffix('.json'),dict(**c,measured=extract(image)))
        for row in carriers:
            inputs[row['sketch']]=sha(row['sketch'])
            generation.append(dict(id=row['carrier']+'_'+c['name'],carrier=row['carrier'],condition=c['name'],
                attribute=c['attribute'],params=c['params'],reference=str(path),sketch=row['sketch'],caption=row['caption']))
    assert len(generation)==160
    canvas=Image.new('RGB',(5*210,2*250),'white');draw=ImageDraw.Draw(canvas)
    for i,c in enumerate(conditions()):
        x=i%5*210;y=i//5*250;im=Image.open(OUT/'generation_references'/(c['name']+'.png'))
        canvas.paste(im.resize((204,204)),(x,y));draw.text((x,y+207),c['name'],fill='black')
    canvas.save(review/'controlled_interventions.jpg')
    write(OUT/'references.json',refs);write(OUT/'generation.json',generation);write(OUT/'input_hashes.json',inputs)
    write(OUT/'protocol.json',dict(experiment='E41',seed=42,cfg=7,ddim_steps=50,generation_updates=0,
        E5_sha256=EXPECTED_E5,levels=LEVELS,baseline=BASE,timesteps=TIMES,generation_images=160,
        representation_layers=LAYERS,controlled_reference_views=864,real_sources=len(real),
        attribute_definitions=dict(orientation='axial motif orientation modulo180 degrees',period='lattice centre spacing in pixels',
            scale='motif major diameter in pixels; multiscale spectral width proxy',density='occupied lattice site fraction; edge density secondary'),
        probe='train-only scale and PCA<=32; fixed Ridge alpha1, 4 grouped OOF folds; null and shuffled-label controls',
        calibration_thresholds=cal['summary'],score_weights=[.25]*4,
        score='Erep normalized against constant null; Econd epsilon response deficit proxy; Edenoise calibrated early-to-late deterioration; Ergb calibrated normalized error; missing stage => composite NA; weights sensitivity required',
        confounding='Occupancy superstructure can alter spectra. Single changed rendering parameter is not guaranteed to change only one measured descriptor.',
        real_evidence='Only input-derived pseudo-labels, transformations and read-only E39 trajectory/RGB reuse; no fabric physical scale truth',
        decoder='VAE reference roundtrip positive control and independently calibrated latent readout required; uncalibrated generated latent remains OOD',
        scope='No new method training. No forced dominant attribute if uncertainty or measurement validity prevents identification.'))
    print('E41 PREPARED',len(refs),len(generation),cal['summary'],flush=True)


if __name__=='__main__':prepare()
