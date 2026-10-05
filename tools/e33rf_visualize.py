"""固定选择与跨阶段面板；RF3未运行时明确标注，不用其他阶段代替。"""
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from data.e33rf_real_rotation_dataset import rotate
from tools.e33r_visualize import direction,scalar,fit_panel
from tools.e33rf_common import *

TRACKED=['182400888','165558044','169695916','147663211']
def fixed(rows,salt):
    return sorted(rows,key=lambda r:hashlib.sha256((salt+r['id']).encode()).hexdigest())[:16]
def selection(rows,baseline):
    valid=[r for r in rows if r['near_advantage'] is not None]
    old={r['id']:r for r in baseline}
    group=dict(near_worst16=[r['id'] for r in sorted(valid,key=lambda r:(r['near_advantage'],r['id']))[:16]],
        near_best16=[r['id'] for r in sorted(valid,key=lambda r:(-r['near_advantage'],r['id']))[:16]],
        rot90_worst16=[r['id'] for r in sorted(valid,key=lambda r:(-r['rot90_response_error'],r['id']))[:16]],
        fixed_hash16=[r['id'] for r in fixed(rows,'E33RF/visual/hash/')],
        rf0_success_to_failure=[r['id'] for r in fixed([r for r in rows if old[r['id']]['rot90_response_success'] is True and r['rot90_response_success'] is False],'E33RF/visual/lost/')],
        rf0_success_to_success=[r['id'] for r in fixed([r for r in rows if old[r['id']]['rot90_response_success'] is True and r['rot90_response_success'] is True],'E33RF/visual/retained/')],
        tracked=[sid for sid in TRACKED if sid in old])
    return dict(groups=group,unique_ids=sorted(set(sum(group.values(),[]))),selection='fixed extremes/hash/cohorts; no manual reselection',tracked_absent=[sid for sid in TRACKED if sid not in old])
def fields(path,sid):
    with np.load(path/'real/fields'/(sid+'.npz')) as z:return {k:np.asarray(z[k]) for k in z.files}
def create_panels(stage,seed):
    baseline=OUT/'RF0_reproduction'/('seed%d'%seed)
    rows=read(stage/'real/rows.json');chosen=selection(rows,read(baseline/'real/rows.json'))
    visual=stage/'visual_audit';visual.mkdir(exist_ok=True);write(visual/'selection.json',chosen)
    split=read(OUT/'split_manifest.json');lookup={r['id']:r for r in split['dev']}
    donors={r['id']:r for r in split['dev']+split['confirmation_all']}
    metrics={r['id']:r for r in rows};cfrows=read(OUT/'controlled_manifest.json')['dev']
    cfids={r['id'] for r in cfrows};fallback=hash_order(cfrows,'E33RF/visual/controlled-probe')[0]['id']
    rf2=folder(seed,'RF2');rf3=folder(seed,'RF3')
    missing=Image.new('RGB',(200,180),'#eeeeee');ImageDraw.Draw(missing).text((15,70),'NOT RUN / NOT AVAILABLE',fill='black')
    for sid in chosen['unique_ids']:
        row=lookup[sid];p=fields(stage,sid);base=fields(baseline,sid);support=p['support'];gt=p['gt']
        near=metrics[sid]['wrong_references']['color_near'];match=image_at(DATASET/row['reference'])
        second=fields(rf2,sid) if (rf2/'phase_complete.json').exists() else None
        ranked=fields(rf3,sid) if (rf3/'phase_complete.json').exists() else None
        probe=sid if sid in cfids else fallback
        with np.load(stage/'controlled/fields'/(probe+'.npz')) as z:
            cf_ori=np.asarray(z['orientation']);cf_support=np.asarray(z['support'])
        err=np.degrees(np.arccos(np.clip(-(cf_ori[0]*cf_ori[1]).sum(0),-1,1)))/2
        error_pic=np.asarray(scalar(err,90)).copy();error_pic[~cf_support]=235
        residual=np.linalg.norm(.1*p['adapter_residual'][0],axis=-1)
        items=[('target sketch',image_at(DATASET/row['sketch'])),('GT orientation',direction(gt[:2],support)),
            ('matched reference',match),('R90 reference',rotate(match,90)),('R180 reference',rotate(match,180)),
            ('color-near '+near,image_at(DATASET/donors[near]['reference'])),
            ('RC0 matched',direction(base['orientation'][0],support)),('RC0 R90',direction(base['orientation'][1],support)),
            ('current matched',direction(p['orientation'][0],support)),('current R90',direction(p['orientation'][1],support)),
            ('current R180',direction(p['orientation'][2],support)),('current near',direction(p['orientation'][3],support)),
            ('RF2 matched',direction(second['orientation'][0],support) if second else missing),
            ('RF2 R90',direction(second['orientation'][1],support) if second else missing),
            ('RF3 wrong near',direction(ranked['orientation'][3],support) if ranked else missing),
            ('Q_ref current',direction(p['q'][0],support)),
            ('scaled residual norm max=%.4g'%max(float(residual.max()),1e-8),scalar(residual,max(float(residual.max()),1e-8))),
            ('C_ref current',scalar(1/(1+np.exp(-np.clip(p['confidence_logits'][0,0],-60,60))))),
            ('controlled error '+probe,Image.fromarray(error_pic)),('current zero',direction(p['orientation'][5],support))]
        canvas=Image.new('RGB',(5*210,4*275+65),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),'E33-RF '+str(stage.relative_to(OUT))+' '+sid,fill='black')
        label=lambda value:'N/A (empty GT)' if value is None else '%.3f'%value
        draw.text((8,23),'near='+label(metrics[sid]['near_advantage'])+' R90error='+label(metrics[sid]['rot90_response_error']),fill='black')
        draw.text((8,41),'Axial hue [0,180); gray outside GT; confidence is diagnostic, not truth',fill='black')
        for i,(label,pic) in enumerate(items):
            x,y=(i%5)*210+5,(i//5)*275+65;draw.text((x,y),label,fill='black');canvas.paste(fit_panel(pic),(x,y+20))
        canvas.save(visual/(sid+'.png'))
    return chosen['unique_ids'],fallback
