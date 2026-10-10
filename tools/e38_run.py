"""E38 CPU 可行性实验：固定扰动、三个表示、最终 RGB 配对统计。"""
import sys
import time
import random
import shutil
import platform
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata
from PIL import Image, ImageDraw
from tools.e38_protocol import *
from tools.e38_representation import *

NAMES = CONFIG['candidates']+['one_minus_IoU']


def load(path):
    return np.asarray(Image.open(path).convert('RGB').resize(SIZE, Image.Resampling.BICUBIC))


def save(path, rgb):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def prepare():
    if (OUT/'protocol_locked.json').exists():
        assert read(OUT/'protocol_locked.json') == CONFIG
        return read(OUT/'splits/dev32.json')
    rows = read(SOURCE/'splits/dev32.json')
    assert len(rows) == 32 and len({r['id'] for r in rows}) == 32
    write(OUT/'protocol_locked.json', CONFIG); write(OUT/'splits/dev32.json', rows)
    frozen = {}
    prior = read(SOURCE/'audit/input_hashes.json')
    for r in rows:
        for kind in ['gt', 'sketch', 'reference']:
            path = Path(r[kind]); frozen[str(path)] = sha(path)
            assert frozen[str(path)] == prior[str(path)]
            dest = OUT/'inputs'/r['id']/(kind+path.suffix)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        for arm in ['A0_E5_OFF', 'B1_CONV']:
            path = SOURCE/'s1/full_rgb/images'/arm/(r['id']+'.png')
            frozen[str(path)] = sha(path)
            shutil.copyfile(path, OUT/'inputs'/r['id']/(arm+'.png'))
    for p, digest in prior.items():
        if 'joint_model.pt' in p or 'step0800.pt' in p or 'step0400.pt' in p:
            assert sha(p) == digest; frozen[p] = digest
    for path in [SOURCE/'s1/full_rgb/metrics.json', SOURCE/'s1/full_rgb/params.json', SOURCE/'splits/dev32.json',
                 Path('data/train_bf_texture.json'), Path('eval/benchmarks/phase1_bf_val_split.json')]:
        frozen[str(path)] = sha(path)
    write(OUT/'audit/input_hashes.json', frozen)
    write(OUT/'audit/implementation.json', dict(commit=commit(), python=sys.version, platform=platform.platform(),
          numpy=np.__version__, opencv=cv2.__version__, source_hashes={str(p):sha(p) for p in Path('tools').glob('e38*.py')},
          training_updates=0, gpu_used=False, original_rgb_count=64, dataset_modified=False, weights_modified=False))
    write(OUT/'decision.json', dict(status='running', go=None, new_method_supported=False, training_updates=0))
    return rows


def blind_sheets(rows):
    order = list(rows); rng = random.Random(38099); rng.shuffle(order); key = []
    for i, r in enumerate(order):
        arms = ['A0_E5_OFF', 'B1_CONV']; rng.shuffle(arms)
        key.append(dict(review_id='R%02d'%(i+1), id=r['id'], X=arms[0], Y=arms[1]))
    write(OUT/'blind/key_do_not_read_before_ratings.json', key)
    for page in range(8):
        board = Image.new('RGB', (900, 1080), 'white'); draw = ImageDraw.Draw(board)
        for j, rec in enumerate(key[page*4:page*4+4]):
            row = next(r for r in rows if r['id'] == rec['id']); folder = OUT/'inputs'/rec['id']
            paths = [row['sketch'], row['gt'], row['reference'], str(folder/(rec['X']+'.png')), str(folder/(rec['Y']+'.png'))]
            for k, (label, path) in enumerate(zip(['Sketch','GT','Reference','X','Y'], paths)):
                image = Image.open(path).convert('RGB').resize((180,240))
                board.paste(image, (k*180,j*270+25)); draw.text((k*180+4,j*270+5),rec['review_id']+' '+label,fill='black')
        board.save(OUT/'blind'/('page%02d.png'%(page+1)))
    write(OUT/'blind/rating_template.json', [dict(review_id=k['review_id'], X=None, Y=None) for k in key])


def auc(app, structure):
    a = np.asarray(app); b = np.asarray(structure)
    return float(((b[:,None] > a[None,:])+.5*(b[:,None] == a[None,:])).mean())


def boot_mean(values):
    a = np.array(values, dtype=float)
    if not len(a): return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(38042)
    means = a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(n=len(a), mean=float(a.mean()), ci95=np.quantile(means,[.025,.975]).tolist())


def association(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return dict(n=len(x), rho=None, ci95=None, valid_bootstrap=0)
    rho = float(spearmanr(x,y).statistic)
    rng = np.random.default_rng(38042); bs = []
    for _ in range(20):
        ii = rng.integers(0,len(x),(500,len(x)))
        a,b = rankdata(x[ii],axis=1),rankdata(y[ii],axis=1)
        a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
        den=np.sqrt((a*a).sum(1)*(b*b).sum(1)); ok=den>0
        bs.extend(((a*b).sum(1)[ok]/den[ok]).tolist())
    return dict(n=len(x),rho=rho,ci95=np.quantile(bs,[.025,.975]).tolist() if bs else None,valid_bootstrap=len(bs))


def perturbation_experiments(rows):
    records=[]; audit=[]
    for index,r in enumerate(rows):
        rgb=load(r['gt']);base,valid,diag=foreground(rgb)
        item=dict(id=r['id'],base_valid=valid,base_diag=diag,garment_type=garment_type(r['caption']),graph=graph(base,r['caption']))
        audit.append(item)
        # 诊断图始终输出，包括失败 mask，避免只展示成功样本。
        overlay=rgb.copy();overlay[cv2.morphologyEx(base.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=[255,0,0]
        for label,pt in item['graph']['nodes'].items():
            cv2.circle(overlay,tuple(np.round(pt).astype(int)),4,(0,255,0),-1)
            cv2.putText(overlay,label,tuple(np.round(pt).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        save(OUT/'audit/masks_graph'/('%02d_%s.png'%(index,r['id'])),overlay)
        if not valid: continue
        original=representations(base,r['caption'])
        for spec in CONFIG['appearance']+CONFIG['structure']:
            kind,level=spec['kind'],spec['level']
            if kind=='sleeve' and garment_type(r['caption'])!='sleeved_upper':continue
            changed,oracle_mask=perturb(rgb,base,kind,level)
            extracted,ev,ediag=foreground(changed)
            oracle=representations(oracle_mask,r['caption']);obs=representations(extracted,r['caption']) if ev else None
            for pipeline,rep in [('oracle',oracle),('rgb',obs)]:
                records.append(dict(id=r['id'],kind=kind,level=level,pipeline=pipeline,
                    family='appearance' if kind in ['hue','texture','brightness'] else 'structure',
                    valid=rep is not None, distances={name:distance(original[name],rep[name],name) if rep is not None else None for name in NAMES}))
            if index<8:
                save(OUT/'perturbation_previews'/r['id']/('%s_%s.png'%(kind,level)),changed)
    write(OUT/'audit/extraction.json',audit);write(OUT/'perturbations/records.json',records)
    reports={}
    for pipeline in ['oracle','rgb']:
        reports[pipeline]={}
        for name in NAMES:
            per=[]
            for r in rows:
                rr=[v for v in records if v['id']==r['id'] and v['pipeline']==pipeline]
                aa=[v['distances'][name] for v in rr if v['family']=='appearance']
                ss=[v['distances'][name] for v in rr if v['family']=='structure']
                if not aa or not ss or any(v is None for v in aa+ss):continue
                by_family={kind:dict(min_distance=min(v['distances'][name] for v in rr if v['kind']==kind),
                    structure_over_appearance=min(v['distances'][name] for v in rr if v['kind']==kind)>max(aa)+1e-12)
                    for kind in ['sleeve','hem','contour'] if any(v['kind']==kind for v in rr)}
                per.append(dict(id=r['id'],auc=auc(aa,ss),max_appearance=max(aa),min_structure=min(ss),
                                separated=min(ss)>max(aa)+1e-12,families=by_family))
            families={kind:dict(n=sum(kind in v['families'] for v in per),
                separation_rate=float(np.mean([v['families'][kind]['structure_over_appearance'] for v in per if kind in v['families']])) if any(kind in v['families'] for v in per) else None)
                for kind in ['sleeve','hem','contour']}
            coverage=len(per)/32;sep=float(np.mean([v['separated'] for v in per])) if per else None
            a=boot_mean([v['auc'] for v in per])
            passed=coverage>=.8 and sep is not None and sep>=.8 and a['mean']>=.9 and all(f['separation_rate'] is None or f['separation_rate']>=.8 for f in families.values())
            reports[pipeline][name]=dict(fixed_n=32,valid_n=len(per),coverage=coverage,separation_rate=sep,auc=a,families=families,pass_gate=bool(passed),identities=per)
        baseline={v['id']:v for v in reports[pipeline]['one_minus_IoU']['identities']}
        for name in CONFIG['candidates']:
            rr=reports[pipeline][name]
            differences=[v['auc']-baseline[v['id']]['auc'] for v in rr['identities'] if v['id'] in baseline]
            rr['auc_gain_vs_iou']=boot_mean(differences)
            rr['synthetic_superiority']=bool(differences and rr['auc_gain_vs_iou']['ci95'][0]>0)
    write(OUT/'perturbations/report.json',reports)
    return reports


def prediction_experiment(rows):
    prior=read(SOURCE/'s1/full_rgb/metrics.json'); prior={(v['id'],v['arm']):v for v in prior}
    records=[]
    for r in rows:
        masks={};reps={};valids={};diagnostics={}
        for kind in ['gt','sketch','A0_E5_OFF','B1_CONV']:
            path=r[kind] if kind in ['gt','sketch'] else OUT/'inputs'/r['id']/(kind+'.png')
            if kind=='sketch':
                from garment_mask_utils import build_sketch_garment_mask
                mask_img,diag=build_sketch_garment_mask(Image.open(path),*SIZE)
                mask=np.asarray(mask_img)>127;valid=not diag['mask_low_confidence'] and not diag['mask_fallback']
            else:mask,valid,diag=foreground(load(path))
            masks[kind]=mask;valids[kind]=valid;diagnostics[kind]=diag
            reps[kind]=representations(mask,r['caption']) if valid else None
        for name in NAMES:
            def dist(a,b):
                return distance(reps[a][name],reps[b][name],name) if reps[a] is not None and reps[b] is not None else None
            e5,b1=dist('A0_E5_OFF','gt'),dist('B1_CONV','gt')
            es,bs=dist('A0_E5_OFF','sketch'),dist('B1_CONV','sketch')
            a,b=prior[(r['id'],'A0_E5_OFF')],prior[(r['id'],'B1_CONV')]
            def delta(key,sign=1):
                v,w=a.get(key),b.get(key)
                return sign*(w-v) if v is not None and w is not None and np.isfinite([v,w]).all() else None
            records.append(dict(id=r['id'],candidate=name,E5_distance=e5,B1_distance=b1,delta_distance=b1-e5 if e5 is not None and b1 is not None else None,
                delta_distance_to_sketch=bs-es if es is not None and bs is not None else None,
                minus_delta_IoU=delta('struct_iou',-1),minus_delta_EdgeF1=delta('struct_edge_f1',-1),delta_LPIPS=delta('lpips_gt'),
                source_mask_valid=bool(a['mask_valid'] and b['mask_valid']),rgb_valids=valids))
    reports={}
    for name in NAMES:
        rr=[v for v in records if v['candidate']==name]; comparisons={}
        for target in ['minus_delta_IoU','minus_delta_EdgeF1','delta_LPIPS']:
            # LPIPS 不依赖原 sketch mask，但结构关联必须保留原指标的有效 mask 分母。
            valid=[v for v in rr if v['delta_distance'] is not None and v[target] is not None and (target=='delta_LPIPS' or v['source_mask_valid'])]
            comparisons[target]=association([v['delta_distance'] for v in valid],[v[target] for v in valid])
            comparisons[target]['valid_ids']=[v['id'] for v in valid]
        main=[comparisons[k] for k in ['minus_delta_IoU','minus_delta_EdgeF1']]
        passed=all(c['n']/32>=.8 and c['rho'] is not None and c['rho']>.4 for c in main)
        reports[name]=dict(fixed_n=32,comparisons=comparisons,pass_gate=bool(passed))
    write(OUT/'prediction/records.json',records);write(OUT/'prediction/report.json',reports)
    return reports


def run():
    start=time.monotonic();rows=prepare();blind_sheets(rows)
    print('E38 prepared input hashes and blinded review sheets',flush=True)
    pert=perturbation_experiments(rows)
    print('E38 stability/sensitivity complete',flush=True)
    pred=prediction_experiment(rows)
    candidates={}
    for name in CONFIG['candidates']:
        checks=dict(oracle_invariance_sensitivity=pert['oracle'][name]['pass_gate'],
                    rgb_invariance_sensitivity=pert['rgb'][name]['pass_gate'],
                    full_rgb_prediction=pred[name]['pass_gate'],
                    synthetic_superiority=pert['rgb'][name]['synthetic_superiority'],
                    blinded_real_superiority=None,
                    semantic_graph_validated=None if name=='C_GRAPH' else True)
        # 只有全部数值门槛通过才等待盲评/语义审核；任何失败不通过补样本凑门槛。
        numeric=all(v for k,v in checks.items() if k not in ['blinded_real_superiority','semantic_graph_validated'])
        candidates[name]=dict(checks=checks,numeric_pass=bool(numeric),go=None if numeric else False,
                              status='await_visual_review' if numeric else 'no_go')
    assert all(sha(p)==h for p,h in read(OUT/'audit/input_hashes.json').items())
    write(OUT/'audit/inputs_unchanged.json',dict(pass_unchanged=True,count=len(read(OUT/'audit/input_hashes.json'))))
    result=dict(experiment='E38_Prior',protocol=CONFIG,perturbations=pert,prediction=pred,candidates=candidates,
        training_updates=0,new_method_supported=False,confirm96_used=False,validation500_used=False)
    write(OUT/'E38_structure_representation_eval.json',result)
    write(OUT/'decision.json',dict(status='await_visual_review' if any(v['numeric_pass'] for v in candidates.values()) else 'no_go',
        go=None if any(v['numeric_pass'] for v in candidates.values()) else False,candidates=candidates,
        training_updates=0,novel_method_supported=False,structure_supervision_training='not_run',
        independent_generalization='not_run',stop_reason='no candidate passes frozen gates' if not any(v['numeric_pass'] for v in candidates.values()) else None))
    write(OUT/'resource_runtime.json',dict(elapsed_seconds=time.monotonic()-start,gpus=0,gpu_hours=0,commit=commit()))
    print('E38 evaluation complete; decision written',flush=True)


if __name__=='__main__': run()
