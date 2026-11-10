"""冻结诊断的分组线性读出、原v3公式复核和独立归档；不伪造训练臂。"""
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from models.e2_texture_diagnostics import PATTERNS, grouped_ridge, group_accuracy
from tools.e2_fft_experiment import read, write, sha
from tools.e2_texture_bottleneck import (BASELINE_SHA, PAIRS_SHA, SHORT_GROUPS,
    initialize, diagnostic_environment)


def descriptive(values):
    """每个身份只有一个聚合值，时间步/patch不是独立样本。"""
    a=np.asarray([x for x in values if x is not None and np.isfinite(x)],np.float64)
    if not len(a):return dict(n=0,mean=None,descriptive_ci95=None)
    rng=np.random.default_rng(110142)
    boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),descriptive_ci95=np.quantile(boot,[.025,.975]).tolist())


def verify_cases(out):
    spec=read(out/'protocol.json');pairs=read(out/'strict_pairs.json')['pairs']
    assert len(pairs)==32 and sha(out/'strict_pairs.json')==PAIRS_SHA
    assert sha(spec['checkpoint'])==BASELINE_SHA and read(out/'audit/pilot.json')['passed']
    seen=[];digests=[]
    for i in range(2):
        audit=read(out/'audit'/('shard_%d.json'%i))
        assert audit['passed'] and audit['generator_updates']==0
        assert audit['weights_before']==audit['weights_after'] and audit['checkpoint_sha256']==BASELINE_SHA
        assert audit['environment']['commit']==spec['environment']['commit']
        seen.extend(audit['cases']);digests.append(audit['weights_after'])
    assert len(seen)==32 and set(seen)=={p['case_id'] for p in pairs} and len(set(digests))==1
    for pair in pairs:
        folder=out/'cases'/pair['case_id'];record=read(folder/'result.json')
        assert record['passed'] and record['generator_updates']==0
        assert record['features_sha256']==sha(folder/'features.npz')
        assert record['downstream_sha256']==sha(folder/'downstream.npz')
        assert record['baseline_replay']['max_pixel_error']==0 and record['baseline_replay']['latent_equal']
    return spec,pairs


def probe_arrays(rows, arrays):
    """每条原控制有同身份/同参考的相移控制，不能跨身份或跨参考误配。"""
    lookup={(r['case_id'],r['condition'],r['variant']):i for i,r in enumerate(rows)}
    selected=[i for i,r in enumerate(rows) if r['variant'] in PATTERNS and r.get('readout_eligible',True)]
    assert selected
    shifted=[lookup[(rows[i]['case_id'],rows[i]['condition'],rows[i]['variant']+'_shift')] for i in selected]
    labels=np.array([PATTERNS.index(rows[i]['variant']) for i in selected])
    groups=np.array([rows[i]['identity_group'] for i in selected])
    return selected,shifted,labels,groups,{k:(v[selected],v[shifted]) for k,v in arrays.items()}


def readout(args):
    _,out=initialize(args);spec,pairs=verify_cases(out)
    results=[];predictions=[];eligible={}
    for r in read(out/'inputs/controls.json'):
        eligible[r['case_id'],r['condition']]=r['readout_eligible']
    for stem,row_file in [('features','feature_rows.json'),('downstream','downstream_rows.json')]:
        rows=[];arrays=defaultdict(list)
        for pair in pairs:
            folder=out/'cases'/pair['case_id'];local=read(folder/row_file)
            for r in local:r['readout_eligible']=eligible[r['case_id'],r['condition']]
            rows.extend(local)
            with np.load(folder/(stem+'.npz')) as z:
                for key in z.files:arrays[key].append(z[key])
        arrays={k:np.concatenate(v) for k,v in arrays.items()}
        selected,shifted,labels,groups,values=probe_arrays(rows,arrays)
        for stage,(x,xshift) in values.items():
            for task,y,classes in [('joint',labels,4),('orientation',labels//2,2),('period',labels%2,2)]:
                a,b=grouped_ridge(x,xshift,y,groups,spec['folds'],classes)
                for phase,prediction in [('original',a),('shifted',b)]:
                    results.append(dict(source=stem,stage=stage,task=task,test_phase=phase,chance=1/classes,
                        **group_accuracy(prediction,y,groups)))
                    predictions.extend(dict(source=stem,stage=stage,task=task,test_phase=phase,
                        case_id=rows[i]['case_id'],condition=rows[i]['condition'],variant=rows[i]['variant'],
                        identity_group=rows[i]['identity_group'],fold=spec['folds'][rows[i]['identity_group']],
                        label=int(label),prediction=int(pred)) for i,label,pred in zip(selected,y,prediction))
        if stem=='features':
            controls=[r for r in results if r['stage'].startswith('input_') and r['task']=='joint']
            assert all(r['accuracy']==.25 for r in controls if r['stage'].startswith('input_color'))
            assert all(r['accuracy']>=.95 for r in controls if r['stage'].startswith('input_spectrum'))
    write(out/'metrics/readout_predictions.json',predictions)
    write(out/'metrics/readouts.json',results)
    grouped=defaultdict(list);low_t=defaultdict(list)
    for pair in pairs:
        rows=read(out/'cases'/pair['case_id']/'common_states.json')
        for variant in sorted({r['variant'] for r in rows}):
            matching=[r for r in rows if r['variant']==variant]
            for branch in ['conditional','guided']:
                for region in ['body','boundary','background']:
                    a=[r[branch][region]['relative_rms'] for r in matching if branch in r]
                    a=[v for v in a if v is not None]
                    if a:grouped[variant,branch,region].append(float(np.mean(a)))
        for variant in ['old','new','delta2']:
            a=[r['decoded_fft']['margin'] for r in rows if r['variant']==variant and 'decoded_fft' in r]
            a=[v for v in a if v is not None]
            if a:low_t[variant].append(float(np.mean(a)))
    write(out/'metrics/common_state_summary.json',dict(
        relative_rms=[dict(variant=v,branch=b,region=r,**descriptive(a)) for (v,b,r),a in grouped.items()],
        x0_fft=[dict(variant=v,**descriptive(a)) for v,a in low_t.items()],
        interpretation='RMS measures sensitivity,not pattern readability or fidelity;FFT is a separate exploratory descriptor;mean over timesteps within identity first'))
    write(out/'audit/readout_complete.json',dict(passed=True,generator_updates=0,
        eligible_references=sum(eligible.values()),eligible_identities=len({k[0] for k,v in eligible.items() if v}),
        fold_groups=spec['folds'],rows=len(results),environment=diagnostic_environment()))
    print('TEXTURE_DIAG_READOUT_PASS',len(results),flush=True)


def output_metadata(out,spec,pairs):
    records=[];reused=0
    for pair in pairs:
        latents=[]
        for group in SHORT_GROUPS:
            if group!='baseline' and pair['case_id'] not in spec['short_cases']:continue
            for condition in ['old','new']:
                path=out/'outputs'/group/pair['case_id']/(condition+'.png');meta=read(path.with_suffix('.json'))
                assert meta['group']==group and meta['case_id']==pair['case_id'] and meta['condition']==condition
                assert (meta['index'],meta['seed'],meta['caption'])==(pair['index'],pair['seed'],pair['caption'])
                assert meta['output_sha256']==sha(path) and meta['checkpoint_sha256']==BASELINE_SHA
                assert meta['manifest_sha256']==PAIRS_SHA and meta['generator_updates']==0 and not meta['GT_used_in_inference']
                assert meta['input_sketch_sha256']==pair['hashes']['sketch']
                if meta['reused']:
                    parent=Path(meta['parent_output']);assert sha(parent)==sha(path)
                    assert sha(parent.with_suffix('.json'))==meta['parent_metadata_sha256']
                    key='original_reference' if condition=='old' else 'swapped_reference'
                    assert meta['input_reference_sha256']==pair['hashes'][key];reused+=1
                else:
                    assert meta['input_references']=={k:pair['hashes'][k] for k in ['original_reference','swapped_reference']}
                latents.append(meta['initial_latent_sha256']);records.append(meta)
        assert len(set(latents))==1
    assert len(records)==100 and reused==70
    assert len(list((out/'outputs').rglob('*.png')))==len(list((out/'outputs').rglob('*.json')))==100
    return records


def response_rows(records):
    lookup={(r['group'],r['case_id'],r['condition']):r for r in records};result=[]
    for group,case_id,condition in lookup:
        if condition!='old':continue
        old,new=lookup[group,case_id,'old'],lookup[group,case_id,'new']
        row=dict(group=group,case_id=case_id,identity_group=old['identity_group'])
        for mode in ['gray','rgb']:
            for scale in ['strict','tolerant']:
                row[mode+'_'+scale+'_response_change']=new['scores'][mode][scale+'_margin']-old['scores'][mode][scale+'_margin']
        result.append(row)
    return result


def evaluate(args):
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask
    from eval.metrics import compute_structure_preservation,compute_texture_leakage,compute_texture_color_fidelity
    from tools.e2_a_evaluation import boundary_f1
    from tools.e2a_attr_eval import frozen_lpips
    from tools.bf_t3_three_arm_report import gray
    from tools.e2_followup_data import rgb
    from tools import t3_response_run as response
    core,out=initialize(args);spec,pairs=verify_cases(out)
    assert read(out/'audit/readout_complete.json')['passed']
    metadata=output_metadata(out,spec,pairs)
    response.ROOT=out;dists=response.load_dists(args.device);lpips,lpips_assets=frozen_lpips()
    references={};records=[]
    def feature(parts):
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().to(args.device)/255.
        return dists.forward_once(value)
    def ref(path,scale,is_gray):
        key=(str(path),scale,is_gray)
        if key not in references:
            size=(round(384*scale),round(512*scale));im=np.asarray(rgb(path,size))
            patches=[im[y:y+64,x:x+64] for y in np.linspace(0,size[1]-64,4).round().astype(int)
                for x in np.linspace(0,size[0]-64,4).round().astype(int)]
            references[key]=feature([gray(p) for p in patches] if is_gray else patches)
        return references[key]
    with torch.inference_mode():
        for pair in pairs:
            sketch=Path(args.data)/pair['sketch'];gt=Path(args.data)/pair['gt']
            for key in ['sketch','gt','original_reference','swapped_reference']:
                assert sha(Path(args.data)/pair[key])==pair['hashes'][key]
            masks=prepare_evaluation_masks((384,512),sketch_path=str(sketch),mask_policy='sketch_only')
            for meta in [m for m in metadata if m['case_id']==pair['case_id']]:
                group,condition=meta['group'],meta['condition'];path=out/'outputs'/group/pair['case_id']/(condition+'.png')
                ref_key='original_reference' if condition=='old' else 'swapped_reference'
                kwargs=dict(gen_path=str(path),sketch_path=str(sketch),target_path=str(gt),mask_bundle=masks)
                structure=compute_structure_preservation(**kwargs)
                im=Image.open(path).convert('RGB');predicted=estimate_foreground_mask(im,(384,512))
                row=dict(group=group,index=pair['index'],case_id=pair['case_id'],identity_group=pair['identity_group'],
                    condition=condition,patch_n=len(pair['patch_coordinates']),pattern=pair['qualification']['old_pattern'],
                    edge_f1=structure['edge_f1'],iou=structure['sketch_iou'],edge_l1=structure['edge_l1'],
                    leak=compute_texture_leakage(**kwargs)['leak_colored_frac'],lpips=None,
                    tcf_lab=compute_texture_color_fidelity(texture_path=str(Path(args.data)/pair[ref_key]),**kwargs)['tcf_lab_delta'],
                    boundary_f1_r2=boundary_f1(predicted,masks['garment'],2),scores={})
                if condition=='old':row['lpips']=float(lpips(core.tensor(path).to(args.device),core.tensor(gt).to(args.device)).mean())
                row={k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in row.items()}
                image=np.asarray(im);parts=[image[y:y+64,x:x+64] for x,y in pair['patch_coordinates']]
                for is_gray in [False,True]:
                    f=feature([gray(p) for p in parts] if is_gray else parts)
                    old={str(s):float(response.distances(dists,f,ref(Path(args.data)/pair['original_reference'],s,is_gray)).mean()) for s in [.75,1.,1.33]}
                    new={str(s):float(response.distances(dists,f,ref(Path(args.data)/pair['swapped_reference'],s,is_gray)).mean()) for s in [.75,1.,1.33]}
                    row['scores']['gray' if is_gray else 'rgb']=dict(old=old,new=new,strict_margin=old['1.0']-new['1.0'],
                        tolerant_margin=min(old.values())-min(new.values()),old_best_scale=min(old,key=old.get),new_best_scale=min(new,key=new.get))
                records.append(row);print('TEXTURE_DIAG_EVALUATED',group,pair['case_id'],condition,flush=True)
            write(out/'metrics/per_image_progress.json',records)
    parent={(r['case_id'],r['condition']):r for r in read(Path(args.parent)/'metrics/per_image.json') if r['arm']=='B0'}
    errors=[]
    for row in records:
        if row['group']!='baseline':continue
        old=parent[row['case_id'],row['condition']]
        assert row['scores']==old['scores'],'原v3逐图分数回归失败'
        for key in ['edge_f1','iou','edge_l1','leak','tcf_lab','lpips']:
            assert row[key]==old[key],('旧保护指标回归失败',key,row['case_id'])
        errors.append(0.)
    responses=response_rows(records);lookup={(r['group'],r['case_id']):r for r in responses}
    images={(r['group'],r['case_id'],r['condition']):r for r in records};comparisons=[]
    for group in SHORT_GROUPS[1:]:
        for metric in ['gray_strict_response_change','gray_tolerant_response_change','rgb_strict_response_change','rgb_tolerant_response_change']:
            delta=[lookup[group,c][metric]-lookup['baseline',c][metric] for c in spec['short_cases']]
            comparisons.append(dict(group=group,metric=metric,comparison=group+'_minus_baseline',**descriptive(delta)))
        for condition in ['old','new']:
            for metric in ['edge_f1','iou','leak','tcf_lab','lpips','boundary_f1_r2']:
                delta=[images[group,c,condition][metric]-images['baseline',c,condition][metric] for c in spec['short_cases']
                    if images[group,c,condition][metric] is not None and images['baseline',c,condition][metric] is not None]
                comparisons.append(dict(group=group,metric=condition+'_'+metric,comparison=group+'_minus_baseline',**descriptive(delta)))
    write(out/'metrics/per_image.json',records);write(out/'metrics/responses.json',responses)
    write(out/'metrics/descriptive_comparisons.json',comparisons)
    write(out/'audit/baseline_metric_regression.json',dict(passed=True,outputs=len(errors),max_absolute_error=max(errors)))
    write(out/'metrics/version.json',dict(protocol=spec['version'],outputs=len(records),metadata=len(metadata),baseline_identities=32,
        intervention_identities=6,interventions=3,baseline_recomputed_old_v3_exact=True,
        dists_distance='unchanged v3 all fixed crosspatch mean;gray legacy bounded contrast proxy',
        statistics='identity-level descriptive paired bootstrap10000 seed110142;no p-values/no confirmatory family',
        generator_updates=0,lpips_assets=lpips_assets,environment=diagnostic_environment()))
    print('TEXTURE_DIAG_EVALUATE_PASS',len(records),flush=True)


def bundle(args):
    out=Path(args.out);assert read(out/'metrics/version.json')['outputs']==100
    verify_cases(out);output_metadata(out,read(out/'protocol.json'),read(out/'strict_pairs.json')['pairs'])
    assert read(out/'audit/baseline_metric_regression.json')['passed']
    prefixes=['inputs','audit','cases','reused_evidence','outputs','metrics','logs','tests','x0']
    files=sorted({p for prefix in prefixes for p in (out/prefix).rglob('*') if p.is_file()
        and p.suffix in ['.npz','.json','.jsonl','.png','.jpg','.jpeg','.txt','.out','.err']})
    files += [out/'strict_pairs.json',out/'protocol.json']
    write(out/'bundle_manifest.json',{p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with zipfile.ZipFile(out/'review_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'bundle_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    write(out/'bundle_info.json',dict(bytes=(out/'review_results.zip').stat().st_size,sha256=sha(out/'review_results.zip'),files=len(files)))
    print('TEXTURE_DIAG_BUNDLE_PASS',sha(out/'review_results.zip'),flush=True)
