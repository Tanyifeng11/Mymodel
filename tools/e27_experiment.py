"""E27：审核完整服装 C3，依次运行现有基线与 oracle panel 门槛。"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.local_pattern_field import estimate_local_pattern_field, rectify_reference, render_confidence_aware
from models.pattern_coordinate_field import build_affine_pattern_field, sample_reference_with_field
from tools.e27_correspondence import (VERSION, file_sha, local_readout, oracle_scaffold,
    pattern_variant, similarity, target_panel_masks)


ARMS = ('B0_raw_global','B1_global_rectified','B2_original_E5')
SEEDS = (42,43)
THRESHOLDS = {'local_theta_max':20., 'identity_min':.65, 'local_failure_gap':.10,
              'oracle_follow_gain':.10, 'oracle_theta_reduction':.20,
              'delta_contour_f1_min':-.02, 'delta_leakage_max':.01}


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')


def protocol(root):
    return {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'version':VERSION,'seeds':SEEDS,'scheduler':'DDIMScheduler','ddim_steps':50,'cfg':7.,
        'refinement_strength':.15,'texture':'ON','mask_backend':'opencv', 'training_steps':0,
        'arms':ARMS,'thresholds_fixed_before_generation':THRESHOLDS,
        'noise_control':'共享相同原始 Gaussian tensor；B2 从纯噪声跑 50 步，scaffold 臂从 VAE mean 加噪跑最后 8 步；初始 latent 不相同。',
        'counterfactual':'original 保留真实完整服装；rot90 仅旋转各人工源衣片 canonical pattern 并回填同一 mask。',
        'metric_reference':'各臂共用未矫正人工 canonical panel transfer prototype；不是某臂自己的 scaffold；没有真实 dense UV GT。',
        'geometry_readability':'GT patch 不可读的方向不计方向成功；生成不可读的方向计 90deg 失败；报告覆盖率。',
        'patches':'body 64px、窄袖/领/口袋 32px native geometry；FFT/selfsim descriptor 统一 64px。',
        'bootstrap':'reference 为唯一 bootstrap 单位，seed/variants/patches 在 reference 内聚合；2000 重采样 seed42。',
        'pilot':'6 个 reference，2/档，人工预注册；B 对全部18张确认，C 为预注册8张 M/H。',
        'limits':['仅 BF validation 产品图，无强透视/遮挡；不代表一般非刚性形变。',
                  '人工矩形 panel 为近似划分；未训练 parser；只有一个固定 target sketch。',
                  'B2 与 scaffold 臂去噪预算不同，不能将差异归因于 correspondence 单一变量。']}


def prepare(root,out,dataset):
    config=json.loads((root/'data/e27_c3_cases.json').read_text(encoding='utf-8'))
    for folder in ('A_audit/inputs','A_audit/previews','B_baseline/scaffolds','expected'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'data/e27_c3_audit.csv',out/'A_audit/audit.csv')
    old=root/'output_eval/e26_20260929'
    sketch=json.loads((old/'cases.json').read_text())['sketches'][1]
    # 固定具有 body + 两袖 + 领的长袖 target；source 与 target 实物 ID 不相同。
    for key in ('path','mask'):
        source=old/sketch[key]; dest=out/sketch[key];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    mask=np.asarray(Image.open(out/sketch['mask']).convert('L'))>0
    mask_image=Image.fromarray(mask.astype(np.uint8)*255)
    proofs=[]; hashes=set(); ids=set()
    refs=config['references']
    for ref in refs:
        source,smask=dataset/ref['source'],dataset/ref['source_mask']
        assert file_sha(source)==ref['source_sha256'],source
        assert file_sha(smask)==ref['mask_sha256'],smask
        assert ref['source_sha256'] not in hashes and ref['source'] not in ids
        hashes.add(ref['source_sha256']);ids.add(ref['source'])
        assert Path(ref['source']).stem != Path(sketch['source']).stem
        original=Image.open(source).convert('RGB'); srcmask=np.asarray(Image.open(smask).convert('L'))>0
        assert original.size==(256,256) and srcmask.shape==(256,256)
        ref['variants']=[]
        for name in ('original','rot90'):
            prefix=f"c{ref['id']:02d}_{name}"
            image=pattern_variant(original,srcmask,ref['panels'],name)
            path=f'A_audit/inputs/{prefix}.png';image.save(out/path)
            assert np.array_equal(np.asarray(image)[~srcmask],np.asarray(original)[~srcmask])
            ref['variants'].append({'variant':name,'path':path,'theta':0.,'sha256':file_sha(out/path)})
            expected,areas,info=oracle_scaffold(original,mask,ref['panels'],name,rectified=False)
            expected.save(out/'expected'/f'{prefix}.png')
            field=build_affine_pattern_field(mask_image,0.,8.,0.,image.size)
            raw=sample_reference_with_field(image,field,mask_image)
            rect,uv,rect_info=rectify_reference(image)
            local=estimate_local_pattern_field(rect)
            mapped,blend=render_confidence_aware(rect,field,local,mask_image)
            for arm,scaffold in zip(ARMS,(raw,mapped)):
                scaffold.save(out/'B_baseline/scaffolds'/f'{prefix}_{arm}.png')
            np.savez_compressed(out/'B_baseline/scaffolds'/f'{prefix}_fields.npz',
                global_uv=field.uv,rectification_uv=uv,confidence=blend,
                panel_masks=np.stack(areas),source_mask=srcmask,target_mask=mask)
            proofs.append({'case':ref['id'],'variant':name,'source_sha256':file_sha(source),
                'source_mask_sha256':file_sha(smask),'reference_sha256':file_sha(out/path),
                'shape_fixed':True,'background_pixel_equal':True,'rectification':rect_info,
                'confidence_coverage':float((blend[mask]>.5).mean()),'panels':info})
        # 原图、纹样反事实、人工 canonical crop 的科学审计预览。
        sheet=Image.new('RGB',(256*4,286),'white');draw=ImageDraw.Draw(sheet)
        sources=[original,Image.open(out/ref['variants'][1]['path']),expected.resize((256,256)),original.copy()]
        d=ImageDraw.Draw(sources[-1])
        for p in ref['panels']:d.rectangle(p['crop'],outline='red',width=1)
        for j,im in enumerate(sources):
            sheet.paste(im,(256*j,0));draw.text((256*j+2,257),['full reference','pattern only rot90','common prototype','canonical crops'][j],fill='black')
        sheet.save(out/'A_audit/previews'/f"c{ref['id']:02d}.jpg",quality=94)
    counts={d:sum(r['difficulty']==d for r in refs) for d in ('Easy','Medium','Hard')}
    config['sketches']=[sketch]
    write(out/'cases.json',config);write(out/'protocol.json',protocol(root))
    with open(root/'data/e27_c3_audit.csv',encoding='utf-8') as stream:
        audit=list(csv.DictReader(stream))
    decision={'c3_dataset_complete':all(n==6 for n in counts.values()),
              'easy_n':counts['Easy'],'medium_n':counts['Medium'],'hard_n':counts['Hard'],
              'global_baseline_pass':None,'oracle_panel_gain':None,'automatic_local_gain':None,
              'rotation_pass':None,'scale_pass':None,'identity_pass':None,'structure_safe':None,
              'background_safe':None,'next_route':'run_existing_baselines'}
    write(out/'A_audit/report.json',{'counts':counts,'audited':len(audit),'selected':len(refs),
        'independent_unique_sha256':len(hashes),'source_root':str(dataset),'source_split':'validation',
        'family_counts':{p:sum(r['pattern_type']==p for r in refs) for p in ('stripe','plaid','repeated','floral','geometric','mixed','other')},
        'audit_sha256':file_sha(out/'A_audit/audit.csv'),'manifest_sha256':file_sha(root/'data/e27_c3_cases.json'),
        'source_integrity_pass':True,'no_source_target_id_overlap':True,'proofs':proofs,
        'annotation':config['annotation'],'limits':config['limits']})
    write(out/'decision_summary.json',decision)
    print('[A]',json.dumps(decision),flush=True)


def original_one(pipe,noise,sketch,mask,reference,tokens):
    import torch
    from torchvision.transforms.functional import to_tensor
    from tools import e23_mechanism as e23
    saved,saved_tokens=pipe.prepare_latents,pipe.get_image_embeds
    latent=noise*pipe.scheduler.init_noise_sigma
    pipe.prepare_latents=MethodType(lambda self,*a,**kw:latent.clone(),pipe)
    pipe.get_image_embeds=MethodType(lambda self,**kw:e23.tree(tokens,pipe.device),pipe)
    try:
        image=pipe(prompt='a cloth',null_prompt='',negative_prompt=' worst quality, low quality',
            ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference,
            width=mask.width,height=mask.height,num_inference_steps=50,guidance_scale=7.,sketch_scale=.6,
            ipa_scale=1.,texture_mode='patch_resampled',texture_condition_mode='token',
            texture_preprocess_mode='plain_resize',texture_num_tokens=16,texture_scale=1.,
            spatial_mask=to_tensor(mask)[None].to(pipe.device,torch.float16),
            generator=torch.Generator(device=pipe.device).manual_seed(42))[0]
    finally:
        pipe.prepare_latents,pipe.get_image_embeds=saved,saved_tokens
    from tools.e25_spatial_diagnosis import sha
    return image,sha(noise.cpu().numpy().tobytes()),sha(latent.cpu().numpy().tobytes())


def generate(root,out,stage):
    import torch
    from torchvision.transforms.functional import to_tensor
    from tools import e23_mechanism as e23
    from tools import e25_spatial_diagnosis as e25
    from tools.e22_4_generation import module_hashes,native_pipeline
    from tools.e22_o4_metrics import measure
    from tools.e26_audit import structural_drift
    torch.set_num_threads(4);torch.manual_seed(42)
    cases=json.loads((out/'cases.json').read_text())
    if stage=='C':
        decision=json.loads((out/'decision_summary.json').read_text())
        if decision['next_route']!='test_oracle_panel':raise RuntimeError('B 门槛未允许 oracle panel')
        refs=[r for r in cases['references'] if r['oracle']];arms=('C_oracle_panel',);base=out/'C_oracle'
    else:
        refs=cases['references'];arms=ARMS;base=out/'B_baseline'
    base.mkdir(exist_ok=True)
    pipe,modules,width,height=native_pipeline(root,'E5');pipe.set_progress_bar_config(disable=True)
    assert type(pipe.scheduler).__name__=='DDIMScheduler'
    frozen=module_hashes(modules);start=e25.timestep_for(pipe.scheduler,.15)
    bank=e23.token_bank(pipe,{'references':refs},out,width,height,'E5')
    p=protocol(root);p.update(frozen_model_hashes=frozen,checkpoint_sha256=file_sha(root/'output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt'),
        actual_canvas=[width,height],refinement_timesteps=start,scheduler_config=e23.normalize_scheduler(pipe))
    write(base/'protocol.json',p)
    sk=cases['sketches'][0];mask=Image.open(out/sk['mask']).convert('L');sketch=Image.open(out/sk['path']).convert('RGB')
    assert mask.size==(width,height)
    maskarray=np.asarray(mask)>0
    for ref in refs:
        areas=target_panel_masks(maskarray,ref['panels'])
        original=Image.open(out/ref['variants'][0]['path']).convert('RGB')
        for variant in ref['variants']:
            name=variant['variant'];reference=Image.open(out/variant['path']).convert('RGB')
            prefix=f"c{ref['id']:02d}_{name}"
            expected=Image.open(out/'expected'/f'{prefix}.png').convert('RGB')
            wrong=Image.open(out/'expected'/f"c{ref['id']:02d}_{'rot90' if name=='original' else 'original'}.png").convert('RGB')
            for arm in arms:
                folder=base/arm;folder.mkdir(exist_ok=True)
                scaffold=None;z0=None
                if arm!='B2_original_E5':
                    if stage=='C':
                        scaffold_dir=base/'scaffolds';scaffold_dir.mkdir(exist_ok=True)
                        source_mask=np.load(out/'B_baseline/scaffolds'/f'{prefix}_fields.npz')['source_mask']
                        scaffold,_,panel_info=oracle_scaffold(original,maskarray,ref['panels'],name,rectified=True,
                            field_path=scaffold_dir/f'{prefix}_fields.npz',source_mask=source_mask)
                        scaffold.save(scaffold_dir/f'{prefix}.png');write(scaffold_dir/f'{prefix}.json',{'panels':panel_info})
                    else:scaffold=Image.open(base/'scaffolds'/f'{prefix}_{arm}.png').convert('RGB')
                    pixels=to_tensor(scaffold)[None].to(pipe.device,pipe.vae.dtype)*2-1
                    z0=pipe.vae.encode(pixels).latent_dist.mean*pipe.vae.config.scaling_factor
                for seed in SEEDS:
                    path=folder/f"c{ref['id']:02d}_s{seed}_{name}.png";metadata=path.with_suffix('.json')
                    if path.exists() and metadata.exists():continue
                    noise=torch.randn((1,4,height//8,width//8),generator=torch.Generator(device=pipe.device).manual_seed(seed),device=pipe.device,dtype=pipe.vae.dtype)
                    if arm=='B2_original_E5':image,nsha,lsha=original_one(pipe,noise,sketch,mask,reference,bank[ref['id'],name])
                    else:image,nsha,lsha=e25.refine_one(pipe,z0,noise,start,sketch,mask,reference,bank[ref['id'],name],True,mask.size)
                    image.save(path)
                    m=measure(image,mask,sketch,reference,0.,sk['roi'])
                    local=local_readout(image,expected,areas,ref['panels'])
                    wrongmetrics=local_readout(image,wrong,areas,ref['panels'])
                    local['rotation_identity_margin']=local['identity']-wrongmetrics['identity']
                    ownership=ownership_readout(image,expected,areas,ref['panels'])
                    write(metadata,{'case':ref['id'],'reference_id':ref['reference_id'],'seed':seed,'variant':name,'arm':arm,
                        'difficulty':ref['difficulty'],'family':ref['pattern_type'],'path':str(path.relative_to(out)),
                        'noise_sha256':nsha,'initial_latent_sha256':lsha,'output_sha256':file_sha(path),
                        'reference_sha256':file_sha(out/variant['path']),'mask_sha256':file_sha(out/sk['mask']),
                        'expected_sha256':file_sha(out/'expected'/f'{prefix}.png'),
                        'contour_f1':m['contour_f1'],'sketch_iou':m['sketch_iou'],'leakage':m['leakage'],
                        **local,**ownership,**structural_drift(image,expected,mask)})
                    print('[E27]',stage,arm,ref['reference_id'],seed,name,flush=True)
    after=module_hashes(modules);assert after==frozen,'模型被修改'
    write(base/'frozen_check.json',{'pass':after==frozen,'before':frozen,'after':after})
    report(out,stage)


def ownership_readout(image,expected,areas,panels):
    """仅在参考衣片可区分时评估归属；共同 prototype 的 phase tolerant descriptor。"""
    exemplars=[]
    for area in areas:
        y,x=np.where(area)
        if not len(x):exemplars.append(None);continue
        # 取最大距离变换中心，裁相同 32px native 块；不使用方法自己的 scaffold。
        distance=cv2.distanceTransform(area.astype(np.uint8),cv2.DIST_L2,5)
        cy,cx=np.unravel_index(distance.argmax(),distance.shape)
        half=min(16,max(4,int(distance[cy,cx])-1))
        exemplars.append(expected.crop((cx-half,cy-half,cx+half,cy+half)))
    rows=[]
    for i,(area,proto) in enumerate(zip(areas,exemplars)):
        if proto is None:continue
        options=[j for j,p in enumerate(exemplars) if p is not None and j!=i and similarity(proto,p)[0]<.8]
        if not options:continue
        distance=cv2.distanceTransform(area.astype(np.uint8),cv2.DIST_L2,5)
        cy,cx=np.unravel_index(distance.argmax(),distance.shape);half=min(16,max(4,int(distance[cy,cx])-1))
        crop=image.crop((cx-half,cy-half,cx+half,cy+half))
        margin=similarity(crop,proto)[0]-max(similarity(crop,exemplars[j])[0] for j in options)
        rows.append({'panel':panels[i]['name'],'margin':margin,'pass':margin>0})
    return {'ownership_margin':float(np.mean([r['margin'] for r in rows])) if rows else None,
            'ownership_accuracy':float(np.mean([r['pass'] for r in rows])) if rows else None,
            'ownership_panel_n':len(rows),'ownership_rows':rows}


def stats(rows):
    from tools.e20_utilization import case_stat
    result={}
    for key in ('theta_error','period_error','identity','self_similarity','rotation_identity_margin',
                'ownership_margin','ownership_accuracy','output_geometry_coverage','contour_f1','sketch_iou',
                'leakage','boundary_rgb_deviation','boundary_occupancy','boundary_occupancy_error','contour_displacement_px','background_rgb_deviation'):
        values=[(r['case'],r[key]) for r in rows if r.get(key) is not None]
        result[key]=case_stat(values) if values else None
    pairs=[];rotation_pairs=[];rotation_eligible_pairs=0
    for case in sorted(set(r['case'] for r in rows)):
        for seed in SEEDS:
            pair=[r for r in rows if r['case']==case and r['seed']==seed]
            assert len(pair)==2
            valid_geometry=[r for r in pair if r['local_geometry_follow'] is not None]
            follow=all(r['local_geometry_follow'] for r in valid_geometry) and all(r['identity']>=THRESHOLDS['identity_min'] for r in pair)
            # 非定向花卉可报 identity，而不声称旋转几何成功。
            pairs.append((case,float(follow)))
            original=next(r for r in pair if r['variant']=='original')
            rotated=next(r for r in pair if r['variant']=='rot90')
            other={(p['panel'],tuple(p['roi'])):p for p in rotated['patches']}
            localpairs=[]
            for pa in original['patches']:
                pb=other.get((pa['panel'],tuple(pa['roi'])))
                if pb is None or not(pa['expected_valid'] and pb['expected_valid']):continue
                distance=lambda a,b:abs((a-b+90)%180-90)
                expected_delta=distance(pa['theta_expected'],pb['theta_expected'])
                if expected_delta<45:continue
                actual_delta=distance(pa['theta_output'],pb['theta_output'])
                passed=pa['output_valid'] and pb['output_valid'] and max(pa['theta_error'],pb['theta_error'])<=20 and abs(actual_delta-expected_delta)<=20
                localpairs.append(float(passed))
            if localpairs:
                rotation_pairs.append((case,float(np.mean(localpairs)>=.75)))
                rotation_eligible_pairs+=1
    result['follow']=case_stat(pairs)
    result['rotation_follow']=case_stat(rotation_pairs) if rotation_pairs else None
    result['rotation_pair_coverage']=rotation_eligible_pairs/max(len(pairs),1)
    result['follow_definition']='同一 case 的 original/rot90 各自局部方向符合共同 prototype 且 identity>=.65；非方向 case 只评 identity。rotation_follow 单独评成对局部响应。'
    result['panels']={}
    for panel in sorted({p['panel'] for r in rows for p in r['patches']}):
        result['panels'][panel]={}
        for key in ('identity','self_similarity','theta_error','period_error'):
            values=[(r['case'],p[key]) for r in rows for p in r['patches'] if p['panel']==panel and (key not in ('theta_error','period_error') or p['expected_valid'])]
            result['panels'][panel][key]=case_stat(values) if values else None
    result.update(cases=len(set(r['case'] for r in rows)),images=len(rows),geometry_images=sum(r['theta_error'] is not None for r in rows))
    return result


def failure_tags(row,proof):
    tags=[]
    if row['theta_error'] is not None and row['theta_error']>20:tags+=['F2_local_orientation']
    if row['period_error'] is not None and row['period_error']>.5:tags+=['F3_local_scale']
    if row['identity']<.65:tags+=['F4_identity']
    if row['ownership_margin'] is not None and row['ownership_margin']<=0:tags+=['F5_ownership']
    if row['contour_f1']<.9:tags+=['F10_structure']
    if row['leakage']>.02:tags+=['F11_background']
    if row['arm']=='B1_global_rectified' and proof['confidence_coverage']<.5:tags+=['F12_fallback']
    return tags


def report(out,stage):
    from tools.e20_utilization import case_stat
    config=json.loads((out/'cases.json').read_text())
    proofs={(r['case'],r['variant']):r for r in json.loads((out/'A_audit/report.json').read_text())['proofs']}
    b={arm:[json.loads(p.read_text()) for p in sorted((out/'B_baseline'/arm).glob('c*.json'))] for arm in ARMS}
    if stage=='B':
        byarm={}
        failures=[]
        for arm,rows in b.items():
            assert len(rows)==72,(arm,len(rows))
            byarm[arm]={'all':stats(rows),'difficulty':{d:stats([r for r in rows if r['difficulty']==d]) for d in ('Easy','Medium','Hard')},
                'family':{d:stats([r for r in rows if r['family']==d]) for d in sorted(set(r['family'] for r in rows))}}
            for row in rows:failures.append({'case':row['case'],'seed':row['seed'],'variant':row['variant'],'arm':arm,'difficulty':row['difficulty'],'tags':failure_tags(row,proofs[row['case'],row['variant']])})
        rect=b[ARMS[1]]
        def localfailed(r):return r['theta_error'] is not None and r['theta_error']>20 or r['identity']<.65 or r['ownership_margin'] is not None and r['ownership_margin']<=0
        easy=case_stat([(r['case'],float(localfailed(r))) for r in rect if r['difficulty']=='Easy'])
        mh=case_stat([(r['case'],float(localfailed(r))) for r in rect if r['difficulty']!='Easy'])
        needs_oracle=mh['mean']>=.25 and mh['mean']-easy['mean']>=.10
        # 如果所有档已经失败，仍可测 oracle 诊断，但不宣称 M/H 分层证据成立。
        broad_failure=mh['mean']>=.50
        decision=json.loads((out/'decision_summary.json').read_text())
        decision.update(global_baseline_pass=not(needs_oracle or broad_failure),
            next_route='test_oracle_panel' if needs_oracle or broad_failure else 'keep_global_mapping',
            baseline_difficulty_gap=mh['mean']-easy['mean'],baseline_gate_scope='M/H gap or broad C3 local failure')
        write(out/'B_baseline/report.json',{'methods':byarm,'local_failure_easy':easy,'local_failure_medium_hard':mh,
            'difficulty_specific_failure':needs_oracle,'broad_c3_failure':broad_failure,'test_oracle':needs_oracle or broad_failure,
            'failure_counts':{arm:{f:sum(f in r['tags'] for r in failures if r['arm']==arm) for f in ('F1_global_orientation','F2_local_orientation','F3_local_scale','F4_identity','F5_ownership','F6_seam','F7_fold','F8_perspective','F9_occlusion','F10_structure','F11_background','F12_fallback')} for arm in ARMS},
            'taxonomy_limits':'F1/F6/F7/F8/F9 必须结合目视归因；未自动赋因果标签，零计数不代表没有失败。',
            'noise_hash_pass':noise_check(b),'frozen_pass':json.loads((out/'B_baseline/frozen_check.json').read_text())['pass']})
        write(out/'B_baseline/failures.json',failures)
    else:
        rows=[json.loads(p.read_text()) for p in sorted((out/'C_oracle/C_oracle_panel').glob('c*.json'))]
        assert len(rows)==32
        ids=set(r['case'] for r in rows);baseline=[r for r in b[ARMS[1]] if r['case'] in ids]
        a,z=stats(rows),stats(baseline)
        index={(r['case'],r['seed'],r['variant']):r for r in baseline}
        contrasts={key:case_stat([(r['case'],r[key]-index[r['case'],r['seed'],r['variant']][key]) for r in rows if r.get(key) is not None and index[r['case'],r['seed'],r['variant']].get(key) is not None]) for key in ('contour_f1','leakage','identity','background_rgb_deviation')}
        follow_gain=a['follow']['mean']-z['follow']['mean']
        theta_reduction=1-a['theta_error']['mean']/max(z['theta_error']['mean'],1e-9) if a['theta_error'] and z['theta_error'] else 0.
        structure=contrasts['contour_f1']['mean']>=-.02
        background=contrasts['leakage']['mean']<=.01
        gain=(follow_gain>=.10 or theta_reduction>=.20) and structure and background
        pair_follow=[]
        for cid in ids:
            sa=stats([r for r in rows if r['case']==cid]);sz=stats([r for r in baseline if r['case']==cid])
            pair_follow.append((cid,sa['follow']['mean']-sz['follow']['mean']))
        write(out/'C_oracle/report.json',{'oracle':a,'matched_B1':z,'matched_reference_ids':sorted(ids),'follow_gain':follow_gain,
            'theta_error_reduction_fraction':theta_reduction,'paired_follow_gain':case_stat(pair_follow),'contrasts':contrasts,
            'structure_safe':structure,'background_safe':background,'oracle_panel_gain':gain,
            'frozen_pass':json.loads((out/'C_oracle/frozen_check.json').read_text())['pass'],
            'gate':'follow >=10 percentage points OR theta error reduction >=20%; delta F1>=-.02 AND delta leakage<=.01; reference clustered CI reported.',
            'noise_hash_pass':noise_check({'oracle':rows,'matched_B1':baseline})})
        decision=json.loads((out/'decision_summary.json').read_text())
        decision.update(oracle_panel_gain=bool(gain),structure_safe=bool(structure),background_safe=bool(background),
            next_route='panel_local_correspondence' if gain else 'nonrigid_dense_correspondence',
            stopping_reason=None if gain else 'oracle 未同时满足收益与结构/背景安全，按 E27-C 停止 panel 路线；D/E/F 不运行。')
    write(out/'decision_summary.json',decision);previews(out,stage,config)
    print('[decision]',json.dumps(decision,ensure_ascii=False),flush=True)


def noise_check(methods):
    hashes={}
    for rows in methods.values():
        for row in rows:
            key=(row['case'],row['seed'],row['variant'])
            if key in hashes:assert hashes[key]==row['noise_sha256']
            hashes[key]=row['noise_sha256']
    return True


def previews(out,stage,cases):
    base=out/('B_baseline' if stage=='B' else 'C_oracle');folder=base/'previews';folder.mkdir(exist_ok=True)
    refs=cases['references'] if stage=='B' else [r for r in cases['references'] if r['oracle']]
    for ref in refs:
        cols=5 if stage=='B' else 4;sheet=Image.new('RGB',(cols*192,2*284),'white');draw=ImageDraw.Draw(sheet)
        for y,name in enumerate(('original','rot90')):
            paths=[out/'A_audit/inputs'/f"c{ref['id']:02d}_{name}.png",out/'expected'/f"c{ref['id']:02d}_{name}.png"]
            labels=['reference '+name,'common prototype']
            if stage=='B':
                paths += [out/'B_baseline'/arm/f"c{ref['id']:02d}_s42_{name}.png" for arm in ARMS];labels+=list(ARMS)
            else:
                paths += [out/'B_baseline/B1_global_rectified'/f"c{ref['id']:02d}_s42_{name}.png",out/'C_oracle/C_oracle_panel'/f"c{ref['id']:02d}_s42_{name}.png"]
                labels+=['B1 global rect','C oracle panel']
            for x,(path,label) in enumerate(zip(paths,labels)):
                im=Image.open(path).convert('RGB');im.thumbnail((192,256));sheet.paste(im,(x*192,y*284));draw.text((x*192+1,y*284+258),label,fill='black')
        sheet.save(folder/f"c{ref['id']:02d}.jpg",quality=94)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',type=Path);p.add_argument('--stage',choices=('A','B','C','report_B','report_C'),required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=='A':prepare(args.root,args.out,args.dataset)
    elif args.stage.startswith('report_'):report(args.out,args.stage[-1])
    else:
        import torch
        with torch.inference_mode():generate(args.root,args.out,args.stage)


if __name__=='__main__':main()
