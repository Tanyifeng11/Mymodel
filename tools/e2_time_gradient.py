"""已批准的完整扩散时间与训练损失竞争诊断；生成器更新固定为0。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from models.e2_geometry import InteriorPatchSampler, build_geometry, extract_patches
from models.e2_local_fft import gray
from tools.e2_fft_experiment import SOURCE, setup, environment, read, write, sha
from tools.e2_texture_bottleneck import BASELINE_SHA, PAIRS_SHA
from tools.e2_time_gradient_common import TRAIN_TIMES, NOISE_SEEDS, SCORE_STEPS


def initialize(args, gradient=False):
    value=copy.copy(args)
    if gradient:value.action='calibrate'
    core,out=setup(value);args.data=str(out/'inputs/frozen')
    return core,out


def code_environment():
    result=environment()
    for name in ['tools/e2_time_gradient.py','tools/e2_time_gradient_common.py','tools/e2_time_gradient_trace.py',
        'tools/e2_time_gradient_gradients.py','tools/e2_time_gradient_analysis.py','tests/test_e2_time_gradient.py',
        'scripts/e2_time_gradient.sbatch','pipelines/IMAGGarment_pipeline.py','inference_IMAGGarment-1.py']:
        result['source'][name]=sha(SOURCE/name)
    result.update(generator_updates=0,new_encoder=False,A=False,B=False,C=False)
    return result


def training_selection(args, core, out):
    from utils.texture_preprocess import preprocess_texture_image
    parent=Path(args.bparent);snapshot=read(parent/'audit/training_inputs.json')
    manifest=core.ROOT/'training_manifest.json';assert sha(manifest)==snapshot['source_manifest_sha256']
    rows=read(manifest);indices=read(parent/'split.json')['train_indices']
    source=core.PROJECT.parent/'datasets/BF/training';mask_snapshot=read(Path(args.aparent)/'audit/mask_inputs.json')
    def label(caption):
        text=str(caption).lower()
        if 'stripe' in text:return 'stripes'
        if any(x in text for x in ['plaid','checkered','gingham']):return 'checks'
        if 'polka' in text or 'dotted' in text:return 'dots'
        if not any(x in text for x in ['floral','print','pattern']):return 'low_texture'
        return None
    chosen=[];identities=set();skipped={}
    for category in ['stripes','checks','dots','low_texture']:
        candidates=sorted([i for i in indices if label(rows[i]['caption'])==category],
            key=lambda i:hashlib.sha256(('e2_time_gradient_1111/'+rows[i]['cloth']).encode()).hexdigest())
        accepted=0
        for index in candidates:
            row=rows[index];identity=Path(row['cloth']).stem.split('_')[0]
            if identity in identities or row.get('mask'):continue
            mask_name='%03d/%05d_mask.png'%(index//1000,index);mask_file=core.MASKS/mask_name
            assert sha(mask_file)==mask_snapshot[mask_name]['sha256']
            mask=torch.from_numpy((np.asarray(Image.open(mask_file).convert('L').resize((384,512),Image.Resampling.NEAREST))>127.5).copy())[None,None].float()
            geometry=build_geometry(mask,details=False)
            if not bool(geometry.valid[0]):skipped['invalid_mask']=skipped.get('invalid_mask',0)+1;continue
            ref=preprocess_texture_image(Image.open(source/row['texture']).convert('RGB'),384,512,'plain_resize')
            sampler=InteriorPatchSampler(1111000+index);coordinates={};valid=0
            for size in [32,64]:
                body,n=sampler.coordinates(geometry.body[0],size,4);reference,_=sampler.coordinates(torch.ones(512,384),size,4)
                std=gray(extract_patches((ref+1)*.5,reference,size)).std((-2,-1),unbiased=False)
                current=int((std>=.02).sum());valid+=current
                coordinates[str(size)]=dict(body=body,reference=reference,body_candidates=n,reference_valid_n=current)
            if any(len(c['body'])<4 for c in coordinates.values()):skipped['few_body_patches']=skipped.get('few_body_patches',0)+1;continue
            if (category=='low_texture') != (valid==0):continue
            for key in ['cloth','sketch','texture']:
                src=source/row[key];assert sha(src)==snapshot['files'][row[key]]['sha256']
                dst=out/'inputs/train'/row[key];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            dst=out/'inputs/train_masks'/mask_name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(mask_file,dst)
            chosen.append(dict(index=index,identity=identity,pattern_candidate=category,
                label_scope='caption candidate plus measured input eligibility;not semantic ground truth',row=row,
                hashes={k:snapshot['files'][row[k]]['sha256'] for k in ['cloth','sketch','texture']},
                mask_file=mask_name,mask_sha256=sha(mask_file),coordinates=coordinates))
            identities.add(identity);accepted+=1
            if accepted==8:break
        assert accepted==8,('原阈值下不足8个诊断输入，不放宽阈值',category,accepted)
    dst=out/'inputs/training_manifest.json';dst.parent.mkdir(exist_ok=True);shutil.copy2(manifest,dst)
    write(out/'inputs/training_probes.json',dict(records=chosen,skipped=skipped,identities=32,
        source_manifest_sha256=sha(manifest),sampling='input-only deterministic24textured+8lowtexture training identities;not population-unbiased'))
    return chosen


def prepare(args):
    import matplotlib
    core,out=initialize(args);parent=Path(args.parent);bp=Path(args.bparent)
    assert read(parent/'metrics/version.json')['generator_updates']==0
    assert sha(parent/'strict_pairs.json')==PAIRS_SHA
    old=read(bp/'training/B0/complete.json');assert sha(bp/'training/B0/latest.pt')==BASELINE_SHA==old['checkpoint_sha256']
    for name in ['models/bf_original_fusion.py','models/bf_sd_skip.py','models/e2_compat.py','tools/bf_t3_frozen_core.py','train_GAM_texture_joint.py']:
        assert sha(SOURCE/name)==old['environment']['source'][name]
    for directory in ['inputs/frozen','inputs/controls','outputs/baseline']:
        shutil.copytree(parent/directory,out/directory,dirs_exist_ok=True)
    for name in ['strict_pairs.json','inputs/controls.json']:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
    old_spec=read(parent/'protocol.json');pairs=read(out/'strict_pairs.json')['pairs']
    for pair in pairs:
        for key in ['sketch','gt','original_reference','swapped_reference']:
            assert sha(out/'inputs/frozen'/pair[key])==pair['hashes'][key]
    for control in read(out/'inputs/controls.json'):assert sha(out/control['file'])==control['sha256']
    probes=training_selection(args,core,out)
    from PIL import ImageDraw
    for page in range(4):
        canvas=Image.new('RGB',(512,8*184),'white');draw=ImageDraw.Draw(canvas)
        for row,p in enumerate(probes[page*8:page*8+8]):
            y=row*184;draw.text((3,y+2),'%d %s | caption candidate'%(p['index'],p['pattern_candidate']),fill='black')
            sources=[out/'inputs/train'/p['row'][key] for key in ['sketch','cloth','texture']]+[out/'inputs/train_masks'/p['mask_file']]
            for column,path in enumerate(sources):
                canvas.paste(Image.open(path).convert('RGB').resize((128,160),Image.Resampling.BILINEAR),(column*128,y+20))
        canvas.save(out/'inputs'/('training_review_%d.png'%page))
    mature=read(SOURCE/'configs/e2_fft_regression.json');mature_path=Path(args.project)/mature['checkpoint']
    assert sha(mature_path)==mature['checkpoint_sha256']
    weights=read(bp/'calibration/result.json')['weights']
    write(out/'protocol.json',dict(version='E2_time_gradient_v1_no_generator_updates',environment=code_environment(),
        parent=str(parent),pairs_sha256=PAIRS_SHA,folds=old_spec['folds'],short_cases=old_spec['short_cases'],
        checkpoints={'mature':dict(path=str(mature_path),sha256=mature['checkpoint_sha256']),
                     'B0':dict(path=str(bp/'training/B0/latest.pt'),sha256=BASELINE_SHA)},
        training_times=list(TRAIN_TIMES),noise_seeds=list(NOISE_SEEDS),gradient_points=32*10*2*2,
        gradient_case_note='24inputqualified texture candidates+8zero-valid-reference-patch controls;two frozen weights;not online training history',
        condition_policy='primary text/reference kept with original pairedGT;dropout frequency from actual history,not simulated as population',
        auxiliary_weights=weights,auxiliary_real_gate='t<=249 and valid patches and reference kept',
        auxiliary_counterfactual='same descriptor with t gate opened only for diagnosis;never counted as actual active supervision',
        train_preprocess='unchanged JointTextureDataset plain_resize BICUBIC texture;old GTGram unchanged',
        matched_gram_control='same ReferenceGramDescriptor ImageNet preprocess and predictedbody coordinates,targetGT body patches;diagnostic only',
        trace_steps=list(range(50)),real_dists_score_steps=list(SCORE_STEPS),controlled_phases='four canonical gratings+one saved shift',
        readout='fourfold identity CV;ridge1;two covered phases in train folds;per-step same summary within each representation',
        v3='original complete crosspatchDISTS at11fixed points on own trajectories;exploratory curve,not replacement of old metrics',
        region='legacy sketch_only mask/body erosion9 for evaluation;old training mask for gradients',
        inference='original384x512 DDIM50 CFG7 GAM.6;eta0;original pipeline replay pixel0',
        directional_check='epsilon-space +/-small normalized gradient perturbation;no parameter update,no AdamW causal claim',
        gradient_targets='epsilon,decoded,four injected features,allfour projection weights,listed texture/FiLM/sketch/GAM representative parameters;not wholeoptimizer',
        statistics='within identity average across seeds/t in fixed low/mid/high bands,then descriptivebootstrap;zero norms cosine=null;no new confirmatory p',
        decision='compare true time signal versus scheduleattenuation;weightedGTGram interference versus matchedGTlocalGram and otheroldlosses;choose one falsifiable B candidate after data',
        generator_updates=0,new_encoder=False))
    write(out/'audit/preparation.json',dict(passed=True,train_identities=len(probes),evaluation_identities=len(pairs),
        input_controls=704,training_manifest_sha256=sha(out/'inputs/training_manifest.json'),environment=code_environment()))
    # 历史激活率单独读取真实日志，不把诊断的分层采样比例冒称训练频率。
    history={}
    for arm in ['B0','B1','B2','B4','B5']:
        rows=[json.loads(line) for line in (bp/'training'/arm/'training.jsonl').read_text().splitlines()]
        tokens=[t for r in rows for t in r['tokens']]
        history[arm]=dict(updates=len(rows),samples=len(tokens),low_t=sum(t['t']<=249 for t in tokens),
            reference_kept=sum(not(.2<=t['drop']<.3) for t in tokens),
            low_t_reference_kept=sum(t['t']<=249 and not(.2<=t['drop']<.3) for t in tokens),
            valid_auxiliary=sum(r['patch_counts']['valid_n'] for r in rows),log_sha256=sha(bp/'training'/arm/'training.jsonl'))
    write(out/'audit/historical_activation.json',history)
    print('TIME_GRADIENT_PREPARE_PASS',len(probes),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','pilot-trace','pilot-gradient','trace','gradient','aggregate','bundle','dispatch'])
    p.add_argument('--project',default='/share/home/u2515283058/Mymodel')
    p.add_argument('--parent',default='/share/home/u2515283058/Mymodel/output_eval/e2_texture_bottleneck_20261111')
    p.add_argument('--bparent',default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--aparent',default='/share/home/u2515283058/Mymodel/output_eval/e2_a_minimal_20261109')
    p.add_argument('--out',default='/share/home/u2515283058/Mymodel/output_eval/e2_time_gradient_20261111')
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--stage',choices=['trace','gradient'],default='trace')
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action in ['pilot-trace','trace']:
        from tools import e2_time_gradient_trace as module
        getattr(module,'pilot' if args.action=='pilot-trace' else 'trace')(args)
    elif args.action in ['pilot-gradient','gradient']:
        from tools import e2_time_gradient_gradients as module
        getattr(module,'pilot' if args.action=='pilot-gradient' else 'gradient')(args)
    else:
        from tools import e2_time_gradient_analysis as module
        getattr(module,args.action)(args)


if __name__=='__main__':main()
