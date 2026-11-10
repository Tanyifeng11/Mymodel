"""A0-A5六臂统一评测；原逐图定义不变，扩展比较族并单列描述性机制对照。"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tools.e2_fft_experiment import setup, read, write, sha
from models.e2_structure_ablation import ARMS, NEW_ARMS
from tools.e2_a_evaluation import boundary_f1, local_response, bundle


def image_once(args, core, pipe, network, pair, condition):
    key = 'original_reference' if condition == 'old' else 'swapped_reference'
    sketch = core.tensor(Path(args.data) / pair['sketch']).to(args.device).half()
    reference = core.tensor(Path(args.data) / pair[key]).to(args.device).half()
    network.adapter.condition(sketch, reference)
    latent = torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
        generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
    latent_sha = core.digest(latent)
    image = pipe(ref_image=sketch,prompt=pair['caption'],texture_clip_image=None,texture_embeds=None,
        null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,num_images_per_prompt=1,
        guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,latents=latent,
        generator=torch.Generator(device=args.device).manual_seed(pair['seed']),num_inference_steps=50)[0]
    network.adapter.clear()
    return image, latent_sha


def training_audit(out):
    from tools.e2_a_ablation import MATCH_FIELDS
    for name in ['baseline_replay','A5_replay','training_inputs_final','mask_inputs_final']:
        assert read(out / 'audit' / (name+'.json'))['passed']
    assert read(out / 'smoke/result.json')['passed'] and read(out / 'regression/A5_reuse.json')['passed']
    checkpoints = read(out / 'audit/reuse_A0_A5.json')['checkpoints']
    baseline = None; cfg0 = read(out / 'training/A0/complete.json')['config']; records = {}
    for arm in ARMS:
        complete = read(out / 'training' / arm / 'complete.json')
        path = Path(checkpoints[arm]) if arm in checkpoints else out / 'training' / arm / 'latest.pt'
        assert complete['status'] == 'completed' and complete['updates'] == 2000 and sha(path) == complete['checkpoint_sha256']
        assert all(complete['config'][k] == cfg0[k] for k in MATCH_FIELDS)
        logs = [json.loads(s) for s in (out / 'training' / arm / 'training.jsonl').read_text().splitlines()]
        assert [x['step'] for x in logs] == list(range(1,2001))
        assert all(np.isfinite(x['grad_norm']) and all(np.isfinite(v) for v in x['losses'].values()) for x in logs)
        tokens = [x['tokens'] for x in logs]
        if baseline is None: baseline = tokens
        else: assert tokens == baseline
        if arm != 'A0':
            assert all(len(x['structure']) == 8 for x in logs)
            assert all(np.isfinite(m['residual_rms']).all() for x in logs for m in x['structure'])
        records[arm] = dict(updates=2000, checkpoint_sha256=complete['checkpoint_sha256'],
            reused=arm in checkpoints, training_seconds=sum(x['seconds'] for x in logs),
            peak_gib=max(x['peak_gib'] for x in logs))
    write(out / 'audit/training_completion.json',dict(passed=True,same_legacy_stream=True,matched_budget=True,arms=records))


def generate(args):
    from tools import e2_b_screen_evaluation as legacy
    from tools.e2_a_ablation import load_arm, a_environment
    out = Path(args.out)
    assert args.arm in NEW_ARMS, 'A0/A5只复用，不重复生成完整输出'
    assert read(out / 'audit/training_inputs_final.json')['passed'] and read(out / 'audit/mask_inputs_final.json')['passed']
    with patch.object(legacy,'load',load_arm): legacy.generate(args)
    path = out / 'generation' / (args.arm+'.json'); record = read(path)
    record['environment'] = a_environment(); write(path,record)


def structure_comparisons(structure, legacy, pairs, arms=ARMS):
    from tools.bf_t3_three_arm_report import paired
    from tools.bf_t3_ablation_report import holm
    a = {(x['arm'],x['case_id'],x['condition']):x for x in structure}
    b = {(x['arm'],x['case_id'],x['condition']):x for x in legacy}
    metrics = [('old_boundary_f1_r2',a,'boundary_f1_r2'),('old_edge_f1',b,'edge_f1'),
               ('old_iou',b,'iou'),('old_inner_support_proxy',a,'inner_support_proxy')]
    comparisons = []
    for arm in arms[1:]:
        for metric,table,key in metrics:
            delta = []
            for p in pairs:
                x,y = [table[k,p['case_id'],'old'][key] for k in [arms[0],arm]]
                if x is not None and y is not None: delta.append(y-x)
            comparisons.append(dict(arm=arm,comparison=arm+'_minus_'+arms[0],metric=metric,
                family='primary' if metric=='old_boundary_f1_r2' else 'secondary',**paired(delta)))
    for family in ['primary','secondary']: holm([x for x in comparisons if x['family']==family])
    descriptive = []
    for arm,reference in [('A2','A1'),('A5','A3'),('A5','A4')]:
        for metric,table,key in metrics:
            delta=[]
            for p in pairs:
                x,y=[table[k,p['case_id'],'old'][key] for k in [reference,arm]]
                if x is not None and y is not None: delta.append(y-x)
            stats=paired(delta); stats.pop('p')
            descriptive.append(dict(comparison=arm+'_minus_'+reference,metric=metric,
                inference='descriptive unadjusted CI;no significance claim',**stats))
    return comparisons, descriptive


def reused_metric_equivalence(out, parent):
    """同环境重算A0/A5必须保留原逐图值；失败先落盘证据。"""
    errors = {}
    def diff(a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            assert set(a)==set(b)
            return max((diff(a[k],b[k]) for k in a),default=0.)
        if isinstance(a,list) and isinstance(b,list):
            assert len(a)==len(b);return max((diff(x,y) for x,y in zip(a,b)),default=0.)
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)
        assert a==b;return 0.
    for name in ['per_image.json','structure_per_image.json']:
        old={(x['arm'],x['case_id'],x['condition']):x for x in read(parent/'metrics'/name)}
        new={(x['arm'],x['case_id'],x['condition']):x for x in read(out/'metrics'/name) if x['arm'] in ['A0','A5']}
        assert set(old)==set(new);errors[name]=max(diff(old[k],new[k]) for k in old)
    write(out/'audit/reused_metric_equivalence.json',dict(passed=max(errors.values())==0,max_abs=errors))
    assert max(errors.values())==0,errors


def evaluate(args):
    from tools import e2_b_screen_evaluation as legacy
    from tools.e2_a_ablation import a_environment
    from models.e2_structure import structure_maps
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask
    core,out=setup(args)
    with patch.object(legacy,'ARMS',list(ARMS)),patch.object(legacy,'training_audit',training_audit): legacy.evaluate(args)
    texture=read(out/'metrics/paired_statistics.json')
    for x in texture:x['family']='texture_gray_strict_guard' if x['family']=='primary' else 'texture_other_guard'
    write(out/'metrics/texture_paired_statistics.json',texture)
    pairs=read(out/'strict_pairs.json')['pairs'];rows=[]
    for pair in pairs:
        sketch=Path(args.data)/pair['sketch']
        mask=prepare_evaluation_masks((384,512),sketch_path=str(sketch),mask_policy='sketch_only')['garment']
        assert mask.any()
        maps=structure_maps(core.tensor(sketch));candidate=maps[0,4].cpu().numpy()>.5
        gt=Image.open(Path(args.data)/pair['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        target=local_response(gt);eligible=candidate & (target>=.05)
        for arm in ARMS:
            for condition in ['old','new']:
                image=Image.open(out/'outputs'/arm/pair['case_id']/(condition+'.png')).convert('RGB')
                predicted=estimate_foreground_mask(image,(384,512));response=local_response(image)
                inner=float(np.minimum(response[eligible]/target[eligible],1.).mean()) if eligible.any() else None
                row=dict(arm=arm,condition=condition,case_id=pair['case_id'],index=pair['index'],identity_group=pair['identity_group'],
                    predicted_mask_area=float(predicted.mean()),inner_candidate_pixels=int(candidate.sum()),
                    inner_eligible_pixels=int(eligible.sum()),inner_support_proxy=inner)
                for radius in [1,2,4]:row['boundary_f1_r%d'%radius]=boundary_f1(predicted,mask,radius)
                rows.append(row)
    write(out/'metrics/structure_per_image.json',rows)
    comparisons,descriptive=structure_comparisons(rows,read(out/'metrics/per_image.json'),pairs)
    write(out/'metrics/paired_statistics.json',comparisons)
    write(out/'metrics/mechanistic_descriptive.json',descriptive)
    reused_metric_equivalence(out,Path(args.a_parent))
    version=read(out/'metrics/version.json')
    original_version=read(Path(args.a_parent)/'metrics/version.json')
    version.update(protocol=read(out/'protocol.json'),primary_comparisons=5,secondary_comparisons=15,
        texture_comparisons=20,environment=a_environment(),boundary=original_version['boundary'],inner=original_version['inner'],
        descriptive_mechanistic_comparisons=12,old_A0_A5_report_preserved=True)
    write(out/'metrics/version.json',version)
    print('A_ABLATION_EVALUATION_COMPLETED',flush=True)
