"""复用旧冻结推理/六指标后端，新的 P2 独立写出；GT 不进入推理。"""
import argparse
import torch
from PIL import Image,ImageDraw
from tools.e2a_sup_common import *
from tools import e2a_attr_eval as evaluator


def evaluate(arm,stage,phase='formal'):
    assert stage in ['P2a','P2b','reference_swap'] and arm in ['B','C','E1']+ARMS
    if stage=='reference_swap':
        return reference_swap(arm)
    if phase=='pilot':
        assert arm in ARMS and stage=='P2a'
        path=folder(arm,'pilot')/'checkpoint-final/joint_model.pt'
        rows=read(ROOT/'splits/P2a.json')[:8]
        output_stage='pilot_P2a'
    else:
        path=model_path(arm);rows=read(ROOT/'splits'/(stage+'.json'));output_stage=stage
    if arm in EXPECTED:assert sha(path)==EXPECTED[arm]
    else:assert sha(path)==read(path.parent.parent/'checkpoint_verification.json')['sha256']
    evaluator.OUT=ROOT
    evaluator.model_path=lambda name:path
    evaluator.input_rows=lambda name:rows
    freeze(ROOT/'checkpoint_manifest.json',read(OLD/'checkpoint_manifest.json'))
    evaluator.generate(output_stage,arm)
    evaluator.measure(output_stage,arm)
    write(ROOT/'evaluation'/output_stage/arm/'supervision_evaluation.json',dict(checkpoint_sha256=sha(path),
        rows_sha256=sha(ROOT/'splits'/(stage+'.json')),phase=phase,
        loss_disabled_at_inference=True,metric_backend_unchanged=True,GT_only_in_metrics=True))


def reference_swap(arm):
    from eval.eval_utils import prepare_evaluation_masks,json_safe
    from eval.metrics import compute_structure_preservation,compute_texture_color_fidelity
    from tools.e2a_sup_reference import approved_reference_manifest
    reference_manifest,rows=approved_reference_manifest()
    path=model_path(arm)
    if arm in EXPECTED:assert sha(path)==EXPECTED[arm]
    else:assert sha(path)==read(path.parent.parent/'checkpoint_verification.json')['sha256']
    variants=[dict(row,reference=row['reference_alternative']) for row in rows]
    evaluator.OUT=ROOT;evaluator.model_path=lambda name:path;evaluator.input_rows=lambda name:variants
    # 使用独立目录，只生成换参考版本；原参考版本复用本次 P2b 的同 seed/latent 输出。
    evaluator.generate('reference_swap',arm)
    base=ROOT/'evaluation/P2b'/arm;out=ROOT/'evaluation/reference_swap'/arm
    records=[]
    for row in rows:
        identity=row['id'];first=base/'images'/(identity+'.png');second=out/'images'/(identity+'.png')
        m0=read(first.with_suffix('.json'));m1=read(second.with_suffix('.json'))
        assert m0['seed']==m1['seed'] and m0['initial_latent_sha256']==m1['initial_latent_sha256']
        assert all(m0['inputs'][k]==m1['inputs'][k] for k in ['gt','sketch'])
        assert m0['inputs']['reference']!=m1['inputs']['reference']
        mask=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
        result=dict(id=identity,arm=arm,initial_latent_sha256=m0['initial_latent_sha256'],
            input_reference_changed_only=True,counterfactual_GT_available=False,
            color_is_proxy_only=True,motif_identity_proven=False)
        for name,image in [('original',first),('swapped',second)]:
            structure=compute_structure_preservation(gen_path=str(image),sketch_path=row['sketch'],mask_bundle=mask)
            result[name+'_structure']={k:structure.get(k) for k in ['edge_f1','sketch_iou','edge_l1','metric_warnings']}
            for refname,key in [('original','reference'),('alternative','reference_alternative')]:
                colors=compute_texture_color_fidelity(gen_path=str(image),texture_path=row[key],mask_bundle=mask)
                result[name+'_to_'+refname+'_color']={k:colors.get(k) for k in ['tcf_lab_delta','tcf_hsv_l1','metric_warnings']}
        records.append(json_safe(result))
    write(out/'per_identity_swap_metrics.json',records)
    for start in range(0,32,8):
        page=Image.new('RGB',(5*128,8*195),'white');draw=ImageDraw.Draw(page)
        for j,row in enumerate(rows[start:start+8]):
            paths=[row['sketch'],row['reference'],row['reference_alternative'],
                base/'images'/(row['id']+'.png'),out/'images'/(row['id']+'.png')]
            labels=['Sketch','Ref original','Ref alternative','Original','Swapped']
            for k,(p,label) in enumerate(zip(paths,labels)):
                page.paste(Image.open(p).convert('RGB').resize((128,171)),(k*128,j*195+24))
                draw.text((k*128+2,j*195+3),row['id']+' '+label,fill='black')
        page.save(out/('swap_grid_%02d.png'%start))
    write(out/'evaluation_complete.json',dict(complete=True,n=32,arm=arm,
        matched_reference_condition_count=2,old_GT_LPIPS_not_computed=True,
        reference_manifest_sha256=sha(reference_manifest),
        orientation_scale_correspondence='REQUIRES_BLIND_REVIEW',human_blind_review='PENDING'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True)
    p.add_argument('--stage',choices=['P2a','P2b','reference_swap'],required=True)
    p.add_argument('--phase',choices=['pilot','formal'],default='formal')
    a=p.parse_args();torch.set_num_threads(2);evaluate(a.arm,a.stage,a.phase)
