"""有界续训与配对对照；有恢复状态时继续AdamW和专用训练随机流。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault('T3_RUN_NAME','t3_local_texture_continue_20261101')
from tools import t3_local_pilot as pilot

ROOT = pilot.ROOT
PARENT = pilot.PROJECT/'output_eval'/os.environ.get('T3_PARENT_NAME','t3_local_texture_pilot_20261101')
START = int(os.environ.get('T3_INITIAL_UPDATE','400'))
END = int(os.environ.get('T3_FINAL_UPDATE','800'))
RESUME_OPTIMIZER = os.environ.get('T3_RESUME_OPTIMIZER','0') == '1'
ARCHIVE = 'parent%d'%START


def prepare():
    assert not (ROOT/'protocol.json').exists(), '不覆盖已有续训协议'
    original = pilot.read(PARENT/'protocol.json')
    assert original['updates'] == START
    checkpoints = {}
    resume_states = {}
    for arm in ['residual', 'reference']:
        complete = pilot.read(PARENT/'training'/arm/'complete.json')
        assert complete.get('total_updates',complete['updates']) == START and complete['frozen_unchanged']
        path = PARENT/'training'/arm/'branch.pt'
        checkpoints[arm] = pilot.sha(path)
        assert checkpoints[arm] == complete['branch_sha256']
        if RESUME_OPTIMIZER:
            resume_states[arm] = pilot.sha(PARENT/'training'/arm/'resume_state.pt')
    for folder in ['inputs', 'masks', 'synthetic']:
        shutil.copytree(PARENT/folder, ROOT/folder)
    shutil.copytree(PARENT/'outputs/T3', ROOT/'outputs/T3')
    for name in ['audit/preflight.json', 'cache/training.pt']:
        target = ROOT/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT/name, target)
    # 上一检查点数值留存，后续可在本地独立重算增量。
    for name in ['protocol.json', 'decision.json', 'metrics/per_image.json', 'metrics/summary.json',
                 'training/residual/complete.json', 'training/reference/complete.json']:
        target = ROOT/ARCHIVE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT/name, target)
    protocol = dict(original, name=ROOT.name, updates=END, initial_update=START,
                    initial_training_root=str(PARENT/'training'), initial_branch_sha256=checkpoints,
                    resume_state_sha256=resume_states,
                    optimizer_resume='恢复AdamW和专用训练随机流' if RESUME_OPTIMIZER else '旧检查点未保存优化器；AdamW重新初始化',
                    save_resume_updates=[1200] if END == 1600 else [],
                    job_prefix=os.environ.get('T3_JOB_PREFIX','t3local_c800'),
                    source_commit=subprocess.check_output(['git','-C',str(pilot.SOURCE),'rev-parse','HEAD'],text=True).strip(),
                    budget='两臂各新增%d步；每臂单GPU，最多30分钟；本轮不自动追加'%(END-START),
                    decision='沿用原400步试验门槛；另报告%d-%d配对差异，不据结果改门槛'%(END,START))
    pilot.write(ROOT/'protocol.json', protocol)
    pilot.write(ROOT/'continuation_protocol.json', dict(
        parent_root=str(PARENT), parent_protocol_sha256=pilot.sha(PARENT/'protocol.json'),
        initial_update=START, added_updates=END-START, final_update=END, optimizer_restarted=not RESUME_OPTIMIZER,
        resume_state_sha256=resume_states,
        evaluation_manifest_sha256=pilot.sha(ROOT/'inputs/evaluation.json'),
        train_manifest_sha256=pilot.sha(ROOT/'inputs/train.json'),
        unchanged='冻结T3、支路、训练数据、损失、学习率、有效batch、推理和评价规则',
        main_focus=['IoU','EdgeF1','background_MAE','grayMargin','orientation','period'],
        limitation='已查看的同一开发集；保留合成包袋与garment提示词冲突；上衣分层仅描述性'))
    print('CONTINUATION_PREPARED', json.dumps(checkpoints), flush=True)


def compare():
    from tools.t3_local_pilot_report import paired
    old = pilot.read(ROOT/ARCHIVE/'metrics/per_image.json')
    new = pilot.read(ROOT/'metrics/per_image.json')
    old_lookup = {(r['arm'],r['condition'],r['id']):r for r in old}
    changes = []
    for arm in ['residual','reference']:
        for condition in ['original','swapped','synthetic']:
            metrics = ['iou','edge_f1','background_mae','gray_dists','gray_margin','spectrum']
            if condition == 'synthetic':
                metrics += ['orientation_correct','period_relative_error']
            for metric in metrics:
                by_identity = {}
                conditions = 0
                for row in new:
                    if row['arm'] != arm or row['condition'] != condition:
                        continue
                    before = old_lookup[arm,condition,row['id']][metric]
                    after = row[metric]
                    if before is not None and after is not None:
                        by_identity.setdefault(row['identity'],[]).append(float(after)-float(before))
                        conditions += 1
                diffs = [float(np.mean(v)) for v in by_identity.values()]
                changes.append(dict(arm=arm,condition=condition,metric=metric,comparison='%d-%d'%(END,START),
                                    n_conditions=conditions,resampling_unit='identity',**paired(diffs)))
    pilot.write(ROOT/'metrics/continuation_paired.json', changes)
    # 相同初始噪声不仅在当轮三臂相同，也必须和上一检查点输出逐例一致。
    rows = pilot.read(ROOT/'inputs/evaluation.json')
    for row in rows:
        for arm in pilot.ARMS:
            name = Path('outputs')/arm/row['condition']/(row['id']+'.json')
            a, b = pilot.read(PARENT/name), pilot.read(ROOT/name)
            assert a['initial_latent_sha256'] == b['initial_latent_sha256']
    # 保留全部合成例，额外报告每张草图，避免类别冲突掩盖上衣自身的结果。
    strata = []
    for total, data in [(START,old),(END,new)]:
        for arm in pilot.ARMS:
            identities = sorted({r['identity'] for r in data if r['condition']=='synthetic'})
            for identity in identities:
                part = [r for r in data if r['arm']==arm and r['condition']=='synthetic' and r['identity']==identity]
                strata.append(dict(total_updates=0 if arm=='T3' else total, arm=arm, identity=identity, n=len(part),
                    orientation_accuracy=float(np.mean([r['orientation_correct'] for r in part])),
                    period_relative_error=float(np.mean([r['period_relative_error'] for r in part]))))
    pilot.write(ROOT/'metrics/synthetic_by_sketch.json', strata)
    chosen = [r for r in rows if r['condition']=='swapped'][:6]+[r for r in rows if r['condition']=='synthetic']
    for page,start in enumerate(range(0,len(chosen),4),1):
        part = chosen[start:start+4]
        panel = Image.new('RGB',(1000,270*len(part)),'white')
        draw = ImageDraw.Draw(panel)
        for j,row in enumerate(part):
            suffix = Path(row['condition'])/(row['id']+'.png')
            paths = [row['sketch'],row['reference'],ROOT/'outputs/T3'/suffix,
                     PARENT/'outputs/reference'/suffix,ROOT/'outputs/reference'/suffix]
            for col,(label,path) in enumerate(zip(['Sketch','Reference','T3','Reference %d'%START,'Reference %d'%END],paths)):
                image = Image.open(path).convert('RGB')
                image.thumbnail((190,235),Image.Resampling.LANCZOS)
                panel.paste(image,(col*200,j*270+30))
                draw.text((col*200+4,j*270+3),label,fill='black')
            draw.text((4,j*270+251),row['id'],fill='black')
        (ROOT/'figures').mkdir(exist_ok=True)
        panel.save(ROOT/'figures'/('continue_%02d.png'%page))
    pilot.write(ROOT/'audit/continuation_comparison.json',dict(paired_latent_checks=120,
                final_updates=END,new_updates=END-START,optimizer_restarted=not RESUME_OPTIMIZER,
                fixed_baseline_outputs_reused=True,changed_rules=False))
    print('CONTINUATION_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','fit','evaluate','package'])
    parser.add_argument('--arm',choices=['residual','reference'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'fit':
        assert args.arm
        pilot.train(args.arm)
        pilot.generate(args.arm)
    elif args.action == 'evaluate':
        from tools.t3_local_pilot_report import evaluate
        evaluate()
        compare()
    else:
        pilot.package()
