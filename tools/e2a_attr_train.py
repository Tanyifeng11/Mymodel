"""从历史命令创建各臂；除开关外保持同阶段训练设置一致。"""
import argparse
import os
import runpy
import sys
from tools.e2a_attr_common import *


def train(seed, factor, smoke=False):
    import torch
    folder=train_folder(seed,factor,smoke); folder.mkdir(parents=True,exist_ok=True)
    world=int(os.environ.get('WORLD_SIZE','1')); assert world in [1,2,4,8]
    updates=8 if smoke else CONFIG['updates']
    if not smoke: assert read(OUT/'audit/mask_visual_review.json')['passed']
    assert read(OUT/'audit/masks_complete.json')['passed']
    argv=list(read(OUT/'manifests/historical_command.json'))
    argv += ['--output_dir',str(folder),'--start_global_step','0','--seed',str(seed),
        '--attr_audit',str(folder),'--attr_mask_cache',str(OUT/'masks/train'),
        '--gradient_accumulation_steps',str(8//world),'--max_train_steps',str(updates),
        '--checkpointing_epochs','9999','--dataloader_num_workers','2',
        '--lambda_region_texture',str(.1*int(factor[0])),
        '--lambda_boundary',str(.05*int(factor[1])),'--lambda_leak',str(.1*int(factor[2])),
        '--report_to','none','--wandb_mode','disabled','--val_vis_steps','0','--vis_every_n_steps','0']
    config=dict(seed=seed,factor=factor,smoke=smoke,world_size=world,effective_batch=8,
        accumulation=8//world,updates=updates,execution_commit=commit(),argv=argv,
        scheduler=CONFIG['scheduler'],amp=CONFIG['amp'],plan_sha256=PLAN_SHA)
    if int(os.environ.get('RANK','0'))==0:
        freeze(folder/'config.json',config)
        freeze(folder/'loss_factor_manifest.json',dict(body_rgb=dict(weight=.1*int(factor[0]),target='reference_texture'),
            boundary=dict(weight=.05*int(factor[1]),target='GT'),outside=dict(weight=.1*int(factor[2]),target='GT')))
        freeze(folder/'checkpoint_manifest.json',read(OUT/'checkpoint_manifest.json'))
        (folder/'data_manifest_sha256.txt').write_text(sha(TRAIN)+'  '+str(TRAIN)+'\n',encoding='utf-8')
        target=folder/'mask_audit.csv'
        if not target.exists(): target.symlink_to(OUT/'mask_audit.csv')
        assert not (folder/'training_complete.json').exists(),'禁止重复训练已完成科学运行'
    sys.argv=[str(SOURCE/'train_GAM_texture_joint.py')]+argv
    runpy.run_path(sys.argv[0],run_name='__main__')
    if int(os.environ.get('RANK','0'))==0:
        path=folder/'checkpoint-final/joint_model.pt'
        assert path.resolve().parent.name=='checkpoint-%d'%updates
        record=read(folder/'optimizer_update_count.json')
        assert record['actual_successful_updates']==updates and record['skipped_updates']==0
        weights=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        for key,value in [('lambda_region_texture',.1*int(factor[0])),('lambda_boundary',.05*int(factor[1])),
                          ('lambda_leak',.1*int(factor[2])),('layer_group_enabled',1),('texture_condition_mode','token')]:
            assert weights['meta'][key]==value,(key,weights['meta'][key])
        finite=0
        for group in weights.values():
            if not isinstance(group,dict): continue
            for value in group.values():
                if torch.is_tensor(value): assert torch.isfinite(value).all().item(); finite+=1
        write(folder/'checkpoint_verification.json',dict(passed=True,path=str(path),sha256=sha(path),
            actual_updates=updates,finite_tensors=finite,seed=seed,factor=factor))
        print('CHECKPOINT VERIFIED',seed,factor,updates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--factor',choices=FACTORS,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();train(args.seed,args.factor,args.smoke)
