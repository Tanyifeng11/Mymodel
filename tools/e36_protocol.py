"""E36 训练前固定配置；只有实际通过门槛才进入下一阶段。"""
import random
from pathlib import Path
from tools.e35_freeu_protocol import read,write,sha,commit,bootstrap,E5,SIZE,METRICS,PREVIOUS

OUT=Path('output_eval/e36_dagf_guided_filter_20261010')
E35=Path('output_eval/e35_freeu_skip_20261010')
ARMS=['B1_CONV','B2_DAGF_LITE','B3_CONFLICT']
CONFIG=dict(experiment='E36_DAGF_LITE',version=1,seed_train=42,seed_split=36042,
    seed_generation=42,seed_bootstrap=32042,bootstrap_reps=10000,
    train_id_count=1024,updates=800,effective_batch=8,micro_batch=1,lr=.0001,
    weight_decay=.01,clip_grad=1.,projection=64,alpha=.1,checkpoint_steps=[0,100,400,800],
    width=384,height=512,steps=50,cfg=7.,sketch_scale=.6,texture_scale=1.,ipa_scale=1.,
    dropout=dict(image=.05,text=.20,both=.05,sketch=0.),
    cfg_adapter_policy='conditional only; unconditional adapter OFF by design',
    revision=[dict(reason='E5 training always retains sketch/SA; true CFG unconditional lacks SA and has distinct zero BF embeddings',
        resolution='inherit original joint dropout exactly; adapter only conditional, trace both; no added unconditional training rule')],
    caption_unchanged=True,dataset_modified=False,train_timestep='uniform 0..999 epsilon target',
    safety=dict(edge_min=-.01,clip_min=-.005,tcf_max=.5,leak_max=.005,lpips_max=.01,
        catastrophic_iou_drop=.15,catastrophic_edge_drop=.15,catastrophic_leak_rise=.10),
    promotion=dict(iou_min=.003,edge_min=0.,iou_vs_conv_min=.002),
    confirm=dict(iou_min=.005,iou_ci_lower_min=0.,mask_coverage_min=.8),
    c2_status='not_run; configuration only frozen if C1 passes')


def decision(**updates):
    path=OUT/'decision.json'
    state=read(path) if path.exists() else dict(experiment='E36_DAGF_LITE',date='2026-10-10',
        p0_replay_pass=None,p0_hook_trace_pass=None,p0_grad_pass=None,p1_run_complete=None,
        p1_dagf_beats_conv=None,p1_conflict_beats_dagf=None,p1_full_rgb_safety_pass=None,
        c1_confirm96_pass=None,c2_val500_three_seed_pass=None,new_method_supported=False,stop_reason=None)
    state.update(updates);write(path,state);return state


def verify_sources():
    frozen=read(OUT/'audit/frozen_hashes.json')
    assert all(sha(p)==h for p,h in frozen.items()),'冻结源文件改变'
    write(OUT/'audit/frozen_check.json',dict(pass_unchanged=True,files=len(frozen)))


def prepare():
    import platform,torch,diffusers,cv2,PIL
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==CONFIG;return
    old=read(E35/'audit/frozen_hashes.json');assert all(sha(p)==h for p,h in old.items())
    train=read(PREVIOUS/'splits/train.json');dev=read(E35/'splits/dev32.json');confirm=read(E35/'splits/confirm96.json')
    chosen=random.Random(CONFIG['seed_split']).sample(sorted(train,key=lambda r:r['id']),1024)
    assert not {r['id'] for r in chosen}&{r['id'] for r in dev+confirm}
    for name,rows in [('train1024',chosen),('dev32',dev),('confirm96',confirm)]:write(OUT/'splits'/(name+'.json'),rows)
    frozen=dict(old)
    for row in chosen:
        for key in ['gt','sketch','reference']:frozen[row[key]]=sha(row[key])
    for p in (OUT/'splits').glob('*.json'):frozen[str(p)]=sha(p)
    write(OUT/'audit/frozen_hashes.json',frozen)
    original=read('data/train_bf_texture.json');index={Path(r['cloth']).stem:r for r in original}
    write(OUT/'splits/train_original_records.json',[index[r['id']] for r in chosen])
    write(OUT/'protocol.json',CONFIG);decision()
    write(OUT/'audit/environment.json',dict(python=platform.python_version(),torch=torch.__version__,
        diffusers=diffusers.__version__,opencv=cv2.__version__,pillow=PIL.__version__,commit=commit()))
    write(OUT/'audit/dataset_audit.json',dict(original=45384,validation=500,dev=128,remaining_train=len(train),
        pilot_train=len(chosen),disjoint=True,confirmation_not_inspected=True,
        splits={p.name:sha(p) for p in (OUT/'splits').glob('*.json')}))
    print('E36 protocol frozen',flush=True)


if __name__=='__main__':prepare()
