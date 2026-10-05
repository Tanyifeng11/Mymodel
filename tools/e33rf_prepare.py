"""复用已核验RC输入，不重建target GT/划分；只生成新真实旋转输入。"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from data.e33rf_real_rotation_dataset import rotation_path,cpu_rotation,write_rotations
from tools.e33rf_common import *

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.json').exists():
        old=read(OUT/'protocol.json')
        if old['sha256']!=protocol_sha():
            # 仅允许在零训练步时修正原协议的全批次聚合；保留旧元数据和缓存。
            assert not any(OUT.rglob('training_protocol.json'))
            assert old['protocol']=={k:v for k,v in PROTOCOL.items() if k!='loss_aggregation'}
            write(OUT/'protocol_before_aggregation_fix.json',old)
            write(OUT/'aggregation_fix_before_training.json',dict(before=old['sha256'],after=protocol_sha(),training_steps=0,
                reason='global valid-target averaging instead of averaging independently inside microbatches',rotation_cache_unchanged=True))
            old.update(protocol=PROTOCOL,sha256=protocol_sha(),git_commit=git_commit());write(OUT/'protocol.json',old)
            write(OUT/'prepared.json',dict(complete=True,protocol_sha256=protocol_sha()))
        finish_frozen();return
    assert read(RC/'completion_check.json')['experiment_complete']
    assert read(RC/'frozen_check.json')['pass']
    frozen=dict(read(RC/'frozen_check.json')['after'])
    for name in ['split_manifest.json','controlled_manifest.json','real_wrong_train.json','real_wrong_eval.json']:
        p=RC/name;frozen[str(p)]=sha(p);write(OUT/name,read(p))
    for seed in SEEDS:
        p=source_checkpoint(seed);frozen[str(p)]=sha(p)
    frozen[str(RC/'decision_summary.json')]=sha(RC/'decision_summary.json')
    write(OUT/'frozen_check.json',dict(before=frozen,after=None,**{'pass':None},E5_training_steps=0))
    write(OUT/'protocol.json',dict(protocol=PROTOCOL,sha256=protocol_sha(),git_commit=git_commit(),user_confirmation='开始实施',
        plan_sha256='23147d5ad0db7e135fb940da25d5c6ee983d80183f01731151a9f34d1429608d'))
    split=read(OUT/'split_manifest.json');cf=read(OUT/'controlled_manifest.json')
    assert [len(split[x]) for x in ['train','dev','causal_test','independent_confirmation']]==[45126,256,8,10]
    assert len(cf['dev'])==128 and sum(x['strict'] for x in cf['dev'])==45
    write(OUT/'feature_drift_ids.json',[r['id'] for r in hash_order(split['dev'],'E33RF/drift')[:64]])
    finish_frozen();write(OUT/'prepared.json',dict(complete=True,protocol_sha256=protocol_sha()))
def cache_rotations(dino,training=False):
    split=read(OUT/'split_manifest.json')
    rows=split['train'] if training else split['dev']+split['causal_test']+split['independent_confirmation']
    manifest=OUT/('rotation_train_manifest.json' if training else 'rotation_eval_manifest.json')
    if manifest.exists():
        expected=read(manifest);assert set(expected['files'])=={r['id'] for r in rows}
        assert all(sha(rotation_path(sid))==digest for sid,digest in expected['files'].items());return
    pending=[r for r in rows if not rotation_path(r['id']).exists()]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        for start in range(0,len(pending),64):
            for result in pool.map(cpu_rotation,pending[start:start+64]):write_rotations(result,dino)
            print('[E33RF rotation cache]',training,min(start+64,len(pending)),'/',len(pending),flush=True)
    write(manifest,dict(files={r['id']:sha(rotation_path(r['id'])) for r in rows},
        native_canvas=[384,512],same_reference=True,no_target_GT_in_feature_encoding=True))
