"""在GPU作业内审计既有缓存与冻结来源；禁止重新生成划分/伪GT。"""
import numpy as np
from data.e32_field_dataset import cache_path
from data.e32_wrong_reference_sampler import wrong_references
from data.e33rc_wrong_reference import train_wrong_references
from tools.e33rc_common import *

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.json').exists():
        assert read(OUT/'protocol.json')['sha256']==protocol_sha();finish_frozen();return
    assert read(E33R/'completion_check.json')['experiment_complete']
    assert read(E33R/'decision_summary.json')['controlled_rotation_causality_pass']
    split=read(E32/'split_manifest.json');controlled=read(E33R/'selected_manifest.json')
    assert [len(split[g]) for g in ['train','dev','causal_test','independent_confirmation']]==[45126,256,8,10]
    assert [len(controlled[g]) for g in ['train','dev','causal_test','independent_confirmation']]==[25272,128,4,5]
    groups=[{r['id'] for r in split[g]} for g in ['train','dev','causal_test','independent_confirmation']]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    write(OUT/'protocol.json',dict(protocol=PROTOCOL,sha256=protocol_sha(),git_commit=git_commit(),user_confirmation='按此实施'))
    write(OUT/'split_manifest.json',split);write(OUT/'controlled_manifest.json',controlled)
    probes=read(E33R/'sanity_manifest.json')['train']
    assert len(probes)==512 and {r['id'] for r in probes}<=groups[0]
    write(OUT/'teacher_probes.json',probes)
    inherited=read(E33R/'frozen_check.json')['after']
    frozen=dict(inherited)
    provenance_paths=[E32/'split_manifest.json',E32/'A_geometry/cache_manifest.json',E32/'audits/target_group_split_audit.json',
        E33R/'selected_manifest.json',E33R/'sanity_manifest.json',E33R/'P0_prior/checkpoint_final.pt']
    provenance_paths += [source_checkpoint(s) for s in [42,43,44]]
    provenance_paths += [Path(p) for p in ['models/e33r_rotation_control.py','tools/e33r_losses.py','data/e33r_group_dataset.py','tools/e33r_evaluate.py']]
    for p in provenance_paths:frozen[str(p)]=sha(p)
    for seed in [42,43,44]:
        p=source_checkpoint(seed);record=read(p.parent/'phase_complete.json')
        assert record['steps']==8000 and record['checkpoint_sha256']==sha(p)
    write(OUT/'frozen_check.json',dict(before=frozen,after=None,**{'pass':None},training_steps=0,E5_training_steps=0))
    write(OUT/'input_provenance.json',dict(E32_split_sha256=sha(E32/'split_manifest.json'),E33R_manifest_sha256=sha(E33R/'selected_manifest.json'),
        source_checkpoints={str(s):dict(path=str(source_checkpoint(s)),sha256=sha(source_checkpoint(s))) for s in [42,43,44]},
        actual_real_counts=[45126,256,8,10],controlled_counts=[25272,128,4,5],no_new_pseudogt=True))
    oldhash=read(E32/'audits/stage0_input_hashes.json');split_audit=read(E32/'audits/target_group_split_audit.json')
    pool=split['dev']+split['confirmation_all'];features={}
    for row in pool:
        with np.load(cache_path(E32,row['id'])) as z:features[row['id']]=dict(histogram=np.asarray(z['reference_histogram']))
    wrong=wrong_references(pool,features,oldhash);write(OUT/'real_wrong_eval.json',wrong)
    # 与E32已完成seed42逐例表逐项核对同一heldout donor protocol。
    e32rows=read(E32/'A_geometry/seed42/dev/rows.json')
    assert all(r['wrong_references']==wrong[r['id']] for r in e32rows)
    if not (OUT/'real_wrong_train.json').exists():
        hashes={};hist=[]
        for i,row in enumerate(split['train'],1):
            target_sha=split_audit['training_target_hashes'][row['id']]
            assert target_sha not in split_audit['heldout_hashes']
            assert sha(DATASET/row['target'])==target_sha
            hashes[row['id']]=dict(target=target_sha,reference=sha(DATASET/row['reference']))
            with np.load(cache_path(E32,row['id'])) as z:hist.append(np.asarray(z['reference_histogram']))
            if i%4096==0:print('[E33RC train input audit]',i,'/',len(split['train']),flush=True)
        write(OUT/'audits/train_reference_hashes.json',hashes)
        write(OUT/'real_wrong_train.json',train_wrong_references(split['train'],hist,hashes))
    write(OUT/'audits/input_protocol.json',dict(real_train=45126,real_dev=256,causal=8,independent=10,
        controlled_dev=128,controlled_strict_dev=45,teacher_probes=512,
        heldout_donors_identical_E32=True,train_donors_only_train=True,target_hash_disjoint=True,
        reference_input_channels=394,removed_channel='E26 log_frequency index386',
        real_rot90='same E32 fixed384x512canvas true90 crop/pad; eval only',target_rgb_is_model_input=False))
    finish_frozen();write(OUT/'prepared.json',dict(complete=True,protocol_sha256=protocol_sha()))

if __name__=='__main__':prepare()
