"""记录用户对sanity R90硬停止的单项覆盖；保留原失败及原结果快照。"""
import argparse,shutil
from datetime import datetime,timezone
from tools.e33r_common import *

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-sanity-r90',action='store_true',required=True)
    args=parser.parse_args()
    assert read(OUT/'R_sanity/gate.json')['checks']==dict(clean_r90_success=False,
        r180_identity_success=True,r0_identity_success=True,finite=True)
    assert read(OUT/'decision_summary.json')['prior_pass']
    path=OUT/'sanity_override.json'
    if path.exists():
        print(sanity_continuation(),flush=True);return
    assert all(not (OUT/'P1_controlled'/('seed%d'%s)/'checkpoint_final.pt').exists() for s in PROTOCOL['seeds'])
    archive=OUT/'history/sanity_stop_20261002';archive.mkdir(parents=True,exist_ok=True)
    for name in ('decision_summary.json','completion_check.json','artifact_manifest.json','local_review_bundle.tar.gz'):
        if (OUT/name).exists():shutil.copy2(OUT/name,archive/name)
    write(path,dict(authorizing_request='忽视R90 成功率 **93.36% < 95%**  的停止条件继续往后做',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),overridden_check='clean_r90_success',
        original_sanity_pass=False,original_success=read(OUT/'R_sanity/train/summary.json')['clean_r90_success'],
        sanity_gate_sha256=sha(OUT/'R_sanity/gate.json'),
        sanity_train_summary_sha256=sha(OUT/'R_sanity/train/summary.json'),
        prior_checkpoint_sha256=read(OUT/'P0_prior/summary.json')['checkpoint_sha256'],
        unchanged='P0 Gate, formal per-seed Gate, 2of3 rule, seed42 ablation trigger, data and training protocol'))
    assert sanity_continuation()['allowed']
    update_decision(sanity_gate_overridden=True,next_route='P1_full',controlled_rotation_causality_pass=None)
    write(OUT/'completion_check.json',dict(experiment_complete=False,status='formal_training_in_progress',
        sanity_pass=False,sanity_gate_overridden=True,training_steps=6500,
        previous_completion='history/sanity_stop_20261002/completion_check.json'))
    print(sanity_continuation(),flush=True)

if __name__=='__main__':main()
