"""单GPU作业自动完成固定顺序与唯一统一修订，网页断线不会重训。"""
from models.apacc_features import load_dino
from tools.e33rc_common import *
from tools.e33rc_prepare import prepare
from tools.e33rc_evaluate import evaluate_dual
from tools.e33rc_train_curriculum import train_seed

def zero_shot():
    import torch
    dino,_=load_dino('cuda',WEIGHTS)
    for seed in [42,43,44]:
        folder=OUT/'RC0_zero_shot'/('seed%d'%seed)
        if (folder/'phase_complete.json').exists():continue
        model,digest=load_control(seed);model.eval()
        evaluate_dual(model,seed,folder,dino)
        write(folder/'phase_complete.json',dict(training_steps=0,source_sha256=digest,baseline_only_not_success_gate=True))
        del model;torch.cuda.empty_cache()
    del dino;torch.cuda.empty_cache()

def audit_seed42(revision):
    folder=seed_folder(42,revision);status=read(folder/'run_status.json')
    checks=dict(source_sha=read(folder/'training_protocol.json')['initial_sha256']==sha(source_checkpoint(42)),
        fresh_optimizer=not read(folder/'training_protocol.json')['optimizer_state_inherited'],
        prior_no_gradient=not read(folder/'gradient_check.json')['prior_has_gradient'],
        real_rotation_eval_only=read(folder/'training_protocol.json')['no_real_rot90_in_training'])
    for decision in status['stage_decisions']:
        stage=folder/decision['stage'];done=read(stage/'phase_complete.json')
        checks[decision['stage']+'/checkpoint']=done['checkpoint_sha256']==sha(stage/'checkpoint_final.pt')
        checks[decision['stage']+'/controlled_denominator']=read(stage/'controlled/summary.json')['denominator']==128
        checks[decision['stage']+'/real_denominator']=read(stage/'real/summary.json')['case_count']==256
        checks[decision['stage']+'/strict_denominator']=read(stage/'controlled/summary.json')['strict']['denominator']==45
    write(folder/'implementation_audit.json',dict(checks=checks,**{'pass':all(checks.values())}));assert all(checks.values())

def main():
    prepare()
    from tools.e33rc_gpu_audit import gpu_audit
    gpu_audit();zero_shot()
    revision=selected_revision();statuses=[]
    if not revision:
        for seed in [42,43,44]:
            status=train_seed(seed)
            if seed==42:audit_seed42(False)
            statuses.append(status)
            if status['collapse']:
                path=OUT/'retention_revision/authorization.json'
                write(path,dict(trigger_seed=seed,trigger_status=status,revision_number=1,
                    user_confirmed='按此实施',configuration=PROTOCOL['revision'],original_trials_retained=True))
                revision=True;break
    if revision:
        statuses=[]
        for seed in [42,43,44]:
            status=train_seed(seed,True)
            if seed==42:audit_seed42(True)
            statuses.append(status)
            # 修订配置不再次改变；其余seed仍按同一配置验证2/3，单seed停止保持不变。
    selected={str(s['seed']):s for s in statuses};write(OUT/'selected_full_runs.json',dict(revision=revision,seeds=selected))
    trigger=selected['42']['completed_RC_C']
    write(OUT/'ablations/status.json',dict(required=trigger,variants=ABLATIONS,reason='selected full seed42 completed RC-C'))
    if trigger:
        for variant in ABLATIONS:train_seed(42,revision,variant)
    from tools.e33rc_report import report
    report()

if __name__=='__main__':main()
