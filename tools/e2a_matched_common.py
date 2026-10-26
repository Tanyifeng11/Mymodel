"""E2A-3D-2D-TEX-01：新根运行，旧权重、源码和结果只读。"""
import ast
import copy
from pathlib import Path
from tools.e2a_attr_common import (PROJECT, SOURCE, read, write, freeze as strict_freeze, sha,
    csv_write, commit, CONFIG as HISTORICAL_CONFIG)
from tools.e2a_sup_common import EXPECTED as INITIAL_SHA, model_path as prior_model

ROOT = PROJECT/'output_eval/e2a_3d2d_tex_01_20261026'
SUP = PROJECT/'output_eval/e2a_sup_comp_01_20261021'
OLD = PROJECT/'output_eval/e2a_attr_01_20261019'
PLAN_SHA = 'e8413e18b9583f423e59843d6d49544e0a69aa9aca617581e5af7dd052863684'
T3_SHA = '25f95844696ba5c07eb9223e3bdd4d68c78ba4f1a36d561569c35cc90186d259'
CONFIG = dict(protocol='E2A-3D-2D-TEX-01',plan_sha256=PLAN_SHA,seed=42,
    updates=28365,pilot_updates=800,world_size=2,effective_batch=8,accumulation=4,
    calibration_samples=512,calibration_effective_batches=64,
    body_rgb_coefficient=.1,region_kernel_size=9,
    eta=999,auxiliary_time_gate=False,
    gamma_selection='median over 64 effective batches of RMS(.05*dBoundary+.1*dOutside)/RMS(dRest); epsilon gradients; 8 samples per batch; single fixed scalar',
    calibration_uses_P2_or_LOCK=False,gradient_target='match C weighted boundary+outside, NOT .10 of base',
    primary_family=['C minus T3M IoU','C minus T3M Leak'],statistics_seed=221042,
    bootstrap_samples=10000,alpha=.05,
    meaningful_gain=HISTORICAL_CONFIG['meaningful_gain'],
    noninferiority=HISTORICAL_CONFIG['noninferiority'],
    training_walltime='7-00:00:00',soft_pause_seconds=0,
    stage1='T3M seed42 then P2b development + paired swap diagnostics',
    stage2='conditional C/T3M matched seed43 after development review',
    stage3='conditional unopened LOCK128; freeze metrics and input annotations before generation',
    seed44_authorized=False,LOCK_opened=False,human_blind_ratings=None)

TRAINING_SOURCES=['tools/e2a_matched_common.py','tools/e2a_matched_train.py',
    'tools/e2a_sup_train.py','tools/e2a_sup_losses.py']


def freeze(path,value):
    # 仅规范 Diffusers 来自 set 的默认字段名顺序；其余真实参数严格比较。
    from tools.e2a_sup_common import canonical_environment
    path=Path(path)
    if path.name.startswith('environment_rank'):
        value=canonical_environment(value)
    return strict_freeze(path,value)


def folder(arm='T3M',phase='formal'):
    assert arm=='T3M' and phase in ['calibrate','pilot','formal']
    return ROOT/'training'/phase/arm


def model_path(arm):
    return folder()/'checkpoint-final/joint_model.pt' if arm=='T3M' else prior_model(arm)


def status(**changes):
    p=ROOT/'status.json'
    v=read(p) if p.exists() else dict(complete=False,LOCK='NOT_OPENED',engineering_failure=None)
    v.update(changes);write(p,v);return v


def helper_sources():
    source=(SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    names=['_weighted_mean_per_sample','_masked_channel_mean_std','texture_color_stat_loss',
           'build_region_masks','masked_l1_loss']
    return {n:ast.get_source_segment(source,next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==n)) for n in names}


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    for p in ['audit','logs','splits','evaluation','statistics','training']:(ROOT/p).mkdir(exist_ok=True)
    freeze(ROOT/'config.json',CONFIG)
    if (ROOT/'audit/preparation_complete.json').exists():return
    assert read(SUP/'audit/preparation_complete.json')['passed']
    originals=read(SUP/'audit/historical_source_sha256.json')
    assert all(sha(SOURCE/p)==h for p,h in originals.items())
    freeze(ROOT/'audit/historical_source_sha256.json',originals)
    registry={}
    for arm,expected in dict(INITIAL_SHA,T3=T3_SHA).items():
        path=model_path(arm);assert sha(path)==expected,(arm,'CHECKPOINT_CHANGED')
        registry[arm]=dict(path=str(path),sha256=expected)
    freeze(ROOT/'checkpoint_registry.json',registry)
    c=read(OLD/'training/seed42/F111/config.json')
    assert c['source_sha256']==originals
    argv=c['argv']
    for flag,value in [('--lambda_region_texture',.1),('--lambda_boundary',.05),('--lambda_leak',.1)]:
        assert float(argv[len(argv)-1-argv[::-1].index(flag)+1])==value
    def arguments(values):
        assert len(values)%2==0
        return dict(zip(values[::2],values[1::2]))
    ca=arguments(argv);ba=arguments(read(OLD/'training/seed42/F000/config.json')['argv'])
    excluded={'--lambda_region_texture','--lambda_boundary','--lambda_leak','--output_dir','--attr_audit'}
    assert {k:v for k,v in ca.items() if k not in excluded}=={k:v for k,v in ba.items() if k not in excluded}
    assert ca['--region_kernel_size']=='9'
    freeze(ROOT/'audit/C_base_match.json',dict(passed=True,C_vs_B_argv_only_region_terms_and_output_differ=True,
        base_arguments={k:v for k,v in ba.items() if k not in excluded},
        C_config_sha256=sha(OLD/'training/seed42/F111/config.json'),
        B_config_sha256=sha(OLD/'training/seed42/F000/config.json')))
    freeze(ROOT/'audit/C_config.json',c)
    helpers=helper_sources()
    legacy=(SOURCE/'tools/e2a_sup_losses.py').read_text(encoding='utf-8')
    assert "self.h['texture_color_stat_loss'](pred,reference,body,sample_weight=texture_weight)" in legacy
    freeze(ROOT/'loss_definition.json',dict(
        body='exact historical texture_color_stat_loss: RGB mean/std smooth L1; Ref target; historical texture condition weight reduction',
        C='.1 bodyRGB + .05 maskedL1(GT,boundary) + .1 maskedL1(GT,outside)',
        T3M='.1 bodyRGB + gamma maskedL1(GT,1-body); no extra boundary/outside contribution',
        helpers=helpers,mask_kernel=9,
        document_correction='Section 2.1 calls legacy T3 body masked L1; actual T3 and C both call identical RGB mean/std statistic loss. Main new control differs by gradient-matched rest coefficient.',
        correction_changes_historical_results=False))
    freeze(ROOT/'audit/calibration_protocol.json',dict(samples=512,effective_batches=64,optimizer_updates=0,
        order='same first 512 training samples and rank streams as previous calibration; four microbatches/rank per update',
        pooling='sum squared epsilon-gradient RMS across 8 same-shaped microbatches -> sqrt ratio; median of 64 ratios',
        denominator_zero='fail and review; never silently drop an effective batch',
        gamma_fixed_for_all_samples_and_timesteps=True,used_P2_or_LOCK=False))
    freeze(ROOT/'audit/development_rule.json',dict(
        frozen_before_T3M_outputs=True,source='existing frozen E2a meaningful_gain/noninferiority margins',
        primary_family=['iou','leak'],primary_test='two-sided paired sign flip, two-test Holm',
        continue_structural='at least one oriented primary mean >= meaningful_gain; all other structure endpoints not clearly_failed noninferiority',
        meaningful_gain=CONFIG['meaningful_gain'],noninferiority=CONFIG['noninferiority'],
        texture_gate='review paired P2b/swap diagnostics; no coefficient tuning; explicit review before seed43/LOCK',
        absence_of_clear_degradation_is_not_proof_of_noninferiority=True,
        automatic_seed43_or_LOCK=False,human_preference_NI_lower_bound=.45,
        human_preference_superiority_lower_bound=.5,AI_does_not_replace_human=True))
    freeze(ROOT/'audit/mask_manifest.json',dict(training_masks=str(OLD/'masks/train'),
        historical_mask_audit_sha256=sha(OLD/'mask_audit.csv'),
        retained_defects=17,source_audit_sha256=sha(SUP/'audit/reused_mask_audit.json')))
    freeze(ROOT/'audit/train_manifest_registry.json',read(SUP/'audit/train_manifest_registry.json'))
    for p,d in read(SUP/'audit/train_manifest_registry.json').items():assert sha(p)==d['sha256']
    for split in ['P2a','P2b']:freeze(ROOT/'splits'/(split+'.json'),read(SUP/'splits'/(split+'.json')))
    # LOCK 只记冻结字节哈希，不解析身份或打开图像。开发门槛未过不得揭盲。
    freeze(ROOT/'audit/LOCK_sealed.json',dict(path=str(SUP/'splits/LOCK.json'),
        sha256=sha(SUP/'splits/LOCK.json'),n=128,identities_not_parsed=True,images_not_opened=True))
    from tools.e2a_sup_reference import approved_reference_manifest
    manifest,rows=approved_reference_manifest()
    freeze(ROOT/'splits/reference_swap.json',rows)
    freeze(ROOT/'audit/reused_reference_release.json',dict(path=str(manifest),sha256=sha(manifest),
        release_sha256=sha(SUP/'audit/reference_swap_release.json'),n=32))
    freeze(ROOT/'audit/reused_metric_protocol.json',read(SUP/'evaluation/metric_protocol.json'))
    for stage in ['P2b','reference_swap']:(ROOT/'evaluation'/stage).mkdir(exist_ok=True)
    for arm in ['B','C','T3']:
        for stage in ['P2b','reference_swap']:
            original=SUP/'evaluation'/stage/arm
            assert read(original/'evaluation_complete.json')['complete']
            target=ROOT/'evaluation'/stage/arm
            if not target.exists():target.symlink_to(original,target_is_directory=True)
            assert target.resolve()==original.resolve()
    freeze(ROOT/'audit/preparation_complete.json',dict(passed=True,source_commit=commit(),
        source_sha256={p:sha(SOURCE/p) for p in TRAINING_SOURCES},
        C_and_T3_body_identical=True,old_results_read_only=True,LOCK_opened=False))
    status(stage='CALIBRATION_PENDING',complete=False,engineering_failure=None)
    print('MATCHED_PREPARATION_PASSED',flush=True)


if __name__=='__main__':prepare()
