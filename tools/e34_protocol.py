"""E34 预注册常量；未标注项保持未知，不用自动代理填充人工 GT。"""
from pathlib import Path

OUT = Path('output_eval/e34_capf_20261004')
E32 = Path('output_eval/e32_target_supervised_pattern_field_20261001')
E29 = Path('output_eval/e29_20260930')
E30 = Path('output_eval/e30_a2_affinity_20260930')
E33R = Path('output_eval/e33r_rotation_causality_20261002')
GROUPS = ('train', 'dev', 'causal_test', 'independent_confirmation')
CAUSAL_IDS = (6, 7, 9, 10, 12, 13, 14, 17)
FAMILIES = ('stripe', 'plaid', 'repeated_print', 'floral', 'geometric', 'mixed')
PLAN_SHA = '05e686de8e9d896ebc5936c918d334ee8486769aa95187e22f3f00cf53bcc834'

PROTOCOL = dict(
    experiment='E34-CAPF', version=1, plan_sha256=PLAN_SHA,
    authorization='用户确认：开始实施；没有覆盖人工标注或实验停止条件',
    inherited_split=dict(train=45126, dev=256, causal_test=8, independent_confirmation=10),
    evidence_source='完整服装：E32 train/dev 的 target RGB；不是 texture patch；仅用于参考端 evidence 标注/训练',
    evidence_selection='固定caption词规则：服装类型+pattern cue，排除bag类配饰；随后SHA256排序；不使用纹样标签或成功率筛选，不以纯色/配饰补足数量',
    annotation_minimum=dict(train=100, validation=20, independent_test=20),
    annotation_schema=dict(
        required=['id', 'group', 'reference', 'reference_sha256', 'reviewed_by_human', 'reviewer',
                  'garment_foreground', 'pattern_support', 'evidence', 'pattern_family',
                  'orientation_readable', 'identity_group'],
        evidence_fields=['box', 'confidence', 'orientation_deg', 'structural_contamination'],
        masks='与原参考同尺寸二值PNG，人工复核；pattern_support 必须包含于 garment_foreground',
        coordinates='原图像素 [x0,y0,x1,y1)，1–3 个 canonical boxes',
        confidence=['high', 'medium', 'invalid'], families=list(FAMILIES),
        structural_contamination=['seam', 'pocket', 'logo', 'structure_edge']),
    candidate=dict(sizes=[48, 64, 80, 96], stride=16, foreground_purity=.95),
    evidence_arms=['A0_handcrafted', 'A1_frozen_dino', 'A2_affinity', 'A3_affinity_geometry', 'A4_capf_scorer'],
    seeds=[42, 43, 44], generation_seeds=[42, 43],
    intervention=dict(primary='rot90 within human pattern support',
                      wrong=['random', 'color_near', 'geometry_near'], scale_hard_gate=False,
                      paired=['target_sketch_sha256', 'seed', 'noise_sha256', 'checkpoint_sha256']),
    A_gate=dict(median_crop_iou=.60, identity=.85, homogeneity=.80, contamination=.10,
                top5_recall=.80, evidence_coverage=.80, A4_better_than_A0=True, A4_not_below_A3=True),
    A_stop=dict(evidence_coverage_below=.60, contamination_above=.15, top5_recall_below=.70),
    B_gate=dict(follow=.90, median_direction_error_deg=5., confidence_coverage=.50,
                pattern_correlation=.50, rot90_clean=.90, rot90_paired_nuisance=.90),
    D_gate=dict(background_error_ci95_upper=.03, boundary_error_ci95_upper=.03,
                sketch_iou_delta_ci95_lower=-.02, edge_f1_delta_ci95_lower=-.02,
                leakage_delta_ci95_upper=.02, interior_response_rms_min=1e-4),
    E_gate=dict(follow_recovery=.60, direction_recovery=.60, contour_delta_min=-.02,
                sketch_iou_delta_min=-.02, leakage_delta_max=.01,
                follow_paired_improvement_ci95_lower=0.),
    E_stop=dict(follow_recovery_below=.50, full_not_better_than_base=True,
                confidence_without_safety_gain=True),
    F_gate=dict(minimum_passing_seeds=2, total_seeds=3, matched_better_than_wrong=True,
                rot90_response=.70, independent_confirmation_no_collapse=True, structure_safe=True),
    E_groups=['G0_global', 'G1_old_full_auto', 'G2_evidence_E26_O4',
              'G3_evidence_canonical_O4', 'G4_full_confidence', 'G5_oracle_same_injector'],
    bootstrap=dict(unit='独立 case/reference；seed 和 original/rot90 在 case 内配对', draws=2000, seed=34042),
    fixed=['冻结 E5', 'E29 target/assignment/warp/compositor', 'E22-O4 injection position',
           'E5 原正向条件', '原 DDIM/CFG；不修改 negative branch'],
    sequence=['0_protocol_and_human_annotation', 'A_evidence', 'B_canonicalization',
              'C_field', 'D_injection', 'E_end2end', 'F_causal', 'G_generalization', 'H_tables'],
    stop_policy='人工标注缺失/泄漏/干预完整性失败：不训练；A/B未通过：不运行 diffusion',
    generalization=dict(minimum_references=30, shapes=3, variants=2, seeds=2),
    limitations=['45k split 未经 unseen-pattern-family 划分',
                 '历史 E33-R 的 target self-reference 旋转结果不能替代 E34 完整 garment reference 的干预完整性',
                 'E34-0 审计或待标注包不代表 E34-A–H 实验完成'])

METRICS = dict(
    crop_iou='候选box和人工canonical box的交并比；每reference中位数后跨reference中位数',
    top5_recall='人工有效evidence中被top5候选IoU>=0.5命中的比例；先在reference内聚合',
    coverage='至少选出一个人工支持内有效evidence的reference比例；invalid reference单列',
    contamination='选中crop内不属于人工pattern_support或被标为结构污染的像素比例',
    homogeneity='复用E29 quadrant descriptor pair similarity，使用人工GT审计污染',
    identity='复用E27 FFT/ACF/RGB similarity；另报告冻结affinity identity，不能替代感知质量',
    follow='复用E28共同manual prototype original/rot90配对定义；identity>=.65且可读方向误差<=20deg',
    direction_error='轴向角误差abs((prediction-GT+90)%180-90)，预期可读但输出不可读记90deg',
    recovery='(CAPF-baseline)/(oracle-baseline)；error指标反向；分母0记未定义，不截断>1',
    safety='复用E22-O4/E28指标；配对case bootstrap95%CI，不能把不同seed当独立样本',
    significance='A4-A0 case paired bootstrap95%CI下界>0；不依据测试结果调权重/阈值')
