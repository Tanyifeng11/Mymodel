"""e49 预先冻结的两臂编码实验；历史失败案例仅作开发诊断。"""
import shutil
from pathlib import Path
from tools.e47_protocol import E5, E5_SHA, B1, B1_SHA, SIZE, LAYER, read, write, sha, commit

OUT = Path('output_eval/e49_ssvm_20261014')
SOURCE = Path('output_eval/e48_b1_structure_20261014')
TRAIN_ARMS = ['CLIP', 'SSVM']
ARMS = ['B', 'D'] + TRAIN_ARMS
CONFIG = dict(experiment='e49', version=1, seed=42, steps=50, cfg=7., size=list(SIZE),
    base_sha256=E5_SHA, original_b1_sha256=B1_SHA, layer=LAYER,
    scales=dict(sketch=.6, texture=1., ipa=1.),
    encoder=dict(backbone='frozen existing E5 CLIP vision weights', hidden_layer='last',
        CLIP='original CLS and original positional embeddings',
        SSVM='no CLS; random permutation of patch positional embeddings BEFORE frozen Transformer',
        references=3, crops='full plus two 0.8-height/width crops; independent RNG; same crops in both arms',
        augmentation='no rotation or flip, to preserve cloth pattern direction',
        preprocessing='bicubic square resize to CLIP image_size and standard CLIP normalization',
        training_seed=49042, inference_seed='SHA256 of reference RGB; independent of sample order'),
    style_embed=dict(type='existing PerceiverResampler', width=640, depth=1, heads=10,
        dim_head=64, num_latents=16, ff_mult=2, trainable=True),
    injection='independent reference softmax; original E5 attention preserved; global zero scalar gate',
    training=dict(arms=TRAIN_ARMS, updates=800, seed=42, microbatch=1, accumulation=8,
        lr=1e-4, weight_decay=.01, keep=.1, initialization='identical fresh seed42, zero gate',
        checkpoint_selection='fixed step800 only', teacher='frozen E5 branch OFF',
        expected_sequence_sha256='05169fe88722fb990ac7bab25caeb7260a66ede7a1e6e61ea32da508cede89f5'),
    mask_invalid='retain; new branch OFF; regional metrics NA',
    bootstrap=dict(draws=10000, seed=32042, unit='garment identity'),
    safety=dict(f1_lower=-.020, iou_lower=-.010, leak_upper=.005, tail=-.15),
    appearance=dict(lpips_ci_upper=0., minimum_gain_retention=.50),
    selection='pass all frozen gates on probe16 and dev32; lowest dev32 LPIPS, name breaks tie; one new96',
    encoder_comparison='paired SSVM versus CLIP CI; exploratory mechanism evidence, not an additional tuning criterion',
    motif_review='full image boards and worst10; direction, repeat scale, large motif; AI review is not human blind evaluation',
    limitations='small 800-update adaptation, not full StyleAdapter reproduction; development splits overlap; no guaranteed success',
    references=dict(StyleAdapter='https://arxiv.org/html/2309.01770v2',
        Refacade='https://arxiv.org/html/2512.04534v1'))


def decision(**changes):
    path = OUT/'decision.json'
    value = read(path) if path.exists() else dict(experiment='e49', audit='not_run', training='not_run',
        probe='not_run', dev32='not_run', new96='not_run', unseen_generalization=False)
    value.update(changes); write(path, value); return value


def image_path(arm, identity, stage='probe'):
    return OUT/stage/'images'/arm/(identity+'.png')


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): assert sha(target) == sha(source), str(target)
    else: shutil.copy2(source, target)


def prepare():
    assert read(SOURCE/'decision.json')['final_decision'] == 'NO_GO'
    assert sha(E5) == E5_SHA and sha(B1) == B1_SHA
    path = OUT/'protocol.json'
    if path.exists(): assert read(path) == CONFIG
    else: write(path, CONFIG)
    names = ['splits/probe.json', 'splits/dev32.json', 'splits/train1024.json',
        'splits/train1024_records.json', 'manifests/e5_training_meta.json', 'manifests/input_hashes.json']
    for name in names: copy_exact(SOURCE/name, OUT/name)
    for name in ['protocol.json', 'mechanism_summary.json', 'decision.json', 'probe/gates.json', 'dev32/gates.json']:
        copy_exact(SOURCE/name, OUT/'source_diagnosis'/name)
    assert all(sha(p)==h for p,h in read(OUT/'manifests/input_hashes.json').items())
    # B/D 复用同种子原图并记录来源，不花 GPU 重复生成既有对照。
    for stage in ['probe', 'dev32']:
        rows = read(OUT/'splits'/(stage+'.json'))
        for arm in ['B', 'D']:
            for row in rows:
                source = SOURCE/stage/'images'/arm/(row['id']+'.png')
                assert sha(source)==read(source.with_suffix('.json'))['png_sha256']
                for suffix in ['.png', '.json']: copy_exact(source.with_suffix(suffix), image_path(arm,row['id'],stage).with_suffix(suffix))
            write(OUT/stage/(arm+'_complete.json'), dict(n=len(rows), reused_from=str(SOURCE), commit=commit()))
    decision(prepared=True, source_diagnosis_sha256=sha(OUT/'source_diagnosis/mechanism_summary.json'))
