"""确认后冻结的两臂800更新实验；不按开发结果反调条件和门槛。"""
from pathlib import Path
from tools.e49_protocol import copy_exact
from tools.e47_protocol import E5,E5_SHA,B1,B1_SHA,SAFE_SCALE,SIZE,LAYER,read,write,sha,commit

OUT=Path('output_eval/e51_residual_controller_20261015')
SOURCE=Path('output_eval/e48_b1_structure_20261014')
TRAIN_ARMS=['NOTIME','TIME']
ARMS=['B','D','SAFE']+TRAIN_ARMS
CONFIG=dict(experiment='e51',version=1,base_sha256=E5_SHA,b1_sha256=B1_SHA,layer=LAYER,
    seed=42,steps=50,cfg=7.,size=list(SIZE),scales=dict(sketch=.6,texture=1.,ipa=1.),
    arms=ARMS,known_bad=['172600892','180633937','193488637','204446309','150553679','79583392','163690164'],
    controller=dict(width=640,grid=[32,24],bottleneck=32,inputs=['target','frozen projected gated B1 increment','Sketch foreground/boundary/distance'],
        reference='same frozen B1 Writer/K/V/norm/gate; joint attention mathematics retained',
        time='64-dimensional sinusoidal embedding of real diffusion t; FiLM before output and sigmoid gate',
        NOTIME='same network and initial weights; entire time embedding zeroed; can learn a constant via biases',
        TIME='real diffusion timestep, not inference step index',
        initialization='only residual output and gate logits heads zero; sigmoid gate0.5; initial E5 exact',
        formula='E5 + lambda * original Sketch soft mask * sigmoid_gate * zero_output_controller',
        residual_scale=1.,cap='per-example feature RMS <= same-state original B1 projected residual RMS; shrink only',
        boundary='19x19 Sketch dilation minus erosion',distance='inside distance /32 clipped to [0,1]',
        soft_mask='existing 9px erosion plus8px feather; invalid OFF; no GT',
        training_precision='FP32 controller, frozen backbone historical FP16; GradScaler',
        limitation='feature mask/RMS bound does not guarantee downstream epsilon or RGB geometry'),
    training=dict(updates=800,microbatch=1,accumulation=8,seed=42,lr=1e-4,weight_decay=.01,
        gradient_clip=1.,outside_keep=.1,boundary_keep=.1,teacher='same-state frozen E5',
        expected_sequence_sha256='05169fe88722fb990ac7bab25caeb7260a66ede7a1e6e61ea32da508cede89f5',
        checkpoint_selection='fixed final800; two scientific runs, failures preserved separately'),
    static_control=dict(arm='SAFE',scale=SAFE_SCALE,calibration='reuse E47 training8 calibrated constant; no new dev calibration'),
    statistics=dict(unit='identity',bootstrap_draws=10000,seed=32042),
    safety=dict(f1_lower=-.020,iou_lower=-.010,leak_upper=.005,tail=-.15),
    appearance=dict(lpips_ci_upper=0.,minimum_gain_retention=.50),
    scope='probe16 and existing dev32 (13 overlap); new96 not_run, no automatic expansion',
    references=dict(IPAdapter='https://arxiv.org/html/2308.06721v1#S3',ControlNet='https://arxiv.org/html/2302.05543v3#S3',
        TASR='https://arxiv.org/html/2412.03355v1#S3'),
    interpretation='lightweight adaptation, not full ControlNet/TASR reproduction; AI review not human blind evaluation')


def decision(**updates):
    path=OUT/'decision.json'
    value=read(path) if path.exists() else dict(experiment='e51',audit='not_run',training='not_run',probe='not_run',
        dev32='not_run',new96='not_run',unseen_generalization=False)
    value.update(updates);write(path,value);return value


def image_path(arm,identity,stage='probe'):return OUT/stage/'images'/arm/(identity+'.png')


def prepare():
    assert read(SOURCE/'decision.json')['final_decision']=='NO_GO'
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==CONFIG,'协议已冻结'
    else:write(path,CONFIG)
    for name in ['splits/probe.json','splits/dev32.json','splits/train1024.json','splits/train1024_records.json',
            'manifests/e5_training_meta.json','manifests/input_hashes.json']:
        copy_exact(SOURCE/name,OUT/name)
    for name in ['protocol.json','decision.json','mechanism_summary.json','probe/gates.json','dev32/gates.json']:
        copy_exact(SOURCE/name,OUT/'source_diagnosis'/name)
    assert all(sha(p)==h for p,h in read(OUT/'manifests/input_hashes.json').items())
    for stage in ['probe','dev32']:
        rows=read(OUT/'splits'/(stage+'.json'))
        for arm in ['B','D','SAFE']:
            for row in rows:
                source=SOURCE/stage/'images'/arm/(row['id']+'.png')
                assert sha(source)==read(source.with_suffix('.json'))['png_sha256']
                for suffix in ['.png','.json']:copy_exact(source.with_suffix(suffix),image_path(arm,row['id'],stage).with_suffix(suffix))
            write(OUT/stage/(arm+'_complete.json'),dict(n=len(rows),reused_from=str(SOURCE),commit=commit()))
    prereg=OUT/'manifests/pre_registration.json'
    if not prereg.exists():
        from datetime import datetime,timezone
        write(prereg,dict(commit=commit(),utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(path),before_new_training=True))
    decision(prepared=True)


if __name__=='__main__':prepare()
