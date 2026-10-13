"""e46 固定权重门控：训练域能量校准、完整回放、开发采样与真实资源归档。"""
import argparse
import platform
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from models.e46_sarg import boundary_band,gate_from_band,ResidualGate
from tools.e46_protocol import (OUT,OLD,E45,CKPT,CKPT_SHA,CONFIG,ARMS,HISTORY,SIZE,
    read,write,sha,commit,protocol,image_path,historical)


def runtime(namespace):
    from tools import e44_runtime as module
    module.OUT=OUT/namespace/'runtime'
    r=module.Runtime('B1');r.load(CKPT)
    r.processor.eval().requires_grad_(False)
    r.writer.unet.eval().requires_grad_(False)
    r.pipe.unet.eval().requires_grad_(False)
    return r


def frozen_processor(r):
    from tools.e22_4_generation import module_hashes
    return module_hashes({'b1_processor':r.processor})


def maps_for(row):
    from tools.e44_runtime import sketch_masks
    sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
    raw,_,info=sketch_masks(sketch,'cpu')
    maps=boundary_band(raw,CONFIG['sigma_rgb_px']);maps['info']['mask']=info
    folder=OUT/'boundary_maps'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    gate=gate_from_band(maps['small'])
    near=maps['distance']<=8
    far=maps['garment']&(maps['distance']>=48)
    rgb_gate=gate_from_band(maps['band'])
    maps['info'].update(gate_min=float(gate.min()),gate_max=float(gate.max()),
        near_mean=float(rgb_gate[near].mean()) if near.any() else None,
        far_interior_mean=float(rgb_gate[far].mean()) if far.any() else None,
        near_pixels=int(near.sum()),far_interior_pixels=int(far.sum()))
    if info['valid']:
        assert maps['info']['contour_count']>0
        if near.any() and far.any():assert maps['info']['near_mean']<maps['info']['far_interior_mean']
    if (folder/'geometry.json').exists():
        assert read(folder/'geometry.json')==maps['info']
        return maps
    entries=[('Sketch',sketch),('Raw mask',raw),('External contour',Image.fromarray(maps['edge']*255)),
        ('RGB band',Image.fromarray(np.round(maps['band']*255).astype('uint8'))),
        ('32x24 band',Image.fromarray(np.round(maps['small']*255).astype('uint8')).resize(SIZE,Image.Resampling.NEAREST)),
        ('Gate [0,1]',Image.fromarray(np.round(gate*255).astype('uint8')).resize(SIZE,Image.Resampling.NEAREST))]
    panel=Image.new('RGB',(192*6,281),'white');draw=ImageDraw.Draw(panel)
    for x,(label,img) in enumerate(entries):
        draw.text((x*192+2,2),row['id']+' '+label,fill='black')
        panel.paste(img.convert('RGB').resize((192,256)),(x*192,25))
    panel.save(folder/'geometry.png');raw.save(folder/'raw_mask.png')
    Image.fromarray(maps['edge']*255).save(folder/'external_contour.png')
    np.savez_compressed(folder/'maps.npz',band_rgb=maps['band'],band32=maps['small'],gate32=gate)
    write(folder/'geometry.json',maps['info'])
    return maps


def exact_trace(new,old):
    a=torch.load(new,map_location='cpu',weights_only=False)
    b=torch.load(old,map_location='cpu',weights_only=False)
    keys=['initial_latent','final_latent','conditional_epsilon','unconditional_epsilon']
    assert all(torch.equal(a[k],b[k]) for k in keys),'完整 DDIM50 张量回放失败'
    return dict(fields=keys,bit_exact=True,conditional_steps=len(a['conditional_epsilon']))


def run_case(r,row,mode,stage,constant=None,audit=False):
    path=image_path(mode,row['id'],stage)
    trace=path.with_name(path.stem+'_gate.json')
    if trace.exists():
        saved=read(trace)
        assert saved['mode']==mode and saved['constant']==constant
        assert sha(path)==saved['generation']['png_sha256']
        assert sha(path.with_suffix('.pt'))==saved['generation']['diagnostics_sha256']
        return saved
    maps=maps_for(row)
    ctl=ResidualGate(r,mode,maps,constant,layout_audit=audit)
    writer_before=r.writer.calls;ctl.install()
    try:rec,contexts,outputs=r.generate(row,path,capture=audit)
    finally:ctl.close()
    assert len(ctl.steps)==ctl.unconditional_calls==50
    assert all(s['enabled']==rec['mask']['valid'] for s in ctl.steps)
    assert r.writer.calls-writer_before==(50 if rec['mask']['valid'] else 0)
    result=dict(mode=mode,constant=constant,generation=rec,steps=ctl.steps,layout=ctl.layout,
        unconditional_calls=ctl.unconditional_calls,writer_calls=r.writer.calls-writer_before,
        training_updates=0,commit=commit())
    if stage!='calibration':
        old=read(historical('G1',row['id']).with_suffix('.json'))
        assert rec['initial_sha256']==old['initial_sha256']
    source=None
    if mode=='identity':source=historical('G1',row['id'])
    if mode=='zero' or (stage!='calibration' and not rec['mask']['valid']):source=historical('G0',row['id'])
    if source:
        old=read(source.with_suffix('.json'))
        assert rec['png_sha256']==old['png_sha256'] and rec['final_sha256']==old['final_sha256']
        result['exact_replay']=exact_trace(path.with_suffix('.pt'),source.with_suffix('.pt'))
    if audit:
        assert ctl.layout and all(v['exact_projection_to_tokens'] for v in ctl.layout)
        # 真实同输入：绕过 Runtime 的 U-Net outer hook，原无条件算子独立运行一次。
        args,kwargs=contexts['unconditional'];r.processor.enabled=False
        with torch.inference_mode():plain=r.pipe.unet.forward(*args,**kwargs)[0]
        assert torch.equal(plain,outputs['unconditional']),'SARG 污染 CFG 无条件路径'
        from tools.e35_freeu_run import digest
        result['same_input_cfg_audit']=dict(bit_exact=True,input_latent_sha256=digest(args[0]),
            timestep=int(args[1]),cross_attention_keys=sorted(kwargs['cross_attention_kwargs']),
            original_processor_used=True)
    write(trace,result)
    print('E46 RGB',stage,mode,row['id'],round(rec['seconds'],2),flush=True)
    return result


def calibrate(r):
    destination=OUT/'calibration_energy.json'
    if destination.exists():
        value=read(destination)
        assert value['protocol_sha256']==sha(OUT/'protocol.json')
        return value
    records=[run_case(r,row,'calibration','calibration') for row in read(OUT/'splits/calibration8.json')]
    steps=[dict(id=v['generation']['id'],**s) for v in records for s in v['steps']]
    assert len(steps)==400 and all(s['enabled'] for s in steps)
    raw=sum(s['raw_energy'] for s in steps);sarg=sum(s['hypothetical_sarg_energy'] for s in steps)
    coefficient=(sarg/raw)**.5
    half=float(torch.tensor(coefficient,dtype=torch.float16))
    error=abs(half**2*raw-sarg)/sarg
    assert .26<=coefficient<=.65 and error<CONFIG['energy_tolerance']
    value=dict(lambda_E=coefficient,fp16_lambda=half,raw_energy=raw,sarg_ideal_energy=sarg,
        fp16_constant_energy_relative_error=error,steps=steps,identities=8,
        ids=[v['generation']['id'] for v in records],trajectory='original ungated B1',
        matching_scope='cumulative squared residual energy on calibration B1 trajectory; free trajectories not guaranteed equal',
        protocol_sha256=sha(OUT/'protocol.json'),commit=commit(),dev_metrics_used=False)
    write(destination,value)
    write(OUT/'protocol_calibrated.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        calibration_sha256=sha(destination),lambda_E=coefficient,frozen_before_dev=True,config=CONFIG))
    print('E46 CALIBRATION lambda_E',coefficient,flush=True)
    return value


def audit():
    rows,frozen=protocol();smoke=[next(r for r in rows if r['id']==x) for x in frozen['smoke_ids']]
    import diffusers
    write(OUT/'manifests/environment.json',dict(commit=commit(),python=platform.python_version(),
        torch=torch.__version__,diffusers=diffusers.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),protocol_sha256=sha(OUT/'protocol.json')))
    r=runtime('p0');before=frozen_processor(r)
    try:
        calibration=calibrate(r)
        for row in rows:maps_for(row)
        for row in smoke:
            run_case(r,row,'identity','p0/replay',audit=True)
            run_case(r,row,'zero','p0/replay',audit=True)
        for row in smoke[:2]:
            run_case(r,row,'G2','p0/constant',constant=.65)
            run_case(r,row,'beta0','p0/constant')
            a=image_path('G2',row['id'],'p0/constant');b=image_path('beta0',row['id'],'p0/constant')
            assert sha(a)==sha(b);exact_trace(a.with_suffix('.pt'),b.with_suffix('.pt'))
        for row in smoke:
            result=run_case(r,row,'G4','p0/smoke',audit=True)
            if result['generation']['mask']['valid']:
                assert result['generation']['final_sha256']!=read(historical('G1',row['id']).with_suffix('.json'))['final_sha256']
        assert before==frozen_processor(r) and sha(CKPT)==CKPT_SHA
    finally:r.close()
    write(OUT/'p0/audit.json',dict(engineering_pass=True,identity_exact_n=8,zero_exact_n=8,
        beta0_constant_exact_n=2,sarg_full_ddim_smoke_n=8,layout_real_projection_exact=True,
        same_input_unconditional_exact=True,training_updates=0,frozen_b1=before,
        lambda_E=calibration['lambda_E'],protocol_sha256=sha(OUT/'protocol.json'),commit=commit()))


def generate(arm):
    rows,_=protocol();assert read(OUT/'p0/audit.json')['engineering_pass']
    calibrated=read(OUT/'protocol_calibrated.json')
    assert calibrated['calibration_sha256']==sha(OUT/'calibration_energy.json')
    constant=.65 if arm=='G2' else calibrated['lambda_E'] if arm=='G3' else None
    r=runtime('p1/'+arm);before=frozen_processor(r)
    try:
        records=[run_case(r,row,arm,'p1',constant=constant) for row in rows]
        assert before==frozen_processor(r) and sha(CKPT)==CKPT_SHA
    finally:r.close()
    write(OUT/'p1'/(arm+'_complete.json'),dict(arm=arm,n=len(records),
        ids=[v['generation']['id'] for v in records],constant=constant,
        frozen_b1=before,calibrated_protocol_sha256=sha(OUT/'protocol_calibrated.json'),commit=commit()))


def import_history(rows):
    audits=[]
    source_manifests={root:read(root/'run_manifest.json') for root in [OLD,E45]}
    for arm in HISTORY:
        root=OLD if arm in ['G0','G1'] else E45
        for row in rows:
            src=historical(arm,row['id']);dst=image_path(arm,row['id'])
            meta=read(src.with_suffix('.json'))
            assert sha(src)==meta['png_sha256']
            relative=str(src.relative_to(root))
            assert sha(src)==source_manifests[root]['files'][relative]
            assert sha(src.with_suffix('.json'))==source_manifests[root]['files'][str(src.with_suffix('.json').relative_to(root))]
            baseline=read(historical('G1',row['id']).with_suffix('.json'))
            assert meta['initial_sha256']==baseline['initial_sha256']
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst);shutil.copy2(src.with_suffix('.json'),dst.with_suffix('.json'))
            audits.append(dict(arm=arm,id=row['id'],source=str(src),source_sha256=sha(src),
                imported_sha256=sha(dst),source_manifest_sha256=sha(root/'run_manifest.json')))
    write(OUT/'manifests/history_sources.json',audits)


def finish(jobs):
    protocol();assert read(OUT/'p1/evaluation_complete.json')['complete']
    accounting=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--noheader','--parsable2',
        '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
    (OUT/'slurm_accounting.txt').write_text(accounting,encoding='utf-8')
    keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
    rows=[dict(zip(keys,line.split('|'))) for line in accounting.strip().splitlines()]
    assert set(jobs)=={r['job_id'] for r in rows}
    assert all(r['state'] not in ['RUNNING','PENDING','COMPLETING'] for r in rows)
    gpu_seconds=0
    for row in rows:
        count=re.search(r'gres/gpu=(\d+)',row['resources'])
        gpu_seconds+=(int(count.group(1)) if count else 0)*int(row['seconds'])
    write(OUT/'resource_ledger.json',dict(jobs=rows,gpu_hours=gpu_seconds/3600,
        failed_jobs_retained=True,training_updates=0))
    for name in ['input_hashes.json','model_file_hashes.json','source_split_hashes.json']:
        assert all(sha(p)==h for p,h in read(OUT/'manifests'/name).items())
    code=list(Path('tools').glob('e46*.py'))+[Path('models/e46_sarg.py'),Path('submit/e46_gpu.sh'),Path('tests/test_e46_sarg.py')]
    code += [Path(p) for p in ['tools/e44_runtime.py','tools/e44_b1_mask.py','tools/e44_metrics.py',
        'tools/e44_protocol.py','tools/e45_eval.py','tools/e35_freeu_protocol.py','models/reference_attention_variants.py',
        'models/reference_feature_writer.py','adapter/attention_processor.py','garment_mask_utils.py',
        'eval/eval_utils.py','eval/metrics.py','pipelines/IMAGGarment_pipeline.py']]
    extensions=['.json','.csv','.png','.txt','.npz','.log','.err']
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.suffix in extensions and p.name!='run_manifest.json'}
    large={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.pt'))}
    write(OUT/'run_manifest.json',dict(commit=commit(),protocol_sha256=sha(OUT/'protocol.json'),
        calibrated_protocol_sha256=sha(OUT/'protocol_calibrated.json'),code={str(p):sha(p) for p in code},
        files=files,server_retained_tensor_files=large,decision=read(OUT/'decision.json')))
    archive=OUT/'e46_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in extensions:tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',archive,'SHA256',sha(archive),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit','generate','evaluate','finish'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--jobs',nargs='+')
    args=parser.parse_args()
    if args.action=='prepare':protocol()
    elif args.action=='audit':audit()
    elif args.action=='generate':generate(args.arm)
    elif args.action=='evaluate':
        from tools.e46_eval import evaluate
        evaluate()
    else:finish(args.jobs)
