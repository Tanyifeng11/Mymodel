"""同一 B1 权重的空间干预；原 P0/P1 产物和决策保持原样。"""
import argparse
import csv
import gc
import re
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from tools.e44_protocol import OUT, E5, E5_SHA, SIZE, LAYER, read, write, sha, commit
from tools.e44_metrics import contour_f1, highpass_error, paired

DIAG = OUT / 'b1_mask_intervention_20261012'
CKPT = OUT / 'p1/B1/checkpoints/step0800.pt'
MODES = ['global', 'interior']
REGIONS = ['interior', 'boundary', 'background']
KEYS = ['lpips_gt', 'iou', 'contour_f1', 'leak', 'lab', 'hp_gray']
CRITICAL = ['172600892', '193488637', '180633937', '204446309', '140540361', '174397224']


def region_masks(raw):
    garment = np.asarray(raw) > 127
    kernel = np.ones((19, 19), np.uint8)
    inner = cv2.erode(garment.astype('uint8'), kernel) > 0
    outer = cv2.dilate(garment.astype('uint8'), kernel) > 0
    masks = dict(interior=inner, boundary=outer & ~inner, background=~outer)
    assert np.all(sum(m.astype('uint8') for m in masks.values()) == 1)
    return masks


def weights_at(masks, size, device):
    return {name: F.interpolate(torch.from_numpy(mask.astype('float32'))[None, None].to(device),
                               size=size, mode='area') for name, mask in masks.items()}


def regional_stats(value, weights):
    # 低分辨率单元可能跨区域；使用像素面积占比，不能把混合单元误报成纯背景。
    if value.ndim == 3:
        value = value.transpose(1, 2)
    else:
        value = value.flatten(2)
    value = value.detach().float()
    square = value.square().mean(1, keepdim=True)
    absolute = value.abs().mean(1, keepdim=True)
    total = float(square.sum())
    result = {}
    for name, weight in weights.items():
        w = weight.flatten(2)
        pixels = float(w.sum())
        energy = float((square * w).sum())
        result[name] = dict(effective_cells=pixels,
            rms=(energy / pixels) ** .5 if pixels else None,
            mae=float((absolute * w).sum()) / pixels if pixels else None,
            energy_fraction=energy / total if total else 0.)
    return result


def protocol():
    rows = read(OUT / 'splits/dev32.json')
    frozen = read(OUT / 'manifests/input_hashes.json')
    selected = {row[k]: frozen[row[k]] for row in rows for k in ['gt', 'sketch', 'reference']}
    selected[str(E5)] = E5_SHA
    selected[str(CKPT)] = read(CKPT.with_suffix('.json'))['sha256']
    assert all(sha(p) == h for p, h in selected.items()), '冻结输入或检查点发生变化'
    value = dict(experiment='E44_B1_mask_causal_intervention', training_updates=0,
        checkpoint=str(CKPT), checkpoint_sha256=selected[str(CKPT)], original_commit='d5e0fa5',
        split_sha256=sha(OUT / 'splits/dev32.json'), identity_n=32, modes=MODES,
        seed=42, steps=50, cfg=7., layer=LAYER, critical_ids=CRITICAL,
        intervention='same original9px erosion/8px feather mask; multiply B1 increment after original to_out',
        regions='Sketch only:19x19 erosion interior, dilation-minus-erosion boundary, outside dilation background',
        low_resolution_regions='area-fraction weights; additionally audit strictly zero mask cells',
        epsilon_effect='conditional prediction minus E5 OFF at SAME latent/timestep/conditions',
        causal_scope='inference localization at fixed weights; no retraining or parameter-count attribution',
        invalid='retain identities; branch OFF in both modes; regional metrics NA',
        bootstrap='identity paired,10000 replicates,seed32042', inputs=selected,
        original_p1_decision=read(OUT / 'decision.json'), source_commit=commit())
    path = DIAG / 'protocol.json'
    if path.exists():
        old = read(path)
        assert {k:v for k,v in old.items() if k!='source_commit'} == {
            k:v for k,v in value.items() if k!='source_commit'}, '协议已冻结，禁止覆盖后重新选择设置'
    else:
        write(path, value)
    return rows, value


class Diagnostic:
    def __init__(self, runtime, masks):
        self.r = runtime
        self.p = runtime.processor
        self.weights = weights_at(masks, (32, 24), runtime.pipe.device)
        self.latent_weights = weights_at(masks, (64, 48), runtime.pipe.device)
        self.steps = []
        self.latents = []
        self.current = None
        self.maps = []

    def injection(self, base, raw, applied):
        self.current['injection_raw'] = regional_stats(raw, self.weights)
        self.current['injection_applied'] = regional_stats(applied, self.weights)
        self.current['base_attention'] = regional_stats(base, self.weights)
        flat = applied.flatten(2).transpose(1, 2) if applied.ndim == 4 else applied
        zero = self.p.mask.flatten(2).squeeze(1) == 0
        self.current['strict_zero_cells'] = int(zero.sum())
        self.current['strict_zero_applied_max'] = float(flat[zero].abs().max()) if zero.any() else None
        if self.p.spatial_intervention == 'interior' and zero.any():
            assert self.current['strict_zero_applied_max'] == 0.
        if len(self.steps) in [0, 5, 15, 25, 49]:
            def rms_map(x):
                if x.ndim == 3:
                    x = x.transpose(1, 2).reshape(1, -1, 32, 24)
                return x.float().square().mean(1).sqrt().cpu()
            self.maps.append(dict(step=len(self.steps), raw=rms_map(raw), applied=rms_map(applied)))

    def install(self):
        self.old_forward = self.r.pipe.unet.forward
        self.p.diagnostic_observer = self.injection

        def forward(*args, **kwargs):
            conditional = 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
            if not conditional:
                assert not self.p.enabled
                return self.old_forward(*args, **kwargs)
            latent = args[0] if args else kwargs['sample']
            timestep = args[1] if len(args) > 1 else kwargs['timestep']
            self.current = dict(step=len(self.steps), timestep=int(timestep), enabled=self.p.enabled,
                                injection_raw=None, injection_applied=None)
            self.latents.append(latent.detach().cpu())
            output = self.old_forward(*args, **kwargs)
            enabled = self.p.enabled
            # 直接调用 forward，避免再次进入 Runtime 的 pre-hook 和采样记录 hook。
            # OFF 教师保持同一输入，绝不推进 scheduler，也不重新读取参考特征。
            try:
                self.p.enabled = False
                teacher = self.old_forward(*args, **kwargs) if enabled else output
            finally:
                self.p.enabled = enabled
            effect = output[0].float() - teacher[0].float()
            assert torch.isfinite(effect).all()
            self.current['matched_epsilon_effect'] = regional_stats(effect, self.latent_weights)
            self.current['matched_cfg_effect'] = regional_stats(effect * 7., self.latent_weights)
            self.steps.append(self.current)
            return output

        self.r.pipe.unet.forward = forward

    def close(self):
        self.r.pipe.unet.forward = self.old_forward
        self.p.diagnostic_observer = None


def generate():
    from tools import e44_runtime as runtime_module
    from tools.e22_4_generation import module_hashes
    rows, frozen = protocol()
    runtime_module.OUT = DIAG / 'runtime'
    runtime = runtime_module.Runtime('B1')
    runtime.load(CKPT)
    before = module_hashes({'processor': runtime.processor})
    try:
        for row in rows:
            sketch = Image.open(row['sketch']).convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)
            raw, soft, info = runtime_module.sketch_masks(sketch, 'cpu')
            masks = region_masks(raw)
            folder = DIAG / 'cases' / row['id']
            folder.mkdir(parents=True, exist_ok=True)
            for name, mask in masks.items():
                Image.fromarray(mask.astype('uint8') * 255).save(folder / (name + '.png'))
            write(folder / 'regions.json', dict(mask=info, pixels={k:int(v.sum()) for k,v in masks.items()}))
            mode_records = {}
            for mode in MODES:
                done = folder / mode / 'diagnostic.json'
                if done.exists():
                    saved = read(done)
                    assert sha(folder / mode / 'image.png') == saved['generation']['png_sha256']
                    mode_records[mode] = saved
                    continue
                runtime.processor.spatial_intervention = mode
                diagnostic = Diagnostic(runtime, masks)
                diagnostic.install()
                try:
                    record, _, _ = runtime.generate(row, folder / mode / 'image.png')
                finally:
                    diagnostic.close()
                assert len(diagnostic.steps) == 50
                old = read(OUT / 'p1/images/B1' / (row['id'] + '.json'))
                assert record['initial_sha256'] == old['initial_sha256']
                if mode == 'global':
                    assert record['png_sha256'] == old['png_sha256'], '全局组没有精确回放原 B1 RGB'
                    assert record['final_sha256'] == old['final_sha256'], '全局组没有精确回放原 B1 latent'
                traces = folder / mode / 'trajectory.pt'
                torch.save(dict(latents=torch.stack(diagnostic.latents), injection_maps=diagnostic.maps), traces)
                saved = dict(generation=record, steps=diagnostic.steps, gate=float(runtime.processor.gate),
                             trajectory_sha256=sha(traces), exact_original_b1_replay=mode == 'global')
                write(done, saved)
                mode_records[mode] = saved
                print('E44 MASK RGB', row['id'], mode, len(diagnostic.steps), flush=True)
            a = torch.load(folder / 'global/trajectory.pt', map_location='cpu', weights_only=False)['latents']
            b = torch.load(folder / 'interior/trajectory.pt', map_location='cpu', weights_only=False)['latents']
            weights = weights_at(masks, (64, 48), 'cpu')
            drift = [dict(step=i, regions=regional_stats(b[i].float()-a[i].float(), weights)) for i in range(50)]
            assert torch.equal(a[0], b[0])
            if not info['valid']:
                assert mode_records['global']['generation']['png_sha256'] == mode_records['interior']['generation']['png_sha256']
            write(folder / 'trajectory_drift.json', drift)
        assert before == module_hashes({'processor':runtime.processor})
        assert sha(CKPT) == frozen['checkpoint_sha256']
        write(DIAG / 'generation_complete.json', dict(identity_n=32, images=64, original_replay_n=32,
            trainable_unchanged=True, processor_hash=before, gate=float(runtime.processor.gate), commit=commit()))
    finally:
        runtime.close()


def overlay(image, garment, sketch):
    from tools.e44_metrics import external_contour
    arr = np.array(image.resize(SIZE).convert('RGB'))
    arr[external_contour(sketch)] = [0, 180, 0]
    arr[external_contour(garment)] = [255, 0, 0]
    return Image.fromarray(arr)


@torch.inference_mode()
def evaluate():
    from eval.eval_utils import prepare_evaluation_masks, estimate_foreground_mask, json_safe
    from eval.metrics import compute_texture_leakage, compute_texture_color_fidelity
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from tools.e44_runtime import sketch_masks
    rows, _ = protocol()
    assert read(DIAG / 'generation_complete.json')['images'] == 64
    metric, metric_info = frozen_lpips()
    metric = metric.cuda()
    records = [{k:v for k,v in row.items() if k in ['id','variant','mask_valid'] + KEYS}
               for row in read(OUT / 'p1/metrics.json') if row['variant'] == 'B0']
    comparisons = []
    old_b1 = {r['id']:r for r in read(OUT / 'p1/metrics.json') if r['variant'] == 'B1'}
    max_replay_metric_error = 0.
    pages = []
    for row in rows:
        gt = Image.open(row['gt']).convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)
        sketch = Image.open(row['sketch']).convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)
        raw, _, info = sketch_masks(sketch, 'cpu')
        regions = region_masks(raw)
        bundle = prepare_evaluation_masks(SIZE, sketch_path=row['sketch'], mask_policy='sketch_only')
        garment = bundle['garment']
        if garment is None:
            garment = np.zeros((SIZE[1],SIZE[0]),bool)
        folder = DIAG / 'cases' / row['id']
        images = {'B0':Image.open(OUT / 'p1/images/B0' / (row['id']+'.png')).convert('RGB')}
        foregrounds = {}
        for mode in MODES:
            path = folder / mode / 'image.png'
            image = Image.open(path).convert('RGB');images[mode] = image
            tensor = lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
            rec = dict(id=row['id'],variant=mode,mask_valid=info['valid'],
                       lpips_gt=float(metric(tensor(image),tensor(gt)).mean()),
                       **{k:None for k in KEYS if k != 'lpips_gt'})
            foreground = estimate_foreground_mask(image, SIZE)
            if foreground is None:
                foreground = np.zeros((SIZE[1],SIZE[0]),bool)
            foregrounds[mode] = foreground
            Image.fromarray(foreground.astype('uint8')*255).save(folder / mode / 'estimated_foreground.png')
            overlay(image, foreground, garment).save(folder / mode / 'contour_overlay.png')
            if info['valid']:
                rec['iou'] = float((foreground&garment).sum()/max((foreground|garment).sum(),1))
                rec['contour_f1'] = contour_f1(foreground, garment)
                rec['leak'] = json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle))['leak_colored_frac']
                rec['lab'] = json_safe(compute_texture_color_fidelity(str(path),row['reference'],mask_bundle=bundle))['tcf_lab_delta']
                inner = cv2.erode(garment.astype('uint8'),np.ones((19,19),np.uint8)) > 0
                if inner.any():rec['hp_gray'] = highpass_error(image,gt,inner)
            if mode == 'global':
                for k in KEYS:
                    if rec[k] is not None:
                        error = abs(rec[k]-old_b1[row['id']][k]);max_replay_metric_error=max(max_replay_metric_error,error)
                        assert error < 1e-5,(row['id'],k,error)
            records.append(rec)
        for a,b in [('interior','global'),('global','B0'),('interior','B0')]:
            diff = np.abs(np.asarray(images[a],np.float32)-np.asarray(images[b],np.float32))/255
            comparisons.append(dict(id=row['id'],comparison=a+'-'+b,mask_valid=info['valid'],
                **{name:float(diff[mask].mean()) if mask.any() and info['valid'] else None for name,mask in regions.items()}))
        difference = np.abs(np.asarray(images['interior'],np.float32)-np.asarray(images['global'],np.float32)).mean(2)/255
        heat = np.zeros((SIZE[1],SIZE[0],3),np.uint8)
        heat[:,:,0] = np.clip(difference/.1,0,1)*255
        heat[:,:,1] = np.clip(difference/.1-.5,0,.5)*510
        Image.fromarray(heat).save(folder / 'rgb_difference_fixed_scale.png')
        entries = [('Sketch',sketch),('Reference',Image.open(row['reference'])),('GT',gt),('E5',images['B0']),
                   ('B1 global',images['global']),('B1 interior',images['interior']),
                   ('global contour',Image.open(folder/'global/contour_overlay.png')),
                   ('interior contour',Image.open(folder/'interior/contour_overlay.png')),
                   ('RGB diff 0-0.1',Image.fromarray(heat))]
        panel = Image.new('RGB',(9*128,195),'white');draw=ImageDraw.Draw(panel)
        for j,(label,image) in enumerate(entries):
            draw.text((j*128+2,2),row['id'],fill='black');draw.text((j*128+2,14),label,fill='black')
            panel.paste(image.convert('RGB').resize((128,171)),(j*128,24))
        panel.save(folder/'panel.png');pages.append(panel)
        if row['id'] in CRITICAL:
            big = Image.new('RGB',(6*384,544),'white');draw=ImageDraw.Draw(big)
            for j,(label,image) in enumerate([entries[k] for k in [0,2,4,5,6,7]]):
                draw.text((j*384+2,2),row['id']+' '+label,fill='black');big.paste(image.resize(SIZE),(j*384,32))
            dest=DIAG/'critical_panels'/(row['id']+'.png');dest.parent.mkdir(exist_ok=True);big.save(dest)
        print('E44 MASK METRICS',row['id'],flush=True)
    for start in range(0,32,4):
        page=Image.new('RGB',(9*128,4*195),'white')
        for i,panel in enumerate(pages[start:start+4]):page.paste(panel,(0,i*195))
        dest=DIAG/'contact_sheets'/('page%02d.png'%(start//4));dest.parent.mkdir(exist_ok=True);page.save(dest)
    write(DIAG/'metrics.json',records);write(DIAG/'rgb_regional_changes.json',comparisons)
    with (DIAG/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    paired_results={a+'-'+b:{k:paired(records,a,b,k) for k in KEYS}
                    for a,b in [('global','B0'),('interior','global'),('interior','B0')]}
    write(DIAG/'paired_deltas.json',paired_results)
    write(DIAG/'evaluation_complete.json',dict(identity_n=32,regional_n=28,lpips=metric_info,
        max_global_replay_metric_error=max_replay_metric_error,original_p1_decision_unchanged=True,
        interpretation='fixed-weight inference causal intervention; estimated foreground proxy requires visual review'))
    print('E44 MASK EVALUATION COMPLETE',paired_results['interior-global'],flush=True)


def finish(jobs):
    assert read(DIAG/'evaluation_complete.json')['identity_n']==32
    accounting=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--noheader','--parsable2',
        '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
    (DIAG/'slurm_accounting.txt').write_text(accounting,encoding='utf-8')
    keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
    rows=[dict(zip(keys,line.split('|'))) for line in accounting.strip().splitlines()]
    assert len(rows)==len(jobs) and all(r['state']=='COMPLETED' and r['exit_code']=='0:0' for r in rows)
    gpu_seconds=sum(int(re.search(r'gres/gpu=(\d+)',r['resources']).group(1))*int(r['seconds'])
        for r in rows if re.search(r'gres/gpu=(\d+)',r['resources']))
    write(DIAG/'resource_ledger.json',dict(jobs=rows,gpu_hours=gpu_seconds/3600))
    protocol()
    sources=['tools/e44_b1_mask.py','models/reference_attention_variants.py','tools/e44_runtime.py']
    write(DIAG/'run_manifest.json',dict(commit=commit(),code={p:sha(p) for p in sources},
        files={str(p.relative_to(DIAG)):sha(p) for p in sorted(DIAG.rglob('*'))
               if p.is_file() and p.suffix not in ['.gz','.tmp','.log','.err'] and p.name!='run_manifest.json'}))
    archive=DIAG/'b1_mask_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(DIAG.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.png','.txt','.log','.err']:
                tar.add(p,arcname=str(p.relative_to(DIAG)))
    print('ARCHIVE',archive,'SHA256',sha(archive),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','finish'])
    parser.add_argument('--jobs',nargs='+');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.action=='finish':finish(args.jobs)
    else:
        generate();gc.collect();torch.cuda.empty_cache();evaluate()


if __name__=='__main__':main()
