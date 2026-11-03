"""冻结192条件的CFG×Lab免训练诊断；不改变模型权重或筛选测试身份。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
import tarfile

import cv2
import numpy as np
from PIL import Image
from tools.bf_t3_ablation import core, ROOT as PARENT
from tools import t3_color_metrics as color

ROOT = core.PROJECT / 'output_eval/t3_cfg5_lab_20261103'
SOURCE = Path(__file__).resolve().parents[1]
ARMS = ['CFG7', 'CFG5', 'CFG7_LAB', 'CFG5_LAB']
CONDITIONS = ['original', 'neutral_original', 'neutral_swapped']
CHECKPOINT = core.PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101/training/BF-SD-T3/step28365.pt'
CHECKPOINT_SHA = '06b9994862ec2cb0be5254c4d30acec2f459ec359e9c53856bc5413255f3fa74'
read, write, sha = core.read, core.write, core.sha


def output(arm, row):
    return ROOT / 'outputs' / arm / row['condition'] / (row['id'] + '.png')


def image(path):
    return np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))


def softmask(path):
    from garment_mask_utils import build_sketch_garment_mask
    raw, info = build_sketch_garment_mask(Image.open(path).convert('RGB'), 384, 512)
    distance = cv2.distanceTransform((np.asarray(raw) > 127).astype('uint8'), cv2.DIST_L2, 5)
    soft = np.clip((distance - 9) / 8, 0, 1).astype('float32')
    valid = bool(info['mask_source'] == 'sketch_flood_fill' and not info['mask_low_confidence'] and soft.mean() >= .03)
    if not valid: soft.fill(0)
    assert not valid or (soft >= .99).sum() >= 50
    return raw, soft, dict(info, valid=valid, soft_coverage=float(soft.mean()), gt_fallback=False)


def prepare():
    assert not (ROOT / 'protocol.json').exists(), '不覆盖冻结协议'
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    rows = read(PARENT / 'evaluation.json')
    assert [sum(r['condition'] == c for r in rows) for c in CONDITIONS] == [128, 32, 32]
    mask_audit = {}
    for row in rows:
        for key in ['sketch', 'reference', 'gt']:
            source = Path(row[key]); expected = row[key + '_sha256']
            assert sha(source) == expected
            dest = ROOT / 'inputs' / expected
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists(): shutil.copy2(source, dest)
            row[key] = str(dest)
        source = PARENT / 'outputs/BF-SD-T3' / row['condition'] / (row['id'] + '.png')
        meta = read(source.with_suffix('.json'))
        assert meta['output_sha256'] == sha(source) and meta['checkpoint_sha256'] == CHECKPOINT_SHA
        assert meta['seed'] == row['seed'] and meta['cfg'] == 7 and meta['steps'] == 50
        for key in ['sketch', 'reference']: assert meta['input_' + key + '_sha256'] == row[key + '_sha256']
        assert not meta['GT_used_in_inference'] and meta['text_used']
        if 'caption' in meta: assert meta['caption'] == row['caption']
        dest = output('CFG7', row); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        write(dest.with_suffix('.json'), dict(meta, arm='CFG7', caption=row['caption'], reused_from=str(source),
            caption_lineage='frozen parent evaluation manifest; original128 legacy metadata lacked caption'))
        if row['id'] not in mask_audit:
            raw, soft, info = softmask(row['sketch'])
            folder = ROOT / 'masks'; folder.mkdir(exist_ok=True)
            Image.fromarray(np.asarray(raw).astype('uint8')).save(folder / (row['id'] + '.png'))
            np.save(folder / (row['id'] + '.npy'), soft)
            mask_audit[row['id']] = info
    write(ROOT / 'evaluation.json', rows)
    write(ROOT / 'audit/masks.json', mask_audit)
    write(ROOT / 'protocol.json', dict(name=ROOT.name, arms=ARMS, checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        parent_manifest_sha256=sha(PARENT / 'evaluation.json'), evaluation_sha256=sha(ROOT / 'evaluation.json'),
        training=False, gpu_seconds_limit=3600, reserved_gpu_seconds=dict(generate=2400, neural=1020, enforcement_margin=180),
        inference=dict(width=384, height=512, steps=50, sampler='DDIM', sketch_scale=.6, ipa_scale=0,
            cfg=[7., 5.], per_identity_frozen_latent=True, texture_tokens=False),
        lab=dict(axes='a,b', strength=1., gain_clip=[.5, 2.], L_held_before_RGB=True,
            erosion=9, feather=8, reference='full image', invalid='exact OFF', outside='exact source pixels'),
        statistics=dict(bootstrap=10000, sign_flip=10000, seed=110142,
            comparisons=[['CFG5','CFG7'],['CFG7_LAB','CFG7'],['CFG5_LAB','CFG5'],['CFG5_LAB','CFG7']],
            primary='each condition independently: two reference color distances x four contrasts, Holm8',
            main_contrast='CFG5_LAB minus CFG5; primary condition original128; neutral32 sensitivity'),
        exploratory_rules=dict(color='both reference RGBuv H and ab W1 delta CI upper <0',
            edge_f1_ci_lower=-.02, iou_ci_lower=-.01, leak_ci_upper=.005, worst_identity_f1_delta=-.15),
        limitations=['one trained checkpoint and fixed seed per identity', 'previously viewed identities; exploratory',
            'reference donor reuse', '64x64 patch does not represent a full motif',
            'Lab modifies output pixels; no new learned reference ability', 'RGB gamut clipping can change final L'],
        color_code_sha256=sha(SOURCE / 'tools/t3_color_metrics.py'),
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip()))
    print('PREPARED', len(rows), 'MASK_VALID', sum(v['valid'] for v in mask_audit.values()), flush=True)


def generate():
    import torch
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    pipe, network = core.load_network('cuda', CHECKPOINT)
    network.eval().requires_grad_(False)
    pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    pipe.image_encoder.to('cuda'); pipe.bf_texture_conditioner.to('cuda')
    assert pipe._execution_device == next(pipe.text_encoder.parameters()).device == next(pipe.unet.parameters()).device
    rows = read(ROOT / 'evaluation.json')
    replay = []
    with torch.inference_mode():
        # 预先固定三个条件各一个输入，核对基线重放；不按重放结果换样。
        selected = [sorted([r for r in rows if r['condition'] == c], key=lambda r:r['id'])[0] for c in CONDITIONS]
        tasks = [(r, 7., True) for r in selected] + [(r, 5., False) for r in rows]
        for row, cfg, checking in tasks:
            dest = ROOT / 'audit/replay' / row['condition'] / (row['id'] + '.png') if checking else output('CFG5', row)
            base_meta = read(output('CFG7', row).with_suffix('.json'))
            if dest.exists() and not checking:
                meta = read(dest.with_suffix('.json')); assert sha(dest) == meta['output_sha256']
                assert meta['initial_latent_sha256'] == base_meta['initial_latent_sha256'] and meta['cfg'] == 5
                continue
            sketch = core.tensor(row['sketch']).cuda().half(); reference = core.tensor(row['reference']).cuda().half()
            network.adapter.condition(sketch, reference)
            latent = torch.randn((1,4,64,48), device='cuda', dtype=torch.float16,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            before = core.digest(latent); assert before == base_meta['initial_latent_sha256']
            started = time.monotonic()
            result = pipe(ref_image=sketch, prompt=row['caption'], texture_clip_image=None, texture_embeds=None,
                null_prompt='', negative_prompt=' worst quality, low quality', width=384, height=512,
                num_images_per_prompt=1, guidance_scale=cfg, sketch_scale=.6, ipa_scale=0.,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']), latents=latent, num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True, exist_ok=True); result.save(dest); network.adapter.clear()
            if checking:
                a, b = image(dest), image(output('CFG7', row))
                replay.append(dict(id=row['id'], condition=row['condition'], exact=bool(np.array_equal(a,b)),
                    max_pixel_delta=int(np.abs(a.astype(int)-b.astype(int)).max()),
                    mean_pixel_delta=float(np.abs(a.astype(float)-b).mean()), latent_exact=True))
                write(ROOT / 'audit/replay.json', replay)
                assert replay[-1]['exact'], 'CFG7重放不一致，停止；保留证据，不替换基线'
            else:
                write(dest.with_suffix('.json'), dict(base_meta, arm='CFG5', cfg=5., reused_from=None,
                    output_sha256=sha(dest), seconds=time.monotonic()-started))
            print('GENERATED', cfg, row['condition'], row['id'], flush=True)
    write(ROOT / 'generation_complete.json', dict(new_cfg5=192, cfg7_replay=3, passed=True))


def process():
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_leakage, compute_texture_color_fidelity
    rows = read(ROOT / 'evaluation.json'); masks_info = read(ROOT / 'audit/masks.json')
    records, reference_cache = [], {}
    for row in rows:
        reference = np.asarray(Image.open(row['reference']).convert('RGB'))
        if row['reference_sha256'] not in reference_cache:
            reference_cache[row['reference_sha256']] = color.descriptor(reference)
        reference_desc = reference_cache[row['reference_sha256']]
        soft = np.load(ROOT / 'masks' / (row['id'] + '.npy')); valid = masks_info[row['id']]['valid']
        for source_arm in ARMS[:2]:
            source = output(source_arm, row); rgb = image(source)
            edited = color.calibrate(rgb, reference, soft, valid)
            assert np.array_equal(edited[soft == 0], rgb[soft == 0])
            if not valid: assert np.array_equal(edited, rgb)
            dest = output(source_arm + '_LAB', row); dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(edited).save(dest)
            write(dest.with_suffix('.json'), dict(read(source.with_suffix('.json')), arm=source_arm+'_LAB',
                output_sha256=sha(dest), calibrated_from_sha256=sha(source), lab_valid=valid, outside_exact=True))
        bundle = prepare_evaluation_masks((384,512), sketch_path=row['sketch'], mask_policy='sketch_only')
        inside, outside = color.regions(bundle['garment'])
        for arm in ARMS:
            path = output(arm, row); rgb = image(path)
            desc = color.descriptor(rgb, inside)
            kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=bundle)
            structure = compute_structure_preservation(**kwargs)
            record = dict(id=row['id'], category=row['category'], condition=row['condition'], arm=arm,
                mask_valid=valid, color_pixels=int(inside.sum()), edge_f1=structure.get('edge_f1'),
                iou=structure.get('sketch_iou'), edge_l1=structure.get('edge_l1'),
                leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'),
                **color.compare(desc, reference_desc, 'ref_'), **color.outside_metrics(rgb, outside))
            record['tcf_lab'] = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta')
            records.append(record)
        print('PROCESSED', row['condition'], row['id'], flush=True)
    write(ROOT / 'metrics/cpu_per_image.json', records)


def neural():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    from tools.bf_t3_three_arm_report import gray
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    response.ROOT=ROOT
    dists=response.load_dists('cuda'); lpips, assets=frozen_lpips()
    write(ROOT/'audit/lpips_assets.json', assets)
    rows=read(ROOT/'evaluation.json'); records=read(ROOT/'metrics/cpu_per_image.json')
    lookup={(r['arm'],r['condition'],r['id']):r for r in records}
    old={r['id']:r for r in rows if r['condition']=='neutral_original'}
    cache={}
    def features(parts):
        return dists.forward_once(torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().cuda()/255)
    def ref(path,is_gray):
        key=(path,is_gray)
        if key not in cache:
            rgb=image(path)
            parts=[rgb[y:y+64,x:x+64] for y in np.linspace(0,448,4).round().astype(int) for x in np.linspace(0,320,4).round().astype(int)]
            cache[key]=features([gray(p) for p in parts] if is_gray else parts)
        return cache[key]
    def distance(a,b):
        matrix=response.distances(dists,a,b)
        return float(.5*(matrix.min(0).values.mean()+matrix.min(1).values.mean()))
    with torch.inference_mode():
        for row in rows:
            bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
            coords,candidates=patches(bundle['garment'])
            latents=[]
            for arm in ARMS:
                path=output(arm,row); meta=read(path.with_suffix('.json'))
                assert sha(path)==meta['output_sha256'] and meta['caption']==row['caption'] and not meta['GT_used_in_inference']
                assert meta['checkpoint_sha256']==CHECKPOINT_SHA and meta['seed']==row['seed']
                for key in ['sketch','reference']: assert meta['input_'+key+'_sha256']==row[key+'_sha256']
                latents.append(meta['initial_latent_sha256'])
                record=lookup[arm,row['condition'],row['id']]
                record.update(lpips=None,gray_dists=None,gray_margin=None,dists_margin=None,patch_n=len(coords))
                if row['condition']=='original':record['lpips']=float(lpips(core.tensor(path).cuda(),core.tensor(row['gt']).cuda()).mean())
                if row['condition']!='original' and len(coords)>=4:
                    rgb=image(path); parts=[rgb[y:y+64,x:x+64] for x,y in coords]
                    for is_gray in [False,True]:
                        generated=features([gray(p) for p in parts] if is_gray else parts)
                        new_distance=distance(generated,ref(row['reference'],is_gray))
                        if is_gray:record['gray_dists']=new_distance
                        if row['condition']=='neutral_swapped':
                            record['gray_margin' if is_gray else 'dists_margin']=distance(generated,ref(old[row['id']]['reference'],is_gray))-new_distance
            assert len(set(latents))==1
            print('NEURAL',row['condition'],row['id'],flush=True)
    assert len(records)==768
    write(ROOT/'metrics/per_image.json',records)
    summarize(ROOT)
    write(ROOT/'evaluation_complete.json',dict(complete=True,outputs=768,independent_confirmation=False))


def summarize(root):
    from tools.bf_t3_three_arm_report import paired
    from tools.bf_t3_ablation_report import holm,table
    records=read(root/'metrics/per_image.json'); protocol=read(root/'protocol.json')
    metrics=[k for k,v in records[0].items() if isinstance(v,(int,float)) and k not in ['mask_valid','color_pixels','patch_n']]
    metrics+=['lpips','gray_dists','gray_margin','dists_margin']
    metrics=list(dict.fromkeys(metrics))
    summary=[]; stats=[]
    for condition in CONDITIONS:
        part=[r for r in records if r['condition']==condition]
        for arm in ARMS:
            for metric in metrics:
                values=[r[metric] for r in part if r['arm']==arm and r[metric] is not None]
                summary.append(dict(arm=arm,condition=condition,metric=metric,n=len(values),mean=float(np.mean(values)) if values else None))
        index={(r['arm'],r['id']):r for r in part}
        for a,b in protocol['statistics']['comparisons']:
            for metric in metrics:
                values=[index[a,i][metric]-index[b,i][metric] for i in sorted({r['id'] for r in part})
                        if index[a,i][metric] is not None and index[b,i][metric] is not None]
                stats.append(dict(comparison=a+' minus '+b,condition=condition,metric=metric,**paired(values)))
        primary=[r for r in stats if r['condition']==condition and r['metric'] in ['ref_rgbuv_h','ref_ab_w1']]
        holm(primary)
    write(root/'metrics/summary.json',summary); table(root/'metrics/summary.csv',summary)
    write(root/'metrics/paired_statistics.json',stats)
    table(root/'metrics/paired_statistics.csv',[dict(r,ci95=json.dumps(r['ci95'])) for r in stats])


def package():
    assert read(ROOT/'evaluation_complete.json')['complete']
    jobs=read(ROOT/'submitted_jobs.json')
    accounting=subprocess.check_output(['sacct','-X','-j',','.join(map(str,jobs.values())),'-P',
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,AllocCPUS,NodeList'],text=True)
    (ROOT/'audit/sacct_at_packaging.psv').write_text(accounting)
    files=[p for name in ['audit','inputs','masks','outputs','metrics','logs'] for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix!='.tmp']
    files += [ROOT/name for name in ['protocol.json','evaluation.json','generation_complete.json','evaluation_complete.json','submitted_jobs.json']]
    files += [SOURCE/name for name in ['tools/t3_cfg_lab.py','tools/t3_color_metrics.py','scripts/t3_cfg_lab.sbatch','scripts/t3_cfg_lab_submit.py']]
    write(ROOT/'bundle_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as tar:
        for p in files+[ROOT/'bundle_manifest.json']:tar.add(p,arcname=str(p.relative_to(ROOT)))
    write(ROOT/'bundle_info.json',dict(bytes=(ROOT/'review_results.tar.gz').stat().st_size,sha256=sha(ROOT/'review_results.tar.gz'),members=len(files)+1))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','generate','process','neural','package'])
    globals()[parser.parse_args().action]()
