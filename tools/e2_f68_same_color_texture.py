"""复用冻结 f68 的同色条纹开关图，诊断双图 0.25 的参考响应。"""
import argparse
import os
import platform
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from eval.eval_utils import derive_region_masks
from models.e2_lab_anchor import rgb_to_lab
from tools.e2_f68_edge_residual import mix_pixels
from tools.e2_f68_real_paired import B_HASH, CHECKPOINT_SHA, NET_HASH, SIZE, rgb, write
from tools.e2_f68_region_diagnostic import shape_score
from tools.e2_fft_experiment import read, sha
from tools.e2_state_b_pilot import texture_metrics

RUN = 'same_color_texture_20261119'
HISTORICAL = 'e2_state_b_texture_independent_holdout_20261116'
CONDITIONS = ('v16', 'h16', 'v32', 'h32')
ARMS = ('off', 'raw', 'blend')
PAIRS = (('direction16', 'v16', 'h16'), ('direction32', 'v32', 'h32'),
         ('period_v', 'v16', 'v32'), ('period_h', 'h16', 'h32'))
SOURCE_ARCHIVE = '438b134a54bdb466467346ad1d22a86998e91f59928b86424cac2f3124e3517c'
SOURCE_PROTOCOL = '6c48c387659927a53e52d0f3cc8a8534b64b4461b86141fd78957483027836eb'
SOURCE_ROWS = '08bd6f5d4b1b83a4211700051130f34906046840f2e155852cac1fc8b8c77264'
HANN = np.outer(np.hanning(64), np.hanning(64))


def lightness(picture):
    rgb_f64 = torch.from_numpy(picture.copy()).permute(2,0,1)[None].double()/255
    return rgb_to_lab(rgb_f64)[0,0].numpy()


def score_texture(picture, coords, condition):
    light = lightness(picture)
    result = texture_metrics(np.repeat((light*2.55)[...,None], 3, -1), coords, condition)
    k = 64 // int(condition[1:])
    y, x = (0,k) if condition[0]=='v' else (k,0)
    components, total = [], []
    for left, top in coords:
        patch = light[top:top+64,left:left+64]
        spectrum = np.fft.fft2((patch-patch.mean())*HANN)
        components.append(abs(spectrum[y,x])*np.sqrt(2)/HANN.sum())
        total.append(float(patch.std()))
    result.update(target_contrast=float(np.mean(components)), patch_l_std=float(np.mean(total)))
    return result


def score_image(picture, reference, anchor, coords, condition, mask, outside):
    values = score_texture(picture, coords, condition)
    values['target_contrast_ratio'] = values['target_contrast']/reference['target_contrast']
    values.update(shape_score(picture, mask))
    hue = np.asarray(Image.fromarray(picture).convert('HSV'))
    values['background_saturated_fraction'] = float((hue[...,1][outside]/255>.16).mean())
    values['outside_rgb_mae_to_off'] = float(np.abs(picture[outside].astype('int16')-anchor[outside]).mean())
    return values


def two_ends(rows, pictures, outside, case):
    output = []
    for kind, a, b in PAIRS:
        entry = dict(case=case,kind=kind,conditions=[a,b],arms={})
        for arm in ARMS:
            m,n = (rows[c]['arms'][arm] for c in (a,b))
            direction_spectral = all(v['fft'] and v['direction'] for v in (m,n))
            period_switch = all(v['period_hit'] for v in (m,n))
            background_delta = np.abs(pictures[a][arm][outside].astype('int16') - pictures[b][arm][outside])
            entry['arms'][arm] = dict(
                passed=direction_spectral and (kind.startswith('direction') or period_switch),
                strict_passed=m['hit'] and n['hit'],
                target_preference_margin=(m['bands'][a]-m['bands'][b]+n['bands'][b]-n['bands'][a])/2,
                periods=[m['period'],n['period']], contrast_ratios=[m['target_contrast_ratio'],n['target_contrast_ratio']],
                contour_min=min(m['contour_f1'],n['contour_f1']), outside_rgb_mae_for_swap=float(background_delta.mean()))
        output.append(entry)
    return output


def grouped_stats(rows, pairs):
    result = {}
    for arm in ARMS:
        records = [row['arms'][arm] for row in rows]
        result[arm] = dict(
            endpoints=len(records), direction=sum(r['direction'] for r in records),
            period=sum(r['period_hit'] for r in records), fft=sum(r['fft'] for r in records),
            all_three=sum(r['hit'] for r in records), no_period=sum(r['period'] is None for r in records),
            target_contrast_ratio_mean=float(np.mean([r['target_contrast_ratio'] for r in records])),
            target_contrast_ratio_median=float(np.median([r['target_contrast_ratio'] for r in records])),
            contrast_ratio_ge_half=sum(r['target_contrast_ratio'] >= .5 for r in records),
            by_condition={}, switches={})
        for name in ('contour_f1','iou','background_saturated_fraction','outside_rgb_mae_to_off','target_contrast','patch_l_std'):
            result[arm][name] = float(np.mean([r[name] for r in records]))
        for c in CONDITIONS:
            current = [r['arms'][arm] for r in rows if r['condition']==c]
            result[arm]['by_condition'][c] = dict(n=len(current), direction=sum(x['direction'] for x in current),
                period=sum(x['period_hit'] for x in current), hit=sum(x['hit'] for x in current),
                contrast_ratio=float(np.mean([x['target_contrast_ratio'] for x in current])))
        for key in (*[p[0] for p in PAIRS], 'direction', 'period'):
            group = [p['arms'][arm] for p in pairs if p['kind'].startswith(key)]
            result[arm]['switches'][key] = dict(n=len(group), passed=sum(p['passed'] for p in group),
                strict=sum(p['strict_passed'] for p in group),
                mean_preference=float(np.mean([p['target_preference_margin'] for p in group])),
                mean_background_mae=float(np.mean([p['outside_rgb_mae_for_swap'] for p in group])))
    return result


def diagnostic_panel(examples, path):
    arms = ('ref', *ARMS)
    image = Image.new('RGB', (len(CONDITIONS)*137, len(examples)*(19+len(arms)*184)), 'white')
    draw = ImageDraw.Draw(image)
    for i,(case,batch) in enumerate(examples):
        offset=i*(19+len(arms)*184)
        draw.text((3,offset+2),case,fill='black')
        for j,arm in enumerate(arms):
            y=offset+19+j*184
            for k,c in enumerate(CONDITIONS):
                image.paste(Image.fromarray(batch[c][arm]).resize((128,166),Image.Resampling.BILINEAR),(k*137,y))
                draw.text((k*137+3,y+168),f'{arm} {c}',fill='black')
    image.save(path,quality=92)


@torch.no_grad()
def analyze(out, source, assets, data=None):
    protocol, generation = (read(source/n) for n in ('texture_protocol.json','generation.json'))
    assert sha(source/'texture_protocol.json') == SOURCE_PROTOCOL and sha(source/'generation.json') == SOURCE_ROWS
    assert protocol['checkpoint_sha256'] == generation['checkpoint_sha256'] == CHECKPOINT_SHA
    assert protocol['frozen_b_hash'] == B_HASH and protocol['frozen_m0_hash'] == NET_HASH
    assert protocol['sampler'] == dict(steps=50,cfg=7.,sketch_scale=.6,active=[25,50])
    assert generation['frozen_unchanged'] and generation['paired_noise'] and generation['paired_prefix_bitwise']
    assert generation['heldout_used'] and protocol['heldout_used'] and generation['updates_committed']==0
    assert tuple(protocol['conditions']) == CONDITIONS
    cases = protocol['evaluation_pairs']
    assert len(cases)==len({p['identity_group'] for p in cases})==11
    assert not {p['identity_group'] for p in cases} & {p['identity_group'] for p in protocol['train']}
    generated = {(r['case'],r['condition']):r for r in generation['rows']}
    assert len(generated)==len(generation['rows'])==len(cases)*len(CONDITIONS)
    folder = out/'images/blend'
    folder.mkdir(parents=True,exist_ok=True)
    rows, pairs, examples = [], [], []
    for pair in cases:
        case, coords = pair['case_id'], pair['patch_coordinates']
        assert len(coords)>=3
        root = assets/case
        mask_path = root/'mask.png'
        assert sha(mask_path)==pair['mask_sha256']
        mask = np.asarray(rgb(mask_path))[...,0]>0
        assert mask.shape == (SIZE[1],SIZE[0])
        assert all(x>=0 and y>=0 and x+64<=SIZE[0] and y+64<=SIZE[1]
                   and mask[y:y+64,x:x+64].all() for x,y in coords)
        outside,_ = derive_region_masks(mask,kernel_size=9)
        assert outside.sum()>=50
        sketch_sha = pair['source_hashes']['target']['sketch']
        if data is not None:
            assert sha(data/pair['sketch'])==sketch_sha
        assert 'following the provided garment sketch.' in pair['caption']
        batch, case_rows, signatures, noise = {}, {}, [], set()
        for condition in CONDITIONS:
            stem = case+'_'+condition
            entry = generated[case,condition]
            assert entry['seed']==pair['seed'] and entry['caption']==pair['caption']
            assert entry['sketch_sha256']==sketch_sha
            assert entry['exact_same_colour'] and entry['paired_prefix_bitwise']
            ref_path = root/(condition+'_ref.png')
            assert sha(ref_path)==entry['reference_sha256']==pair['reference_hashes'][condition]
            reference = np.asarray(rgb(ref_path))
            signatures.append(np.sort(np.ascontiguousarray(reference).view('V3'),axis=None))
            ref_score = score_texture(reference,coords,condition)
            assert ref_score['hit'] and ref_score['target_contrast']>0, (case, condition)
            pictures = {}
            for arm in ('off','raw'):
                png = source/'images'/arm/(stem+'.png')
                recorded = entry['arms'][arm]
                assert sha(png)==recorded['sha256'], png
                noise.add(recorded['noise_digest'])
                pictures[arm] = np.asarray(rgb(png))
                assert pictures[arm].shape==reference.shape==(SIZE[1],SIZE[0],3)
            pictures['blend'] = mix_pixels(pictures['off'],pictures['raw'],mask,{'blend':.25})['blend']
            destination = folder/(stem+'.png')
            Image.fromarray(pictures['blend']).save(destination)
            scores = {arm:score_image(pictures[arm],ref_score,pictures['off'],coords,condition,mask,outside)
                      for arm in ARMS}
            row = dict(case=case,condition=condition,caption=pair['caption'],seed=pair['seed'],
                       ref=ref_score,reference_sha256=entry['reference_sha256'],blend_sha256=sha(destination),arms=scores)
            rows.append(row)
            case_rows[condition] = row
            batch[condition] = {**pictures,'ref':reference}
        assert len(noise)==1 and all(np.array_equal(signatures[0],sig) for sig in signatures[1:])
        pairs.extend(two_ends(case_rows,batch,outside,case))
        if len(examples)<2:
            examples.append((case,batch))
        print('F68_TEXTURE_SWAP_CASE',case,len(rows),flush=True)
    result = grouped_stats(rows,pairs)
    meta = dict(
        origin_archive_sha256=SOURCE_ARCHIVE, origin_protocol_sha256=SOURCE_PROTOCOL, origin_rows_sha256=SOURCE_ROWS,
        sampler=protocol['sampler'], frozen_b_hash=B_HASH, frozen_m0_hash=NET_HASH, frozen_checkpoint_sha256=CHECKPOINT_SHA,
        source_sha256=sha(Path(__file__)), sample_identities=len(cases), historical_images_reused=2*len(rows),
        synthetic_references=len(rows), new_s0_trajectories=0, optimizer_updates=0, same_rgb_multiset=True,
        blend=dict(edge_radius=8, feather_sigma=2, angle_tolerance=20, period_relative_tolerance=.25, alpha_inside=.25),
        contrast='RMS of the target Fourier fundamental on Hann-windowed 64px L* patches, normalized to the same reference patches',
        background='9px sketch dilation exterior; per-pixel RGB MAE 0..255 and saturation>0.16, no replacement GT',
        environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, torch=torch.__version__))
    write(out/'meta.json',meta)
    write(out/'per_image.json',rows)
    write(out/'per_pair.json',pairs)
    write(out/'result.json',result)
    diagnostic_panel(examples,out/'examples.jpg')
    return result


def archive(out):
    files = [out/name for name in ('meta.json','per_image.json','per_pair.json','result.json','examples.jpg')]
    files.extend(sorted((out/'images/blend').glob('*.png')))
    manifest = {path.relative_to(out).as_posix():sha(path) for path in files}
    write(out/'manifest.json',manifest)
    temporary = out/'texture_swap_review.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for path in (*files,out/'manifest.json'):
            z.write(path,path.relative_to(out).as_posix())
    temporary.replace(out/'texture_swap_review.zip')
    print('F68_TEXTURE_SWAP_ARCHIVE',sha(out/'texture_swap_review.zip'),flush=True)


def run(prior):
    source = prior.parent/HISTORICAL
    out = prior/RUN
    out.mkdir(exist_ok=True)
    if (out/'texture_swap_review.zip').is_file():
        print('F68_TEXTURE_SWAP_ARCHIVE',sha(out/'texture_swap_review.zip'),flush=True)
        return
    assert sha(source/'endpoint_review.zip') == SOURCE_ARCHIVE
    result = analyze(out, source, source/'cases', prior.parents[2]/'datasets/BF')
    for arm, s in result.items():
        print('F68_TEXTURE_SWAP_RESULT',arm,s['switches']['direction']['passed'],s['switches']['period']['passed'],
              s['all_three'],s['target_contrast_ratio_mean'],flush=True)
    archive(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    cv2.setNumThreads(1)
    run(args.prior)
