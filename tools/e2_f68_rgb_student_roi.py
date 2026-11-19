"""拆分原 64 对单次 RGB 残差图的外观、LPIPS、SSIM 与 FID。"""
import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_rgb_distill as distill
from tools.e2_f68_edge_residual import ACCEPTANCE_SHA, mix_pixels, summarize, verified_files
from tools.e2_f68_real_paired import METRICS, image_scores, metric_assets, metric_models, rgb, verify, write
from tools.e2_fft_experiment import environment, read, sha

ARTIFACT = 'ab_inside_lpips_20261119'
RUN = 'roi_64_20261119'
AB_SHA = 'b5420f86f5b3c09698194d4d3ef3582521ed3aa90e3dc4bd098ef358a7778216'
E11_SHA = '83a3548d6f8287c9246d94b1ff16126ea6d4d0f4d38634ea35ac513b7c36509e'
ORIGINAL = ('off', 'epoch11', 'A', 'B', 'blend')
RADIUS = 8
CORE_RADIUS = 20


def appearance_inputs(picture):
    value = roi.lab(picture)
    return value, roi.gaussian(value, 8), roi.band(value)


def pixels_mixed(base, destination, weight):
    base = base.astype('float32')
    return np.rint(np.clip(base + weight[..., None]*(destination-base), 0, 255)).astype('uint8')


def appearance_probes(images, weights, core, colors):
    edits = {}
    for source in ('off', 'blend'):
        for name, weight in zip(roi.REGIONS, weights):
            edits[f'B_{name}_from_{source}'] = pixels_mixed(images['B'], images[source], weight)
    inside = weights[0]
    edits['B_inside_half'] = pixels_mixed(images['B'], images['off'], inside*.5)
    edits['E11_inside_half'] = pixels_mixed(images['epoch11'], images['off'], inside*.5)
    edits['E11_inside_from_blend'] = pixels_mixed(images['epoch11'], images['blend'], inside)
    current, low, band = colors['B']
    _, ref_low, ref_band = colors['blend']
    phase_delta = ref_band[..., 0] - band[..., 0]
    low_delta = ref_low - low
    active = core if core.sum() >= 256 else inside >= .95
    gain = np.clip((roi.rms(ref_band[active, 0]) + .25)/(roi.rms(band[active, 0]) + .25), .5, 2.) if active.any() else 1.
    amp_delta = (gain-1)*band[..., 0]
    roundtrip = roi.as_rgb(current).astype('int16')
    for name in ('low_L', 'low_ab', 'amplitude', 'band', 'low_Lab_and_band'):
        candidate = current.copy()
        if name in ('low_L', 'low_Lab_and_band'):
            candidate[..., 0] += inside * low_delta[..., 0]
        if name in ('low_ab', 'low_Lab_and_band'):
            candidate[..., 1:] += inside[..., None] * low_delta[..., 1:]
        if name == 'amplitude':
            candidate[..., 0] += inside * amp_delta
        if name in ('band', 'low_Lab_and_band'):
            candidate[..., 0] += inside * phase_delta
        edits['B_inside_' + name] = roi.apply_lab_edit(images['B'], roundtrip, candidate)
    return edits, float(gain)


def band_alignment(before, after, region):
    a, b = before[2][region, 0].astype('float64'), after[2][region, 0].astype('float64')
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denom) if denom > 1e-12 else None


def appearance_summary(rows):
    result = {}
    for group in ('all', 'outwear'):
        selected = [r for r in rows if group == 'all' or r['category'] == group]
        labels = np.array([r['category'] for r in selected])
        sections = {}
        for section in (*roi.REGIONS, 'deep_inside'):
            use = np.array([section != 'deep_inside' or row['core_valid'] for row in selected])
            counts = int(use.sum())
            values = {}
            if not counts:
                continue
            for arm in ORIGINAL:
                sample = next(row['regions'][arm][section] for row, include in zip(selected, use) if include)
                stats = {}
                for key in sample:
                    series = np.array([row['regions'][arm][section][key] for row, include in zip(selected,use) if include], dtype='float64')
                    off = np.array([row['regions']['off'][section][key] for row, include in zip(selected,use) if include], dtype='float64')
                    entry = dict(mean=float(np.nanmean(series)) if np.isfinite(series).any() else None,
                                 n=int(np.isfinite(series).sum()))
                    if np.any(np.isfinite(series) & np.isfinite(off)):
                        entry['vs_off'] = roi.paired(off, series, labels[use])
                    stats[key] = entry
                values[arm] = stats
            sections[section] = dict(n=counts, arms=values)
        result[group] = sections
    return result


def case_images(pair, index, args, original_rows, teachers, students, rows):
    case = pair['case_id']
    assert original_rows[index]['index'] == index and original_rows[index]['case_id'] == case
    assert teachers[index]['case_id'] == case and teachers[index]['index'] == index
    raw, mask = roi.checked_case(args.baseline, args.data, pair, original_rows[index])
    pictures = {'off': raw['off'], 'blend': mix_pixels(raw['off'], raw['on'], mask, {'blend': .25})['blend']}
    truth = {'off': original_rows[index]['scores']['off'], 'blend': teachers[index]['scores']['edge_a025']}
    for arm, path in students.items():
        row = rows[arm][index]
        assert row['case_id'] == case
        picture = path/'images'/(case+'.png')
        assert sha(picture) == row['png_sha256']
        pictures[arm] = np.asarray(rgb(picture))
        truth[arm] = row['scores']['student']
    return pictures, raw['gt'], mask, truth


def panel(items, out):
    if not items:
        return
    names = ('gt', 'off', 'epoch11', 'A', 'B', 'blend', 'B_inside_from_blend', 'B_inside_half')
    image = Image.new('RGB', (144*len(names), 209*len(items)), 'white')
    draw = ImageDraw.Draw(image)
    for j, (case, pictures) in enumerate(items):
        draw.text((3,j*209+2), case, fill='black')
        for k, name in enumerate(names):
            image.paste(Image.fromarray(pictures[name]).resize((136,181), Image.Resampling.BILINEAR),
                        (144*k,j*209+14))
            draw.text((144*k+2,j*209+195), name, fill='black')
    image.save(out/'outwear_examples.jpg', quality=93)


@torch.no_grad()
def run(args):
    baseline, previous = args.baseline, args.baseline/distill.RUN
    ab = previous/ARTIFACT
    out = ab/RUN
    out.mkdir(exist_ok=True)
    if (out/'roi_review.zip').is_file():
        print('ROI_STUDENT_ARCHIVE', sha(out/'roi_review.zip'), flush=True)
        return
    blend = baseline/'edge_residual_acceptance_20261118'
    protocol = verify(argparse.Namespace(out=baseline))
    verified_files(baseline, roi.ARCHIVE_SHA, ('protocol.json','per_image.json','fid_pool3.npz'))
    verified_files(blend, ACCEPTANCE_SHA, ('per_pair.json','inner_mix_pool3.npz'))
    assert sha(ab/'ab_review.zip') == AB_SHA
    assert metric_assets() == protocol['metric_assets']
    ids = distill.subset(protocol)
    students = {'epoch11':previous/'eval64', 'A':ab/'A_l1/eval64', 'B':ab/'B_inside_lpips/eval64'}
    ab_manifest = read(ab/'manifest.json')
    source_rows, expected, features = {}, {}, {}
    with np.load(baseline/'fid_pool3.npz', allow_pickle=False) as f:
        features.update({key:f[key][ids] for key in ('off', 'gt')})
    with np.load(blend/'inner_mix_pool3.npz', allow_pickle=False) as f:
        features['blend'] = f['edge_a025'][ids]
    for arm, path in students.items():
        ref_sha = E11_SHA if arm == 'epoch11' else ab_manifest[path.relative_to(ab).as_posix()+'/review.zip']
        verified_files(path, ref_sha, ('per_pair.json','student_pool3.npz','result.json'))
        source_rows[arm] = {record['index']:record for record in read(path/'per_pair.json')}
        expected[arm] = read(path/'result.json')['groups']['all']['arms']['student']['fid']['fid']
        assert list(source_rows[arm]) == ids
        with np.load(path/'student_pool3.npz', allow_pickle=False) as f:
            features[arm] = f['student']
    orig, teachers = read(baseline/'per_image.json'), read(blend/'per_pair.json')
    metric, inception = metric_models('cpu', protocol['metric_assets'])
    rows, new_features, examples = [], {}, []
    with tempfile.TemporaryDirectory(prefix='f68_roi64_',dir=os.environ.get('SLURM_TMPDIR')) as temp:
        for index in ids:
            pair = protocol['pairs'][index]
            case = pair['case_id']
            images, gt, mask, original_scores = case_images(pair,index,args,orig,teachers,students,source_rows)
            partition = roi.regions(mask, RADIUS)
            masks = torch.from_numpy(partition[None].astype('float32'))
            weights = [roi.gaussian(region.astype('float32'), 2) for region in partition]
            assert np.allclose(np.sum(weights,axis=0), 1, atol=1e-6)
            core = cv2.erode(mask.astype('uint8'), roi.kernel(CORE_RADIUS)).astype(bool)
            colors = {name:appearance_inputs(pic) for name,pic in images.items()}
            colors['gt'] = appearance_inputs(gt)
            edits, gain = appearance_probes(images, weights, core, colors)
            candidates = {**images, **edits}
            target = roi.lpips_features(metric, roi.tensors([gt]))
            scores, regions = {}, {}
            for names0 in range(0,len(candidates),4):
                names = list(candidates)[names0:names0+4]
                batch = roi.tensors([candidates[name] for name in names])
                maps = roi.maps(metric,target,batch)
                lpips = sum(layer.mean(dim=(-2,-1)) for layer in maps).numpy()
                if names0+len(names)>len(ORIGINAL):
                    indices=[j for j,n in enumerate(names) if n in edits]
                    for j, feature in zip(indices,inception(batch[indices])[0].flatten(1).numpy()):
                        new_features.setdefault(names[j], []).append(feature)
                for j,name in enumerate(names):
                    picture = candidates[name]
                    if name in original_scores:
                        score = original_scores[name]
                        assert abs(float(lpips[j])-score['lpips']) < 5e-6, (case,name)
                        ssim, ssim_share, area = roi.ssim_regions(picture, gt, partition)
                        assert abs(ssim-score['ssim']) < 1e-8
                        pieces = roi.apportion([layer[j:j+1] for layer in maps], masks)[0]
                        assert abs(float(pieces.sum())-float(lpips[j])) < 1e-6
                        regions[name] = {}
                        for k, sector in enumerate(roi.REGIONS):
                            style = roi.appearance(colors[name], colors['gt'], partition[k])
                            style.update(lpips_share=float(pieces[k]), ssim_share=ssim_share[k], ssim_area=area[k])
                            regions[name][sector] = style
                        if core.sum() >= 256:
                            regions[name]['deep_inside'] = roi.appearance(colors[name], colors['gt'], core)
                            regions[name]['deep_inside']['band_alignment_gt'] = band_alignment(colors[name], colors['gt'], core)
                            regions[name]['deep_inside']['band_alignment_teacher'] = band_alignment(colors[name], colors['blend'], core)
                    else:
                        path = Path(temp)/'edit.png'
                        frame = Image.fromarray(picture)
                        frame.save(path)
                        score = image_scores(frame, Image.fromarray(gt), path, args.data/pair['gt'], mask)
                        score['lpips'] = float(lpips[j])
                    scores[name] = score
            raw_background = partition[2] & (np.asarray(weights[:2]).sum(axis=0) == 0)
            assert all(np.array_equal(images[arm][raw_background], images['off'][raw_background]) for arm in ORIGINAL)
            rows.append(dict(index=index,case_id=case,category=pair['category'],gain=gain,core_valid=bool(core.sum()>=256),
                             core_pixels=int(core.sum()),scores=scores,regions=regions))
            if pair['category']=='outwear' and len(examples)<4:
                examples.append((case,{'gt':gt,**candidates}))
            if len(rows)%8==0:
                print('ROI_STUDENT_SCORED', len(rows), flush=True)
    extra = {name:np.stack(f) for name,f in new_features.items()}
    features.update(extra)
    assert len(rows)==64 and all(v.shape==(64,2048) and np.isfinite(v).all() for v in features.values())
    variants = (*ORIGINAL, *extra)
    result = summarize(rows,features,variants,('off','B','blend'),('B','B_inside_half','E11_inside_half','B_inside_from_off','B_inside_from_blend'))
    for key, value in expected.items():
        assert abs(result['groups']['all']['arms'][key]['fid']['fid']-value) < 1e-7
    result['regional'] = appearance_summary(rows)
    result['acceptance'] = {}
    for name in variants:
        group = result['groups']['all']['arms']
        result['acceptance'][name] = distill.acceptance({'groups':{'all':{'arms':{**group,'student':group[name]}}}})
    metadata = dict(samples=len(rows), indices=ids, original_sha256=roi.ARCHIVE_SHA, ab_sha256=AB_SHA,
                    epoch11_sha256=E11_SHA, teacher_sha256=ACCEPTANCE_SHA, radius=RADIUS, core_radius=CORE_RADIUS,
                    feather_sigma_px=2, low_Lab_sigma_px=8, band_sigmas_px=[1,4], original_background_equal=True,
                    new_s0_trajectories=0, optimizer_updates=0, implementation_sha256=sha(Path(__file__)), environment=environment())
    write(out/'meta.json',metadata)
    write(out/'per_pair.json',rows)
    write(out/'result.json',result)
    np.savez_compressed(out/'edit_pool3.npz',**extra)
    panel(examples,out)
    files = ('meta.json','per_pair.json','result.json','edit_pool3.npz','outwear_examples.jpg')
    write(out/'manifest.json',{name:sha(out/name) for name in files})
    temp_archive = out/'roi_review.zip.tmp'
    with zipfile.ZipFile(temp_archive,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in (*files,'manifest.json'):
            archive.write(out/name,name)
    temp_archive.replace(out/'roi_review.zip')
    print('ROI_STUDENT_ARCHIVE',sha(out/'roi_review.zip'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    args = parser.parse_args()
    args.baseline = args.prior/distill.BASE
    args.data = args.prior.parents[2]/'datasets/BF'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    cv2.setNumThreads(1)
    run(args)
