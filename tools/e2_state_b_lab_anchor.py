"""冻结B400，在线验证原输出、仅色度约束、色度与低频亮度约束。"""
import argparse
import collections
import copy
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_lab_anchor import LabAnchor
from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_pilot as pilot
from tools import e2_state_b_repair as repair
from tools.e2_fft_experiment import environment, read, write, sha
from tools.e2_oracle_texture import projected, weights_digest
from tools.bf_t3_three_arm_report import paired

SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('first_b', 'raw', 'ab', 'ab_l')
CHECKPOINT = '299eb2472dd718a28e5508141e70f3d081e3cf4bfcdd0f72058df41b265fd924'


def source_hashes():
    return {name: sha(SOURCE / name) for name in (
        'models/e2_lab_anchor.py', 'tools/e2_state_b_lab_anchor.py', 'pipelines/IMAGGarment_pipeline.py')}


def prepare(args):
    first, recent = read(args.baseline / 'protocol.json'), read(args.repair / 'protocol.json')
    assert first['pairs'] == recent['pairs']
    complete = read(args.repair / 'training/complete.json')
    assert complete['checkpoint_sha256'] == CHECKPOINT
    assert sha(args.repair / 'training/step400.pt') == CHECKPOINT
    assert complete['frozen_unchanged'] and recent['active_end'] == 50
    spec = dict(pairs=copy.deepcopy(first['pairs']), source=environment(), code=source_hashes(),
        checkpoint_sha256=CHECKPOINT, frozen_hash=complete['frozen_hash'], updates=0,
        sigma=24., arms=list(ARMS), sampler=recent['sampler'], active_end=50,
        anchor_source='same-seed fixed M0 anchor_s0/s1 from first B; targets are evaluation-only',
        implementation='Lab projection after final VAE decode before PNG; gamut-safe L and RGB feather',
        scope=first['scope'], baseline=str(args.baseline), repair=str(args.repair))
    for p in spec['pairs']:
        src, dst = args.baseline/'cases'/p['case_id'], args.out/'cases'/p['case_id']
        dst.mkdir(parents=True, exist_ok=True)
        names = ['mask.png', 'sketch.png', 'anchor_s0.png', 'anchor_s1.png']
        names += [c+'_ref.png' for c in pilot.CONDITIONS]
        for name in names:
            shutil.copyfile(src/name, dst/name)
            assert sha(dst/name) == sha(src/name)
        assert all(sha(dst/(c+'_ref.png')) == p['reference_hashes'][c] for c in pilot.CONDITIONS)
    write(args.out/'protocol.json', spec)
    print('LAB_ANCHOR_PREPARED', len(spec['pairs']), flush=True)


def layer_for(args, p, seed):
    folder = args.out/'cases'/p['case_id']
    anchor = np.asarray(Image.open(folder/f'anchor_s{seed}.png').convert('RGB'), np.float32)/255
    mask = np.asarray(Image.open(folder/'mask.png')) > 0
    return LabAnchor(anchor, mask, sigma=24.).to(args.device).eval()


def png(pipe, value):
    return pipe.numpy_to_pil(value.detach().cpu().permute(0, 2, 3, 1).float().numpy())[0]


def generate(args):
    spec = read(args.out/'protocol.json')
    assert spec['code'] == source_hashes()
    checkpoint = args.repair/'training/step400.pt'
    assert sha(checkpoint) == CHECKPOINT
    core, pipe, net = pilot.load_core(args)
    b = StateTextureB(color_channels=2).to(args.device)
    b.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False)['b'])
    b.eval().requires_grad_(False)
    before_m0, before_b = weights_digest(net), weights_digest(b)
    assert before_m0 == spec['frozen_hash']
    with torch.inference_mode():
        for p in spec['pairs'][args.shard::2]:
            folder = args.out/'cases'/p['case_id']
            expected = read(args.repair/'cases'/p['case_id']/'generation.json')
            rows = []
            for seed in range(2):
                pair = dict(p, seed=p['seed']+100000*seed)
                layer = layer_for(args, p, seed)
                # B条件仍与修复轮完全相同；本层不读取goal图或其统计。
                write(folder/f'anchor_s{seed}_audit.json', layer.describe())
                for condition in pilot.CONDITIONS:
                    key = f'{condition}_s{seed}'
                    values, _ = pilot.inputs(args, pair, condition, args.device)
                    values['color'] = layer.ab.expand(1, 2, 64, 48)/100
                    features = projected(net, core.tensor(args.data/pair['sketch']).to(args.device).half(),
                        core.tensor(folder/(condition+'_ref.png')).to(args.device).half())
                    captured = {}
                    def transform(value):
                        assert not captured
                        captured['raw'] = png(pipe, value)
                        captured['ab'] = png(pipe, layer(value, 'ab'))
                        captured['ab_audit'] = dict(layer.last_audit)
                        corrected = layer(value, 'ab_l')
                        captured['ab_l_audit'] = dict(layer.last_audit)
                        return corrected
                    latent = torch.randn((1, 4, 64, 48), device=args.device, dtype=torch.float16,
                        generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                    digest = core.digest(latent)
                    prior = next(r for r in expected['rows'] if r['condition']==condition and r['seed']==pair['seed'])
                    assert digest == prior['digest']
                    net.adapter.features = features
                    with repair.Tap(pipe, b, values, end=50) as tap:
                        corrected = pipe(ref_image=core.tensor(args.data/pair['sketch']).to(args.device).half(),
                            prompt=pair['caption'], texture_clip_image=None, texture_embeds=None,
                            null_prompt='', negative_prompt=' worst quality, low quality', width=384, height=512,
                            num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=0.,
                            latents=latent, generator=torch.Generator(device=args.device).manual_seed(pair['seed']),
                            num_inference_steps=50, decoded_output_transform=transform)[0]
                    net.adapter.clear()
                    assert tap.index == 50
                    raw = np.asarray(captured['raw'])
                    original = Image.open(args.repair/'cases'/p['case_id']/(key+'_b400.png')).convert('RGB')
                    assert np.array_equal(raw, np.asarray(original)), (p['case_id'], key, 'raw replay')
                    captured['ab_l'] = corrected
                    hashes = {}
                    for arm in ('raw', 'ab', 'ab_l'):
                        path = folder/(key+'_'+arm+'.png')
                        captured[arm].save(path)
                        hashes[arm] = sha(path)
                    rows.append(dict(condition=condition, seed=pair['seed'], digest=digest,
                        raw_bitwise=True, output_sha256=hashes,
                        projection={a:captured[a+'_audit'] for a in ('ab','ab_l')}))
                print('LAB_ANCHOR_GENERATED', p['case_id'], seed, flush=True)
            write(folder/'generation.json', dict(rows=rows, checkpoint_sha256=CHECKPOINT))
    assert weights_digest(net) == before_m0 and weights_digest(b) == before_b
    assert sha(checkpoint) == CHECKPOINT
    write(args.out/f'shard{args.shard}.json', dict(updates=0, raw_bitwise=True,
        m0_hash=before_m0, b_hash=before_b, frozen_unchanged=True, checkpoint_sha256=CHECKPOINT))


def score(picture, goal, new_goal, positive, anchor, raw, mask, coordinates, condition):
    rgb = np.asarray(picture, np.float32)/255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    def color(target):
        target_rgb = np.asarray(target, np.float32)/255
        difference = lab[mask].mean(0)-cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[mask].mean(0)
        local = cv2.cvtColor(cv2.GaussianBlur(rgb, (0,0), 12), cv2.COLOR_RGB2LAB)[mask, 1:]
        local_goal = cv2.cvtColor(cv2.GaussianBlur(target_rgb, (0,0), 12), cv2.COLOR_RGB2LAB)[mask, 1:]
        return dict(lab=float(np.linalg.norm(difference)), chroma=float(np.linalg.norm(difference[1:])),
                    lightness=float(abs(difference[0])), local_chroma=float(np.linalg.norm(local-local_goal, axis=1).mean()))
    primary, secondary = color(goal), color(new_goal)
    return dict(**primary, **{'new_'+k:v for k,v in secondary.items()},
        outside_change_from_raw=float(np.abs(rgb[~mask]-np.asarray(raw, np.float32)[~mask]/255).mean()),
        texture=pilot.texture_metrics(picture, coordinates, condition),
        **pilot.old_pattern(picture, anchor, positive, coordinates, condition))


def panel(args, p):
    folder = args.out/'cases'/p['case_id']
    canvas = Image.new('RGB', (5*154, 4*224), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, c in enumerate(pilot.VARIANTS):
        paths = [('target', args.repair/'cases'/p['case_id']/f'{c}_s0_goal.png'),
                 ('first B', args.baseline/'cases'/p['case_id']/f'{c}_s0_b400.png')]
        paths += [(a, folder/f'{c}_s0_{a}.png') for a in ('raw','ab','ab_l')]
        for col, (label, path) in enumerate(paths):
            x, y = col*154, row*224
            draw.text((x+3,y+3), c+' '+label, fill='black')
            canvas.paste(Image.open(path).convert('RGB').resize((150,200)), (x,y+22))
    canvas.save(folder/'panel.jpg', quality=94)


def summarize(args):
    spec = read(args.out/'protocol.json')
    audits = [read(args.out/f'shard{s}.json') for s in range(2)]
    assert audits[0] == audits[1] and audits[0]['updates'] == 0 and audits[0]['frozen_unchanged']
    identities = []
    for p in spec['pairs']:
        folder, old, recent = (root/'cases'/p['case_id'] for root in (args.out, args.baseline, args.repair))
        generation = read(folder/'generation.json')
        assert len(generation['rows']) == 16 and all(r['raw_bitwise'] for r in generation['rows'])
        mask = np.asarray(Image.open(folder/'mask.png')) > 0
        rows, switches = {}, {}
        for seed in range(2):
            anchor = Image.open(folder/f'anchor_s{seed}.png').convert('RGB')
            for c in pilot.CONDITIONS:
                key = f'{c}_s{seed}'
                goal, positive = (Image.open(old/(key+'_'+a+'.png')).convert('RGB') for a in ('target','vae'))
                new_goal = Image.open(recent/(key+'_goal.png')).convert('RGB')
                raw = Image.open(folder/(key+'_raw.png')).convert('RGB')
                entry = next(r for r in generation['rows'] if r['condition']==c and r['seed']==p['seed']+seed*100000)
                rows[key] = {}
                for arm in ARMS:
                    path = old/(key+'_b400.png') if arm=='first_b' else folder/(key+'_'+arm+'.png')
                    if arm != 'first_b':
                        assert sha(path) == entry['output_sha256'][arm]
                    rows[key][arm] = score(Image.open(path).convert('RGB'), goal, new_goal, positive,
                        anchor, raw, mask, p['patch_coordinates'], c.replace('_shift',''))
            for phase in ('','_shift'):
                for kind, a, b in [('direction','v16','h16'),('direction','v32','h32'),
                                   ('period','v16','v32'),('period','h16','h32')]:
                    keys = [a+phase+f'_s{seed}', b+phase+f'_s{seed}']
                    switches[f'{kind}_{a}{phase}_s{seed}'] = {
                        arm: bool(all(rows[k][arm]['texture']['hit'] for k in keys)) for arm in ARMS}
        aggregate = {}
        for arm in ARMS:
            metrics = {k:float(np.mean([r[arm][k] for r in rows.values() if r[arm][k] is not None]))
                       for k in rows[next(iter(rows))][arm] if k!='texture'}
            metrics.update({kind+'_strict':float(np.mean([r[arm] for k,r in switches.items() if k.startswith(kind)]))
                            for kind in ('direction','period')})
            aggregate[arm] = metrics
        result = dict(case_id=p['case_id'], split=p['split'], aggregate=aggregate, rows=rows, switches=switches)
        write(folder/'result.json', result)
        identities.append(result)
        panel(args,p)
        print('LAB_ANCHOR_EVALUATED',p['case_id'],flush=True)
    splits = {}
    for split in ('train','holdout'):
        cases = [p for p in identities if p['split']==split]
        metrics = cases[0]['aggregate']['raw']
        arms = {a:{k:float(np.mean([p['aggregate'][a][k] for p in cases])) for k in metrics} for a in ARMS}
        counts = {kind:{a:sum(v[a] for p in cases for k,v in p['switches'].items() if k.startswith(kind))
                        for a in ARMS} for kind in ('direction','period')}
        differences = {reference:{k:paired([p['aggregate']['ab_l'][k]-p['aggregate'][reference][k] for p in cases])
                                  for k in metrics} for reference in ('first_b','raw')}
        gates = dict(retain_switches=all(arms['ab_l'][k]>=arms['raw'][k] for k in ('direction_strict','period_strict')),
            retain_old_removal=arms['ab_l']['old_projection']<=arms['raw']['old_projection'],
            improve_color_vs_first=all(arms['ab_l'][k]<arms['first_b'][k] for k in ('chroma','local_chroma','lightness')),
            improve_color_vs_raw=all(arms['ab_l'][k]<arms['raw'][k] for k in ('chroma','local_chroma','lightness')),
            outside_unchanged=arms['ab_l']['outside_change_from_raw']==0)
        splits[split] = dict(n=len(cases), arms=arms, strict_switch_counts=counts, paired_difference=differences,
                            gates=gates, all_gates=all(gates.values()))
    result = dict(protocol=spec, training=audits[0], images=192, results=splits,
        interpretation='zero-training B400 plus deterministic decoded-output projection; no learned color claim; '
                       'same 8 holdout identities are regression cases, not a fresh independent generalization test',
        next_action='no color-head training needed for this controlled task' if splits['holdout']['all_gates']
                    else 'inspect failed regression gate before any additional training')
    write(args.out/'summary.json',result)
    archive = args.out/'lab_anchor_review.zip'
    files = [args.out/'protocol.json', args.out/'summary.json', *args.out.glob('shard*.json')]
    files += list((args.out/'cases').glob('*/*'))
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path.is_file():z.write(path,path.relative_to(args.out))
    print('LAB_ANCHOR_RESULTS', {s:{'gates':v['gates'],'arms':v['arms']} for s,v in splits.items()},flush=True)
    print('LAB_ANCHOR_ARCHIVE',archive.stat().st_size,sha(archive),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','generate','summarize'))
    parser.add_argument('--project',type=Path,default=pilot.PROJECT)
    parser.add_argument('--data',type=Path,default=pilot.PROJECT.parent/'datasets/BF')
    parser.add_argument('--baseline',type=Path,default=repair.BASELINE)
    parser.add_argument('--repair',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_repair_20261112')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    globals()[args.action](args)


if __name__=='__main__':main()
