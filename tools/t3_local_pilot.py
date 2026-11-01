"""T3 局部纹理小验证：准备、验收、双臂训练、配对生成及停止判定。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get('T3_PROJECT', '/share/home/u2515283058/Mymodel'))
NAME = 't3_local_texture_pilot_20261101'
ROOT = PROJECT / 'output_eval' / NAME
BASE = PROJECT / 'output_eval/e2a_sup_comp_01_20261021/training/formal/T3/checkpoint-final/joint_model.pt'
BASE_SHA = '25f95844696ba5c07eb9223e3bdd4d68c78ba4f1a36d561569c35cc90186d259'
ARMS = ['T3', 'residual', 'reference']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable(value):
    return hashlib.sha256(('T3_LOCAL_1101/' + str(value)).encode()).hexdigest()


def image_tensor(path, size=(384, 512)):
    import torch
    image = Image.open(path).convert('RGB').resize(size, Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, np.float32).transpose(2, 0, 1).copy())[None] / 255


def coordinates(mask, side=128, maximum=4):
    import cv2
    body = cv2.erode(mask.astype('uint8'), np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT)
    points = [(x, y) for y in range(0, 513-side, 32) for x in range(0, 385-side, 32)
              if body[y:y+side, x:x+side].all()]
    if len(points) > maximum:
        points = [points[i] for i in np.linspace(0, len(points)-1, maximum).round().astype(int)]
    return points


def make_mask(row):
    from eval.eval_utils import prepare_evaluation_masks
    mask = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')['garment']
    if mask is None:
        mask = np.zeros((512, 384), bool)
    path = ROOT / 'masks' / (row['id'] + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype('uint8') * 255).save(path)
    row['mask'], row['patches'] = str(path), coordinates(mask)
    return row


def synthetic(row, angle, period, phase, split):
    name = '%s_%s_a%d_p%d' % (row['id'], split, angle, period)
    def pattern(h, w):
        yy, xx = np.mgrid[:h, :w]
        wave = np.sin(2*np.pi*((xx if angle == 90 else yy)/period+phase))
        bright = .50 + .32*np.tanh(4*wave)
        return np.stack([bright*.75, bright*.9, bright], -1)
    folder = ROOT / 'synthetic' / split
    folder.mkdir(parents=True, exist_ok=True)
    reference, target = folder/(name+'_ref.png'), folder/(name+'_target.png')
    Image.fromarray(np.round(pattern(256, 256)*255).astype('uint8')).save(reference)
    mask = np.asarray(Image.open(row['mask']).convert('L'))[..., None] / 255
    sketch = np.asarray(Image.open(row['sketch']).convert('RGB').resize((384, 512))) / 255
    canvas = (1-mask) + mask*pattern(512, 384)*(.75+.25*sketch)
    Image.fromarray(np.round(canvas*255).astype('uint8')).save(target)
    return dict(row, id=name, identity=row['id'], reference=str(reference), gt=str(target),
                condition='synthetic', expected_angle=angle, expected_period=period,
                caption='a blue garment with fine parallel stripes', seed=42)


def prepare():
    assert sha(BASE) == BASE_SHA
    assert not (ROOT/'protocol.json').exists(), '实验目录已准备；不要覆盖冻结协议'
    train = read(PROJECT/'data/train_bf_texture.json')
    old = read(PROJECT/'output_eval/t3_p0_p1_20261028/protocol/input_manifest.json')
    originals = {r['id']: r for r in old if r['condition'] == 'original'}
    swapped = {r['id']: r for r in old if r['condition'] == 'swapped'}
    ids = sorted(swapped, key=stable)[:16]
    evaluation = []
    for identity in ids:
        for condition, rows in [('original', originals), ('swapped', swapped)]:
            row = dict(rows[identity], identity=identity)
            row['condition'] = condition
            evaluation.append(make_mask(row))
    training = []
    data = PROJECT.parent / 'datasets/BF/training'
    for r in sorted(train, key=lambda r: stable(Path(r['cloth']).stem)):
        identity = Path(r['cloth']).stem
        assert identity not in originals
        row = dict(id=identity, identity=identity, sketch=str(data/r['sketch']), reference=str(data/r['texture']),
                   gt=str(data/r['cloth']), caption=r['caption'], condition='original', seed=42)
        ref = np.asarray(Image.open(row['reference']).convert('L'), float) / 255
        if ref.std() < .04:
            continue
        row = make_mask(row)
        if len(row['patches']) < 2:
            continue
        training.append(row)
        if len(training) == 128:
            break
    assert len(training) == 128
    synth_train = [synthetic(r, a, p, 0., 'train') for r in training[:8] for a in [0,90] for p in [12,20]]
    valid_eval = [r for r in evaluation if r['condition']=='original' and len(r['patches'])>=2][:2]
    assert len(valid_eval) == 2
    synth_eval = [synthetic(r,a,p,.23,'dev') for r in valid_eval for a in [0,90] for p in [16,24]]
    # 合成集使用已定义的像素尺度；真实面料不宣称物理周期或逐像素对应。
    training += synth_train
    evaluation += synth_eval
    for row in training + evaluation:
        row['inputs_sha256'] = {k: sha(row[k]) for k in ['sketch','reference','gt','mask']}
    write(ROOT/'inputs/train.json', training)
    write(ROOT/'inputs/evaluation.json', evaluation)
    protocol = dict(name=NAME, baseline_sha256=BASE_SHA, training_real=128, training_synthetic=32,
        evaluation_real_identities=16, evaluation_synthetic_sketches=2, evaluation_rows=len(evaluation),
        arms=ARMS, updates=400, accumulation=2, learning_rate=1e-4, seed=110142,
        reference_weight=.1, anchor_weight=1., training_timesteps=[30,249],
        training_objective='epsilon MSE + boundary/outside same-state T3 epsilon MSE + lambda*reference gray spectrum Hellinger',
        reference_supervision='128px fully interior patches; native256 square reference; no RGB histogram loss',
        branch='raw reference CNN -> 256 local tokens -> target query attention -> zero conv; two high-resolution resnets',
        branch_inference='conditional UNet forward only; timestep<=249; unconditional residual strictly off',
        inference=dict(size=[384,512],steps=50,cfg=7.,sketch_scale=.6,ipa_scale=1.),
        gate=dict(gray_dists_improvement=.002,gray_dists_CI_upper_below=0,
                  minimum_valid_fraction=.8,
                  iou_degradation_max=.01,edge_f1_degradation_max=.02,background_mae_increase_max=.005,
                  synthetic_orientation_accuracy_min=.75,synthetic_period_relative_error_max=.2),
        decision='must beat T3 and residual-only on gray DISTS plus structure and synthetic gates; otherwise stop',
        budget='two single-GPU training arms, 30min each maximum; no automatic larger training',
        independent_confirmation=False,scope='exploratory mechanism pilot; previously viewed real development identities',
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(ROOT/'protocol.json', protocol)
    print('PREPARED', len(training), len(evaluation), flush=True)


def load_pipe():
    import torch
    spec = importlib.util.spec_from_file_location('t3_local_inference', SOURCE/'inference_IMAGGarment-1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sd = PROJECT/'models/stable-diffusion-v1-5'
    args = SimpleNamespace(GAM_model_ckpt=str(BASE),texture_ckpt=str(BASE),base_model_path=str(sd),
        vae_model_path=str(sd/'vae'),image_encoder_path=str(PROJECT/'models/clip'),device='cuda:0',
        width=384,height=512,texture_num_tokens=16,force_texture_num_tokens_override=False,
        texture_condition_mode='token',alpha1=2.,alpha2=2.,alpha3=1.5,alpha4=1.)
    pipe, extra = module.prepare(args)
    del extra
    for model in [pipe.unet,pipe.reference_unet,pipe.vae,pipe.text_encoder,pipe.image_encoder,pipe.bf_texture_conditioner]:
        model.eval().requires_grad_(False)
    pipe.set_layer_group_enabled(True)
    pipe._setup_layer_groups()
    pipe.set_progress_bar_config(disable=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return pipe


def freeze_digest(pipe):
    digest = hashlib.sha256()
    for model in [pipe.unet,pipe.reference_unet,pipe.vae,pipe.text_encoder,pipe.image_encoder,pipe.bf_texture_conditioner]:
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        assert not any(p.requires_grad for p in model.parameters())
    return digest.hexdigest()


def samples(image, points, side=128):
    import torch
    return torch.cat([image[...,y:y+side,x:x+side] for x,y in points])


def ref_samples(reference):
    return samples(reference, [(0,0),(128,0),(0,128),(128,128)])


def build_cache(pipe):
    import torch
    path = ROOT/'cache/training.pt'
    if path.exists():
        return torch.load(path, map_location='cpu')
    cached = []
    rows = read(ROOT/'inputs/train.json')
    with torch.no_grad():
        for row in rows:
            gt = image_tensor(row['gt']).cuda().half()*2-1
            sketch = image_tensor(row['sketch']).cuda().half()*2-1
            reference = image_tensor(row['reference'],(256,256)).cuda()
            ids = pipe.tokenizer(row['caption'],max_length=pipe.tokenizer.model_max_length,
                padding='max_length',truncation=True,return_tensors='pt').input_ids.cuda()
            text = pipe.text_encoder(ids)[0]
            # 与历史训练的 plain_resize 路径一致；新分支独立保留256方图局部特征。
            pil = Image.open(row['reference']).convert('RGB')
            clip = pipe.image_encoder(pipe.clip_image_processor(images=pil,return_tensors='pt').pixel_values.cuda().half(),output_hidden_states=True)
            texture = image_tensor(row['reference']).cuda().half()*2-1
            tokens,_ = pipe.bf_texture_conditioner(clip_image_embeds=clip.image_embeds,
                clip_vision_tokens=clip.hidden_states[-1][:,1:,:],texture_images=texture,texture_mode='patch_resampled')
            value = dict(latent=pipe.vae.encode(gt).latent_dist.mean*pipe.vae.config.scaling_factor,
                sketch_latent=pipe.vae.encode(sketch).latent_dist.mean*pipe.vae.config.scaling_factor,
                encoder=torch.cat([text,tokens],1),reference=reference,
                mask=image_tensor(row['mask'])[:,:1].cuda(),row=row)
            cached.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in value.items()})
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cached,path)
    return cached


def setup_sa(pipe, sketch_latent, timestep):
    import torch
    with torch.no_grad():
        pipe.reference_unet(sketch_latent,torch.zeros_like(timestep),None,return_dict=False)
    return {name:p.cache['hidden_states'] for name,p in pipe.reference_unet.attn_processors.items()
            if 'attn1' in name and hasattr(p,'cache')}


def audit():
    import torch
    from models.t3_local_texture import LocalTextureBranch, reference_consistency
    assert sha(BASE) == BASE_SHA
    protocol = read(ROOT/'protocol.json')
    pipe = load_pipe()
    before = freeze_digest(pipe)
    cached = build_cache(pipe)
    branch = LocalTextureBranch().cuda()
    branch.bind(pipe.unet)
    record = cached[0]
    latent, encoder = record['latent'].cuda(), record['encoder'].cuda()
    t = torch.tensor([100],device='cuda')
    sa = setup_sa(pipe,record['sketch_latent'].cuda(),t)
    branch.condition(record['reference'].cuda(),record['mask'].cuda())
    with torch.no_grad():
        off = pipe.unet(latent,t,encoder_hidden_states=encoder,cross_attention_kwargs={'sa_hidden_states':sa}).sample
        branch.enabled = True
        on = pipe.unet(latent,t,encoder_hidden_states=encoder,cross_attention_kwargs={'sa_hidden_states':sa}).sample
    assert torch.equal(off,on)
    predicted = pipe.unet(latent,t,encoder_hidden_states=encoder,cross_attention_kwargs={'sa_hidden_states':sa}).sample
    predicted.float().square().mean().backward()
    gradients = [float(b.output.weight.grad.float().norm()) for b in branch.blocks]
    assert all(np.isfinite(x) and x > 0 for x in gradients)
    branch.unbind()
    # 已知合成目标经同一VAE的可保留性；不读取候选生成结果选规则。
    controls = []
    from tools.t3_local_pilot_report import pattern_measure
    with torch.no_grad():
        for row in read(ROOT/'inputs/evaluation.json'):
            if row['condition'] != 'synthetic':
                continue
            target = image_tensor(row['gt']).cuda().half()*2-1
            reconstructed = pipe.vae.decode(pipe.vae.encode(target).latent_dist.mean).sample
            pixels = ((reconstructed[0].float().clamp(-1,1)+1)*127.5).permute(1,2,0).cpu().numpy().astype('uint8')
            controls.append(dict(id=row['id'],**pattern_measure(pixels,row)))
    positive = all(r['orientation_correct'] and r['period_relative_error'] < .2 for r in controls)
    write(ROOT/'audit/preflight.json',dict(zero_start_exact=True,output_gradient_norms=gradients,
        frozen_digest_before=before,frozen_digest_after=freeze_digest(pipe),vae_controls=controls,
        vae_controls_pass=positive,trainable_parameters=sum(p.numel() for p in branch.parameters()),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),protocol_sha256=sha(ROOT/'protocol.json')))
    assert positive, 'VAE合成正控未通过，停止训练'
    assert before == freeze_digest(pipe)
    print('PREFLIGHT_PASSED',flush=True)


def train(arm):
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler
    from models.t3_local_texture import LocalTextureBranch,reference_consistency
    assert read(ROOT/'audit/preflight.json')['vae_controls_pass']
    config = read(ROOT/'protocol.json')
    folder = ROOT/'training'/arm
    assert not (folder/'complete.json').exists()
    folder.mkdir(parents=True,exist_ok=True)
    pipe = load_pipe()
    pipe.set_scale(1.)
    pipe.set_ipa_scale(1.)
    before = freeze_digest(pipe)
    cached = build_cache(pipe)
    torch.manual_seed(config['seed'])
    branch = LocalTextureBranch().cuda()
    branch.bind(pipe.unet)
    optimizer = torch.optim.AdamW(branch.parameters(),lr=config['learning_rate'])
    scheduler = DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',num_train_timesteps=1000)
    rng = torch.Generator(device='cuda').manual_seed(config['seed'])
    order = np.random.default_rng(config['seed']).permutation(len(cached)).tolist()
    rows = []
    begin = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(config['updates']):
        optimizer.zero_grad(set_to_none=True)
        logs = []
        for micro in range(config['accumulation']):
            index = order[(step*config['accumulation']+micro)%len(order)]
            item = {k:v.cuda() if torch.is_tensor(v) else v for k,v in cached[index].items()}
            latent = item['latent']
            noise = torch.randn(latent.shape,generator=rng,device='cuda',dtype=latent.dtype)
            t = torch.randint(30,250,(1,),generator=rng,device='cuda')
            noisy = scheduler.add_noise(latent,noise,t)
            sa = setup_sa(pipe,item['sketch_latent'],t)
            branch.enabled = False
            with torch.no_grad():
                base = pipe.unet(noisy,t,encoder_hidden_states=item['encoder'],cross_attention_kwargs={'sa_hidden_states':sa}).sample
            branch.condition(item['reference'],item['mask'])
            branch.enabled = True
            predicted = pipe.unet(noisy,t,encoder_hidden_states=item['encoder'],cross_attention_kwargs={'sa_hidden_states':sa}).sample
            denoise = F.mse_loss(predicted.float(),noise.float())
            body = -F.max_pool2d(-item['mask'],11,1,5)
            exterior = F.interpolate(1-body,predicted.shape[-2:],mode='area')
            anchor = ((predicted.float()-base.float()).square()*exterior).sum()/(exterior.sum()*4).clamp_min(1)
            alpha = scheduler.alphas_cumprod.to('cuda')[t].reshape(1,1,1,1)
            x0 = (noisy.float()-(1-alpha).sqrt()*predicted.float())/alpha.sqrt()
            # decoder冻结，但不能用no_grad，监督需沿输入回传到新支路。
            decoded = (pipe.vae.decode((x0/pipe.vae.config.scaling_factor).half()).sample.float()+1)/2
            reference = reference_consistency(samples(decoded,item['row']['patches'][:2]),ref_samples(item['reference']))
            loss = denoise + config['anchor_weight']*anchor
            if arm == 'reference':
                loss = loss + config['reference_weight']*reference
            (loss/config['accumulation']).backward()
            logs.append(dict(loss=float(loss.detach()),denoise=float(denoise.detach()),anchor=float(anchor.detach()),
                reference=float(reference.detach()),id=item['row']['id'],t=int(t)))
            branch.clear()
        grad = float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1.))
        assert np.isfinite(grad) and all(np.isfinite(r['loss']) for r in logs)
        optimizer.step()
        record = dict(update=step+1,grad_norm=grad,seconds=time.monotonic()-begin,
            **{k:float(np.mean([r[k] for r in logs])) for k in ['loss','denoise','anchor','reference']},
            stream=stable([(r['id'],r['t']) for r in logs]))
        rows.append(record)
        if (step+1)%25 == 0:
            write(folder/'progress.json',record)
            print('TRAIN',arm,json.dumps(record),flush=True)
        if (step+1)%200 == 0:
            torch.save(branch.state_dict(),folder/('branch_%04d.pt'%(step+1)))
        assert time.monotonic()-begin < 1650, '单臂超过27.5分钟软上限'
    branch.unbind()
    after = freeze_digest(pipe)
    assert before == after
    torch.save(branch.state_dict(),folder/'branch.pt')
    write(folder/'log.json',rows)
    write(folder/'complete.json',dict(arm=arm,updates=len(rows),frozen_before=before,frozen_after=after,
        frozen_unchanged=True,branch_sha256=sha(folder/'branch.pt'),seconds=time.monotonic()-begin,
        peak_gib=torch.cuda.max_memory_allocated()/2**30,
        stream_sha256=stable([r['stream'] for r in rows])))
    print('TRAIN_COMPLETE',arm,flush=True)


def generate(arm):
    import torch
    from models.t3_local_texture import LocalTextureBranch
    pipe = load_pipe()
    branch = None
    if arm != 'T3':
        assert read(ROOT/'training'/arm/'complete.json')['frozen_unchanged']
        branch = LocalTextureBranch().cuda().eval()
        branch.load_state_dict(torch.load(ROOT/'training'/arm/'branch.pt',map_location='cuda'))
        branch.bind(pipe.unet)
        branch.enabled = True
    with torch.no_grad():
        for row in read(ROOT/'inputs/evaluation.json'):
            path = ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
            if path.exists() and path.with_suffix('.json').exists():
                assert read(path.with_suffix('.json'))['sha256'] == sha(path)
                continue
            torch.manual_seed(row['seed'])
            generator = torch.Generator(device='cuda').manual_seed(row['seed'])
            latent = torch.randn((1,4,64,48),generator=generator,device='cuda',dtype=torch.float16)
            initial = hashlib.sha256(latent.cpu().numpy().tobytes()).hexdigest()
            if branch is not None:
                branch.condition(image_tensor(row['reference'],(256,256)).cuda(),image_tensor(row['mask'])[:,:1].cuda())
            image = pipe(ref_image=image_tensor(row['sketch'])*2-1,prompt=row['caption'],
                texture_clip_image=Image.open(row['reference']).convert('RGB'),texture_embeds=None,
                null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,
                num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,generator=generator,
                latents=latent,num_inference_steps=50,texture_mode='patch_resampled',texture_num_tokens=16,
                texture_scale=1.,texture_condition_mode='token',fusion_type='minimal',texture_preprocess_mode='plain_resize',
                spatial_mask=None,debug_spatial=False,force_texture_num_tokens_override=False)[0]
            path.parent.mkdir(parents=True,exist_ok=True)
            image.save(path)
            write(path.with_suffix('.json'),dict(sha256=sha(path),initial_latent_sha256=initial,arm=arm,id=row['id'],
                reference_sha256=sha(row['reference']),GT_used_in_generation=False))
            if branch is not None:
                branch.clear()
            print('GENERATED',arm,row['condition'],row['id'],flush=True)
    if branch is not None:
        branch.unbind()
    write(ROOT/'audit'/('generated_'+arm+'.json'),dict(complete=True))


def package():
    import tarfile
    assert (ROOT/'decision.json').exists()
    paths = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['source','cache'])
             and p.suffix not in ['.pt','.bin','.tar','.gz']]
    write(ROOT/'checksums.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    output = ROOT/'review_results.tar.gz'
    with tarfile.open(output,'w:gz') as tar:
        for p in paths+[ROOT/'checksums.json']:
            tar.add(p,arcname=str(p.relative_to(ROOT)))
    write(ROOT/'review_results.sha256.json',dict(file=output.name,sha256=sha(output),bytes=output.stat().st_size))
    print('PACKAGED',output,output.stat().st_size,sha(output),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit','train','generate','evaluate','package'])
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='audit': audit()
    elif args.action=='train':
        assert args.arm in ['residual','reference']
        train(args.arm)
    elif args.action=='generate': generate(args.arm)
    elif args.action=='evaluate':
        from tools.t3_local_pilot_report import evaluate
        evaluate()
    else: package()
