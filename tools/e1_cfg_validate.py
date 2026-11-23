"""固定第20轮权重，审计实际条件开关并隔离负提示词和视觉条件的影响。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from tools import e1_bfm_reference_region as shared

core = shared.core
PROJECT = core.PROJECT
ROOT = PROJECT/'output_eval/e1_cfg_validate_20261123'
PARENT = PROJECT/'output_eval/e1_bfm_continue20_20261121/epochs/epoch_20'
ARM = 'E1-BFM-T3-epoch20'
WEIGHT = PARENT/'training'/ARM/'final.pt'
WEIGHT_SHA = '292ade058ed8db056ce4d1f01b36e05b07c60935f73783586ca84398d729692a'
PROFILES = {
    ARM: dict(negative_prompt=' worst quality, low quality', match_visual=False),
    'E1-CFG-empty-off': dict(negative_prompt='', match_visual=False),
    'E1-CFG-empty-visual': dict(negative_prompt='', match_visual=True),
}
ARMS = ['BF-Fashion']+list(PROFILES)
read, write, sha, digest = core.read, core.write, core.sha, core.digest


class BranchAudit:
    """只在诊断上下文中修改负分支；退出后移除全部临时 hook。"""
    def __init__(self, network, match_visual=False):
        self.network = network
        self.match_visual = match_visual
        self.handles, self.events, self.reference_zero = [], [], []
        self.states = None
        self.current = None
        self.gam_calls = 0
        self.positive_prediction = None

    def before(self, module, args, kwargs):
        supplied = (kwargs.get('cross_attention_kwargs') or {}).get('sa_hidden_states')
        self.current = 'positive' if supplied is not None else 'negative'
        self.gam_calls = 0
        if self.current == 'positive':
            self.states = supplied
        elif self.match_visual:
            assert self.states is not None
            # prepend=True：必须先于 BFSkipFusion 的条件开关 hook 执行。
            kwargs = dict(kwargs, cross_attention_kwargs=dict(
                kwargs.get('cross_attention_kwargs') or {}, sa_hidden_states=self.states))
        return args, kwargs

    def attention(self, module, args, kwargs):
        if kwargs.get('sa_hidden_states') is not None:
            self.gam_calls += 1

    def reference(self, module, args):
        self.reference_zero.append(bool(args[1].count_nonzero().item() == 0))

    def after(self, module, args, kwargs, output):
        predicted = output[0] if isinstance(output, tuple) else output.sample
        event = dict(branch=self.current, bfm_injections=list(self.network.adapter.injection_counts),
            gam_active_attention_calls=self.gam_calls)
        assert bool(predicted.isfinite().all().item())
        if self.current == 'positive':
            self.positive_prediction = predicted.detach()
            if not self.events:
                # NumPy 不支持 BF16；仅转换审计副本，不改变网络输出或采样精度。
                event['first_prediction_sha256'] = digest(predicted.detach().float())
                event['prediction_source_dtype'] = str(predicted.dtype)
        else:
            event['positive_minus_negative_rms'] = float((
                self.positive_prediction.float()-predicted.float()).square().mean().sqrt())
        self.events.append(event)

    def __enter__(self):
        unet = self.network.unet
        self.handles.append(unet.register_forward_pre_hook(self.before, with_kwargs=True, prepend=True))
        self.handles.append(unet.register_forward_hook(self.after, with_kwargs=True))
        self.handles.append(self.network.adapter.fusion.register_forward_pre_hook(self.reference))
        for module in unet.modules():
            processor = getattr(module, 'processor', None)
            if hasattr(processor, 'to_k_ref') and hasattr(processor, 'to_v_ref'):
                self.handles.append(module.register_forward_pre_hook(self.attention, with_kwargs=True))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()

    def inference_summary(self):
        assert len(self.events) == 100
        positive, negative = self.events[::2], self.events[1::2]
        for event in positive:
            assert event['branch'] == 'positive' and event['bfm_injections'] == [3,2,2,3]
            assert event['gam_active_attention_calls'] > 0
        for event in negative:
            assert event['branch'] == 'negative'
            assert event['bfm_injections'] == ([3,2,2,3] if self.match_visual else [0]*4)
            assert (event['gam_active_attention_calls'] > 0) == self.match_visual
        assert self.reference_zero == [False]
        return dict(positive_calls=50, negative_calls=50, all_steps_verified=True,
            negative_keeps_bfm_and_gam=self.match_visual,
            first_positive_prediction_sha256=positive[0]['first_prediction_sha256'],
            first_positive_bfm_injections=positive[0]['bfm_injections'],
            first_negative_bfm_injections=negative[0]['bfm_injections'],
            positive_gam_attention_calls=positive[0]['gam_active_attention_calls'],
            negative_gam_attention_calls=negative[0]['gam_active_attention_calls'],
            mean_positive_minus_negative_rms=float(np.mean([
                event['positive_minus_negative_rms'] for event in negative])))


def prepare():
    assert sha(WEIGHT) == WEIGHT_SHA
    rows = read(PARENT/'evaluation.json')
    assert len(rows) == 192
    for row in rows:
        for key in ['sketch','reference','gt']:
            assert sha(row[key]) == row[key+'_sha256']
    write(ROOT/'evaluation.json', rows)
    for arm in ['BF-Fashion', ARM]:
        shared.copy_outputs(PARENT, ROOT, arm, rows)
    write(ROOT/'protocol.json', dict(checkpoint=str(WEIGHT), checkpoint_sha256=WEIGHT_SHA,
        training=False, optimizer_created=False, profiles=PROFILES, rows=192,
        inference=dict(size=[384,512],steps=50,cfg=7.,sketch_scale=.6,texture_tokens=False),
        primary_comparisons=['E1-CFG-empty-off minus '+ARM,
            'E1-CFG-empty-visual minus E1-CFG-empty-off'],
        source_commit=read(ROOT/'deployment.json')['source_commit'], source_sha256=sha(__file__),
        limitations=['inference intervention does not establish benefit of retraining unconditional branch',
            'visual intervention changes GAM and BFM together; cannot isolate them individually',
            'same historical captions and previously inspected test cases; exploratory diagnostic',
            'keeping visual conditions in both branches changes the meaning of guidance to text guidance']))
    print('PREPARED fixed-weight CFG diagnostic', flush=True)


def training_audit():
    import torch
    from train_GAM_texture_joint import JointTextureDataset, VGGGramStyleLoss
    pipe, network = shared.load_network('cuda', WEIGHT)
    network.eval().requires_grad_(False)
    data = JointTextureDataset(str(PROJECT/'output_eval/e1_bfm_continue20_20261121/training_manifest.json'),
        pipe.tokenizer,str(PROJECT.parent/'datasets/BF/training'),width=384,height=512,
        texture_preprocess_mode='plain_resize',mask_cache=str(core.MASKS))
    batch = {key:value[None].cuda() for key,value in data[0].items()}
    style = VGGGramStyleLoss().cuda().eval().requires_grad_(False)
    records = []
    for label, low, high in [('all',.3,1.),('text_dropped',0.,.2),
            ('reference_dropped',.2,.25),('text_reference_dropped',.25,.3)]:
        for seed in range(92000,93000):
            generator = torch.Generator(device='cuda').manual_seed(seed)
            torch.randn((1,4,64,48),device='cuda',generator=generator)
            probability = float(torch.rand((),device='cuda',generator=generator))
            if low <= probability < high:
                break
        else:
            raise AssertionError(label)
        with torch.inference_mode(), BranchAudit(network) as audit:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                losses, token = core.losses(pipe,network,batch,
                    torch.Generator(device='cuda').manual_seed(seed),style)
        assert low <= token['drop'] < high and len(audit.events) == 1
        event = audit.events[0]
        assert event['bfm_injections'] == [3,2,2,3] and event['gam_active_attention_calls'] > 0
        reference_zero = label in ['reference_dropped','text_reference_dropped']
        assert audit.reference_zero == [reference_zero]
        records.append(dict(condition=label,seed=seed,token=token,reference_input_zero=reference_zero,
            text_empty=label in ['text_dropped','text_reference_dropped'],**event))
        network.adapter.clear()
    write(ROOT/'audit/training_condition_runtime.json',dict(passed=True,records=records,
        real_historical_loss_function=True,training_manifest_index=0,weights_updated=False))
    print('TRAINING_CONDITIONS_VERIFIED',json.dumps(records),flush=True)


def generate(where, arm, rows, smoke=False):
    import torch
    profile = PROFILES[arm]
    pipe, network = shared.load_network('cuda', WEIGHT)
    assert str(pipe._execution_device) == 'cuda:0'
    network.eval().requires_grad_(False)
    pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    first_positive = None
    for row in rows:
        path = where/'outputs'/arm/row['condition']/(row['id']+'.png')
        if path.exists() and not smoke:
            meta = read(path.with_suffix('.json'))
            assert sha(path) == meta['output_sha256'] and meta['checkpoint_sha256'] == WEIGHT_SHA
            assert meta['profile'] == profile and meta['branch_audit']['all_steps_verified']
            continue
        with torch.inference_mode(), BranchAudit(network,profile['match_visual']) as audit:
            sketch = core.tensor(row['sketch']).cuda().half()
            network.adapter.condition(sketch,core.tensor(row['reference']).cuda().half())
            latent = torch.randn((1,4,64,48),device='cuda',dtype=torch.float16,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            assert digest(latent) == row['initial_latent_sha256']
            output = pipe(ref_image=sketch,prompt=row['caption'],texture_clip_image=None,texture_embeds=None,
                null_prompt='',negative_prompt=profile['negative_prompt'],width=384,height=512,
                num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']),
                latents=latent,num_inference_steps=50)[0]
        branch = audit.inference_summary()
        first_positive = branch['first_positive_prediction_sha256']
        path.parent.mkdir(parents=True,exist_ok=True);output.save(path)
        write(path.with_suffix('.json'),dict(id=row['id'],arm=arm,split=row['split'],condition=row['condition'],
            seed=row['seed'],initial_latent_sha256=row['initial_latent_sha256'],output_sha256=sha(path),
            checkpoint_sha256=WEIGHT_SHA,input_sketch_sha256=row['sketch_sha256'],
            input_reference_sha256=row['reference_sha256'],GT_used_in_inference=False,
            profile=profile,branch_audit=branch,steps=50,cfg=7.,native_size=[384,512]))
        network.adapter.clear()
        print('GENERATED',arm,row['id'],row['condition'],flush=True)
    write(where/('generation_complete_'+arm+'.json'),dict(complete=True,outputs=len(rows)))
    return first_positive


def smoke():
    from PIL import Image
    training_audit()
    rows = sorted([r for r in read(ROOT/'evaluation.json') if r['split']=='heldout128'],
        key=lambda r:r['sketch_sha256'])[:1]
    first = [generate(ROOT/'preflight',arm,rows,True) for arm in PROFILES]
    assert len(set(first)) == 1, 'first conditional prediction changed across profiles'
    row = rows[0]
    generated = ROOT/'preflight/outputs'/ARM/row['condition']/(row['id']+'.png')
    archived = ROOT/'outputs'/ARM/row['condition']/(row['id']+'.png')
    assert np.array_equal(np.asarray(Image.open(generated)),np.asarray(Image.open(archived))), 'baseline reproduction failed'
    write(ROOT/'audit/preflight_passed.json',dict(passed=True,source_sha256=sha(__file__),
        baseline_pixels_exact=True,first_conditional_prediction_identical=True,
        positive_and_negative_bfm_and_gam_hooks_verified=True,weights_updated=False))
    print('PREFLIGHT_PASSED',flush=True)


def formal():
    assert read(ROOT/'audit/preflight_passed.json')['source_sha256'] == sha(__file__)
    rows = read(ROOT/'evaluation.json')
    for arm in list(PROFILES)[1:]:
        generate(ROOT,arm,rows)


def evaluate():
    shared.evaluate(ROOT,ARMS,read(ROOT/'evaluation.json'),'heldout128')
    old = {(r['arm'],r['id'],r['condition']):r for r in read(PARENT/'metrics/per_image.json')}
    for row in read(ROOT/'metrics/per_image.json'):
        if row['arm'] in ['BF-Fashion',ARM]:
            assert row == old[row['arm'],row['id'],row['condition']]
    assert sha(WEIGHT) == WEIGHT_SHA
    write(ROOT/'complete.json',dict(complete=True,training=False,weights_updated=False,
        checkpoint_sha256=WEIGHT_SHA,training_condition_mismatch_runtime_verified=True,
        baseline_reproduction_passed=True,new_formal_outputs=384,rows_per_arm=192,
        arms=ARMS,baselines_exact=True,inference_intervention_is_not_training_causal_proof=True))
    files = [p for p in ROOT.rglob('*') if p.is_file() and 'source' not in p.relative_to(ROOT).parts
        and p.suffix in ['.json','.csv','.png','.log'] and p.name not in ['bundle_info.json','bundle_manifest.json']]
    write(ROOT/'bundle_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as archive:
        for path in files+[ROOT/'bundle_manifest.json']:
            archive.add(path,arcname=str(path.relative_to(ROOT)))
    write(ROOT/'bundle_info.json',dict(bytes=(ROOT/'review_results.tar.gz').stat().st_size,
        sha256=sha(ROOT/'review_results.tar.gz')))
    print('COMPLETED',json.dumps(read(ROOT/'complete.json')),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','smoke','formal','evaluate'])
    args = parser.parse_args()
    globals()[args.stage]()
