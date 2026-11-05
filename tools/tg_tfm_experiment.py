"""TG-TFM 分阶段实验：冻结输入、工程预检、同预算筛选和结果归档。"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

from tools import bf_t3_frozen_core as core

PROJECT = Path('/share/home/u2515283058/Mymodel')
ROOT = PROJECT / 'output_eval/tg_tfm_e2_20261105'
HISTORY = PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101'
PARENT = PROJECT / 'output_eval/bf_t3_ablation_20261102'
START = HISTORY / 'training/BF-SD-T3/step28365.pt'
ARMS = ['T0', 'T1', 'T2', 'T3']
read, write, sha = core.read, core.write, core.sha
ATTRIBUTES = {
    'material': r'\b(silk|satin|velvet|denim|leather|cotton|linen|wool|knit(?:ted)?|lace|chiffon)\b',
    'decoration': r'\b(embroider\w*|sequins?|bead\w*|ruffl\w*|pleat\w*|bow|lace|fringe)\b',
    'pattern': r'\b(floral|strip\w*|plaid|check\w*|polka|paisley|geometric|print\w*)\b',
    'style': r'\b(casual|formal|elegant|vintage|sport\w*|bohemian)\b',
    'color': r'\b(black|white|red|blue|green|pink|yellow|purple|beige|brown|gray|grey)\b',
    'geometry': r'\b(sleeve\w*|collar\w*|neck\w*|waist\w*|length|fit\w*|leg)\b',
}


def stable(text):
    return hashlib.sha256(('TG_TFM_20261105/' + text).encode()).hexdigest()


def prepare():
    if (ROOT / 'protocol.json').exists():
        print('ALREADY_PREPARED', flush=True)
        return
    parent = read(HISTORY / 'protocol.json')
    assert sha(core.E1) == parent['e1_sha256']
    assert sha(START) == '06b9994862ec2cb0be5254c4d30acec2f459ec359e9c53856bc5413255f3fa74'
    init = HISTORY / 'initialization/bf_fusion.pt'
    assert sha(init) == parent['fusion_sha256']
    (ROOT / 'initialization').mkdir(parents=True, exist_ok=True)
    shutil.copy2(init, ROOT / 'initialization/bf_fusion.pt')
    train = read(HISTORY / 'training_manifest.json')
    assert len(train) == 45384
    write(ROOT / 'training_manifest.json', train)
    train_ids = {Path(row['cloth']).stem.split('_')[0] for row in train}
    final = read(PARENT / 'evaluation.json')
    excluded = train_ids | {row['id'].split('_')[0] for row in final}
    # 同时排除此前128件确认集；只声明对已检查清单无重叠。
    confirmation = PROJECT / 'output_eval/t3_reference_response_20261027/protocol/input_manifest.json'
    if confirmation.exists():
        excluded |= {row['id'].split('_')[0] for row in read(confirmation)}
    selected = []
    categories = dict(top='top', outwear='coat', pants='pair of pants', dress='dress')
    for category, noun in categories.items():
        base = PROJECT.parent / 'datasets/BF/test' / category
        candidates = []
        for path in (base / 'text').glob('*.txt'):
            if path.stem.split('_')[0] in excluded:
                continue
            caption = path.read_text(encoding='utf-8-sig').strip()
            attrs = {name: sorted(set(re.findall(pattern, caption.lower())))
                     for name, pattern in ATTRIBUTES.items()}
            if not caption or not any(attrs[name] for name in ('material','decoration','style')):
                continue
            paths = {key: base / folder / (path.stem + '.jpg')
                     for key, folder in [('sketch','sketch'),('reference','texture'),('gt','gt')]}
            if all(p.is_file() for p in paths.values()):
                candidates.append((stable(path.stem), path.stem, caption, attrs, paths))
        assert len(candidates) >= 8, (category, len(candidates))
        for _, ident, caption, attrs, paths in sorted(candidates)[:8]:
            row = dict(id=ident, category=category, condition='original', split='dev32',
                       caption=caption, attributes=attrs, seed=int(stable(ident)[:8],16) % 2**31)
            for key, source in paths.items():
                digest = sha(source)
                target = ROOT / 'inputs' / digest
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                row.update({key:str(target),key+'_sha256':digest,key+'_source':str(source)})
            selected.append(row)
            selected.append(dict(row,condition='neutral',caption='a photograph of a ' + noun + '.'))
    write(ROOT / 'dev_evaluation.json', selected)
    for row in final:
        assert row['id'].split('_')[0] not in train_ids
        for key in ('sketch','reference','gt'):
            source = Path(row[key])
            assert sha(source) == row[key+'_sha256']
            target = ROOT / 'inputs' / row[key+'_sha256']
            if not target.exists(): shutil.copy2(source,target)
            row[key] = str(target)
    write(ROOT / 'evaluation.json', final)
    coverage = {name: sum(bool(re.search(pattern, str(r['caption']).lower())) for r in train)
                for name,pattern in ATTRIBUTES.items()}
    write(ROOT / 'audit/text_coverage.json', dict(samples=len(train),keyword_coverage=coverage,
        method='regex counts, not verified semantic annotations',patterns=ATTRIBUTES))
    write(ROOT / 'protocol.json', dict(arms=ARMS,stage='A',screen_updates=1000,
        screen_initialization=str(START),screen_initialization_sha256=sha(START),
        formal_updates=28365,effective_batch=8,learning_rate=5e-5,warmup=500,
        seed=42,loss='unchanged full E2/T3',text_pooling='actual tokenizer first EOS hidden state',
        film=dict(rho=.1,hidden_dim=128,zero_last_layer=True),
        shape=dict(channels=[288,576,1152,1152],hw=[[64,48],[32,24],[16,12],[8,6]]),
        cfg='preserve original conditional-only BF/GAM; unconditioned branch cannot see positive text',
        dev_n=32,dev_excluded_from_final=True,development_product_overlap=[],
        text_evaluation='caption versus neutral with fixed sketch/reference/noise; weak caption labels; CLIP auxiliary only; AI review separate from human',
        formal_initialization='common E1 + trained original BF + zero channel projections',
        inference=dict(width=384,height=512,steps=50,cfg=7.,sketch_scale=.6,texture_tokens=False),
        source_commit=(ROOT / 'source_commit.txt').read_text().strip(),
        advance_rule='review paired development attribute response and visual constraints; do not automatically infer gain from larger image changes or CLIP alone',
        limitations=['single training seed','historical BF pretraining exposure unknown',
                     'caption labels unverified; no human ratings fabricated']))
    print('PREPARED',len(train),len(selected),len(final),coverage,flush=True)


def arguments(arm, folder):
    return ['--variant',arm,'--output-dir',str(ROOT/folder),
            '--initialization-root',str(ROOT),'--train-manifest',str(ROOT/'training_manifest.json')]


def execute(*args):
    subprocess.run([sys.executable,'-m','tools.train_tg_tfm']+list(args),check=True)


def preflight():
    if (ROOT/'audit/preflight_complete.json').exists():
        return
    import torch
    from types import SimpleNamespace
    from tools.train_tg_tfm import load_network
    from models.bf_sd_skip import BFSkipFusion
    core.ROOT = ROOT
    args=SimpleNamespace(device='cuda',variant='T3',start_checkpoint=str(START),output_dir=str(ROOT/'preflight'))
    pipe, network, _ = load_network(args,core)
    network.eval()
    row=read(ROOT/'dev_evaluation.json')[0]
    sketch,ref=core.tensor(row['sketch']).cuda(),core.tensor(row['reference']).cuda()
    ids=pipe.tokenizer(row['caption'],padding='max_length',truncation=True,max_length=77,return_tensors='pt').input_ids.cuda()
    old=BFSkipFusion().cuda()
    old.load_state_dict({k:v for k,v in network.adapter.state_dict().items() if '.text_modulators.' not in k},strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        text=pipe.text_encoder(ids)[0]
        latent=pipe.vae.encode(sketch).latent_dist.mean*pipe.vae.config.scaling_factor
        sa=core.sketch_states(pipe,latent.to(pipe.reference_unet.dtype),torch.tensor([100],device='cuda'))
        noisy=torch.randn_like(latent)
        actual=network(noisy,100,text,sketch,ref,sa,ids)
        features=[x.clone() for x in network.adapter.features]
        network.adapter.unbind()
        old.bind(pipe.unet)
        old.condition(sketch,ref)
        expected=pipe.unet(noisy,100,encoder_hidden_states=text,cross_attention_kwargs={'sa_hidden_states':sa}).sample
        feature_error=[float((a-b).abs().max()) for a,b in zip(features,old.features)]
        prediction_error=float((actual-expected).abs().max())
        assert feature_error == [0.,0.,0.,0.] and prediction_error==0.
        old.unbind()
    write(ROOT/'audit/real_zero_equivalence.json',dict(passed=True,feature_max_error=feature_error,
        prediction_max_error=prediction_error,precision='bf16 autocast',text_shape=list(text.shape),
        tokenizer_eos=int(pipe.tokenizer.eos_token_id),tokenizer_max_length=pipe.tokenizer.model_max_length))
    del old,network,pipe,actual,expected,features
    import gc
    gc.collect();torch.cuda.empty_cache()
    results={}
    for arm in ARMS:
        execute('train',*arguments(arm,'preflight'),'--start-checkpoint',str(START),
                '--max-steps','3','--warmup','0','--accumulation','8','--workers','0',
                '--gradient-audit','--checkpoint-every','3')
        execute('generate',*arguments(arm,'preflight'),'--checkpoint',str(ROOT/'preflight'/arm/'step00003.pt'),
                '--eval-manifest',str(ROOT/'dev_evaluation.json'),'--limit','2')
        results[arm]=read(ROOT/'preflight'/arm/'complete.json')
        if arm!='T0':
            grad=read(ROOT/'preflight'/arm/'gradients.json')
            assert sum(v for k,v in grad.items() if '.text_modulators.' in k)>0
    write(ROOT/'audit/preflight_complete.json',dict(passed=True,arms=results))


def train(arm,gpus):
    assert read(ROOT/'audit/preflight_complete.json')['passed']
    assert gpus in (1,2,4,8)
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node='+str(gpus),'-m','tools.train_tg_tfm','train',*arguments(arm,'screen'),
        '--start-checkpoint',str(START),'--max-steps','1000','--warmup','500',
        '--accumulation',str(8//gpus),'--checkpoint-every','500'],check=True)


def generate(arm):
    assert read(ROOT/'screen'/arm/'complete.json')['updates']==1000
    execute('generate',*arguments(arm,'screen'),'--checkpoint',str(ROOT/'screen'/arm/'step01000.pt'),
            '--eval-manifest',str(ROOT/'dev_evaluation.json'))


def evaluate():
    from tools.tg_tfm_report import evaluate_screen
    evaluate_screen()


def package():
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.png')
           and 'source' not in p.relative_to(ROOT).parts and 'preflight' not in p.relative_to(ROOT).parts
           and p not in (ROOT/'delivery_manifest.json',ROOT/'bundle_info.json')]
    paths += [p for p in (ROOT/'inputs').glob('*') if p.is_file()]
    manifest=[dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p)) for p in paths]
    write(ROOT/'delivery_manifest.json',manifest)
    paths.append(ROOT/'delivery_manifest.json')
    target=ROOT/'review_results.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for p in paths: archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    write(ROOT/'bundle_info.json',dict(file=str(target),size=target.stat().st_size,sha256=sha(target),members=len(paths)))
    print('PACKAGED',read(ROOT/'bundle_info.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preflight','train','generate','evaluate','package'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--gpus',type=int,default=1)
    args=parser.parse_args()
    if args.stage=='train': train(args.arm,args.gpus)
    elif args.stage=='generate': generate(args.arm)
    else: globals()[args.stage]()
