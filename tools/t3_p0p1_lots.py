"""官方LOTS小批量可行性测试；不读取参考纹理、不更改网络结构。"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from tools.t3_p0p1 import ROOT,SOURCE,read,write,sha


def retry(function,name):
    errors=[]
    for attempt in range(1,6):
        try:
            return function()
        except Exception as error:
            errors.append(dict(attempt=attempt,error=repr(error)))
            write(ROOT/'lots/asset_errors'/f'{name}.json',errors)
            print('LOTS_ASSET_RETRY',name,attempt,str(error)[:300],flush=True)
            if attempt==5:
                raise
            time.sleep(20)


def fetch():
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    os.environ['HF_HUB_DISABLE_XET']='1'
    target=ROOT/'lots';target.mkdir(parents=True,exist_ok=True)
    env=ROOT/'lots_env'
    if not (env/'bin/python').exists():
        subprocess.check_call([sys.executable,'-m','venv','--system-site-packages',str(env)])
    # 所有安装只写隔离venv，保留已有torch2.5.1，不更新其他实验环境。
    install=[str(env/'bin/python'),'-m','pip','install','diffusers==0.33.1','transformers==4.48.3',
             'huggingface-hub==0.28.1','accelerate==1.3.0','tokenizers==0.21.0','safetensors','sentencepiece']
    retry(lambda:subprocess.check_call(install),'environment')
    api=HfApi();assets={}
    specs={
      'adapter':('federicogirella/lots','1172dbc1aceefd121e84ea547ef39aa62ad994e1',['ckpts/lots/*']),
      'sdxl':('stabilityai/stable-diffusion-xl-base-1.0',None,
          ['model_index.json','scheduler/*','tokenizer/*','tokenizer_2/*','unet/config.json',
           'unet/diffusion_pytorch_model.fp16.safetensors','vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors',
           'text_encoder/config.json','text_encoder/model.fp16.safetensors','text_encoder_2/config.json','text_encoder_2/model.fp16.safetensors']),
      'dino':('facebook/dinov2-small',None,['config.json','preprocessor_config.json','model.safetensors']),
    }
    for name,(repo,revision,patterns) in specs.items():
        info=retry(lambda:api.model_info(repo,revision=revision),name+'_metadata')
        local=retry(lambda:snapshot_download(repo_id=repo,revision=info.sha,allow_patterns=patterns,
                    local_dir=str(target/'assets'/name),max_workers=2),name)
        files={str(path.relative_to(local)):dict(size=path.stat().st_size,sha256=sha(path))
               for path in Path(local).rglob('*') if path.is_file() and '.cache' not in path.parts}
        assets[name]=dict(repository=repo,revision=info.sha,path=local,files=files)
        write(target/'assets_progress.json',assets)
        print('LOTS_ASSET_READY',name,info.sha,len(files),flush=True)
    # 官方pipeline内部固定使用base版processor；仅缓存这个公共配置。
    processor=retry(lambda:hf_hub_download('facebook/dinov2-base','preprocessor_config.json'),'processor')
    assets['processor']=dict(path=processor,sha256=sha(processor))
    write(target/'assets.json',assets)
    print('LOTS_ALL_ASSETS_READY',flush=True)


def smoke():
    import hashlib
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from transformers import AutoModel
    sys.path.insert(0,str(SOURCE/'vendor/lots/src'))
    from lots.lots_pipeline import LOTSPipeline
    assets=read(ROOT/'lots/assets.json')
    adapter=Path(assets['adapter']['path'])/'ckpts/lots/lots.bin'
    base=StableDiffusionXLPipeline.from_pretrained(assets['sdxl']['path'],torch_dtype=torch.float16,
        variant='fp16',use_safetensors=True,add_watermarker=False,local_files_only=True)
    encoder=AutoModel.from_pretrained(assets['dino']['path'],local_files_only=True)
    pipe=LOTSPipeline(base,image_encoder=encoder,model_type='vits14',lots_ckpt=str(adapter),device='cuda',num_tokens=32)
    for module in [pipe.pipe.unet,pipe.pipe.vae,pipe.pipe.text_encoder,pipe.pipe.text_encoder_2,
                   pipe.image_encoder,pipe.image_proj_model,pipe.text_proj_model,pipe.pair_former]:
        module.eval().requires_grad_(False)
    rows=[r for r in read(ROOT/'protocol/input_manifest.json') if r['condition']=='original']
    rows=sorted(rows,key=lambda r:hashlib.sha256(('T3PP_LOTS/'+r['id']).encode()).hexdigest())[:4]
    results=[]
    for row in rows:
        seed=row['seed'];torch.manual_seed(seed);np.random.seed(seed)
        before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            # 保留官方原生global prompt和30步/0.8scale；仅把本实验草图和caption映射为单pair。
            image=pipe.generate(prompt='High quality photo of a model, artistic, 4k',
                pil_images=[Image.open(row['sketch']).convert('RGB')],descriptions=[row['caption']],
                num_samples=1,num_inference_steps=30,resolution=512,scale=.8,seed=seed)[0]
        path=ROOT/'lots/smoke/native'/(row['id']+'.png');path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        results.append(dict(id=row['id'],seed=seed,output=str(path),output_sha256=sha(path),
            sketch_sha256=row['sketch_sha256'],caption=row['caption'],reference_texture_used=False,GT_used=False,
            native_size=list(image.size),seconds=time.perf_counter()-before,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        write(ROOT/'lots/smoke/manifest.json',results)
        print('LOTS_SMOKE',row['id'],len(results),flush=True)
    write(ROOT/'lots/smoke/status.json',dict(images=4,network_changed=False,official_pipeline=True,
        adapter_sha256=sha(adapter),compatibility='pending actual image review; no expansion before task compatibility confirmed',
        reference_margin_eligible=False,training=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fetch','smoke'])
    args=parser.parse_args();globals()[args.action]()
