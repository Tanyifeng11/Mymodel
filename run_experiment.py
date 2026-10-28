"""官方LOTS小批量可行性测试；不读取参考纹理、不更改网络结构。"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'output/t3_p0_p1_20261028'
EXPERIMENT=PROJECT.parent/'Mymodel/output_eval/t3_p0_p1_20261028'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def retry(function,name):
    errors=[]
    for attempt in range(1,6):
        try:
            return function()
        except Exception as error:
            errors.append(dict(attempt=attempt,error=repr(error)))
            write(OUT/'asset_errors'/f'{name}.json',errors)
            print('LOTS_ASSET_RETRY',name,attempt,str(error)[:300],flush=True)
            if attempt==5:
                raise
            time.sleep(20)


def fetch():
    os.environ['HF_HUB_DISABLE_XET']='1'
    env=PROJECT/'.venv'
    if not (env/'bin/python').exists():
        subprocess.check_call([sys.executable,'-m','venv','--system-site-packages',str(env)])
    # 所有安装只写隔离venv，保留已有torch2.5.1，不更新其他实验环境。
    install=[str(env/'bin/python'),'-m','pip','install','--timeout','30','--retries','2',
             '--index-url','https://pypi.org/simple','diffusers==0.33.1','transformers==4.48.3',
             'huggingface-hub==0.28.1','accelerate==1.3.0','tokenizers==0.21.0','safetensors','sentencepiece']
    retry(lambda:subprocess.check_call(install),'environment')
    # 下载阶段也用同一隔离SDK，避免父环境huggingface_hub版本改变下载行为。
    subprocess.check_call([str(env/'bin/python'),str(__file__),'download'])


def download():
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    target=PROJECT
    api=HfApi(endpoint='https://huggingface.co');assets={}
    specs={
      'adapter':('federicogirella/lots','1172dbc1aceefd121e84ea547ef39aa62ad994e1',['ckpts/lots/*']),
      'sdxl':('stabilityai/stable-diffusion-xl-base-1.0',None,
          ['model_index.json','scheduler/*','tokenizer/*','tokenizer_2/*','unet/config.json',
           'unet/diffusion_pytorch_model.fp16.safetensors','vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors',
           'text_encoder/config.json','text_encoder/model.fp16.safetensors','text_encoder_2/config.json','text_encoder_2/model.fp16.safetensors']),
      'dino':('facebook/dinov2-small',None,['config.json','preprocessor_config.json','model.safetensors']),
    }
    for name,(repo,revision,patterns) in specs.items():
        if name=='adapter':
            # 已有官方冻结revision与文件清单，无需先访问可能超时的模型信息API。
            local=target/'assets'/name
            for filename in ['ckpts/lots/lots.bin','ckpts/lots/pair_former_config.json']:
                retry(lambda:hf_hub_download(repo,filename,revision=revision,local_dir=str(local)),
                      'adapter_'+Path(filename).name)
            files={str(path.relative_to(local)):dict(size=path.stat().st_size,sha256=sha(path))
                   for path in local.rglob('*') if path.is_file() and '.cache' not in path.parts}
            assets[name]=dict(repository=repo,revision=revision,path=str(local),files=files)
            write(target/'assets_progress.json',assets)
            print('LOTS_ASSET_READY',name,revision,len(files),flush=True)
            continue
        info=retry(lambda:api.model_info(repo,revision=revision,timeout=30),name+'_metadata')
        if name=='dino':
            available={item.rfilename for item in info.siblings}
            patterns=['config.json','preprocessor_config.json',
                      'model.safetensors' if 'model.safetensors' in available else 'pytorch_model.bin']
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
    sys.path.insert(0,str(PROJECT/'src'))
    from lots.lots_pipeline import LOTSPipeline
    assets=read(PROJECT/'assets.json')
    adapter=Path(assets['adapter']['path'])/'ckpts/lots/lots.bin'
    base=StableDiffusionXLPipeline.from_pretrained(assets['sdxl']['path'],torch_dtype=torch.float16,
        variant='fp16',use_safetensors=True,add_watermarker=False,local_files_only=True)
    encoder=AutoModel.from_pretrained(assets['dino']['path'],local_files_only=True)
    pipe=LOTSPipeline(base,image_encoder=encoder,model_type='vits14',lots_ckpt=str(adapter),device='cuda',num_tokens=32)
    for module in [pipe.pipe.unet,pipe.pipe.vae,pipe.pipe.text_encoder,pipe.pipe.text_encoder_2,
                   pipe.image_encoder,pipe.image_proj_model,pipe.text_proj_model,pipe.pair_former]:
        module.eval().requires_grad_(False)
    latent_info={};original_prepare=pipe.pipe.prepare_latents
    def capture_latents(*args,**kwargs):
        value=original_prepare(*args,**kwargs)
        latent_info.update(initial_latent_sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
                           latent_shape=list(value.shape),latent_dtype=str(value.dtype))
        return value
    pipe.pipe.prepare_latents=capture_latents
    rows=[r for r in read(EXPERIMENT/'protocol/input_manifest.json') if r['condition']=='original']
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
        path=OUT/'smoke/native'/(row['id']+'.png');path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        mirror=EXPERIMENT/'lots/smoke/native'/path.name;mirror.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,mirror)
        results.append(dict(id=row['id'],seed=seed,output=str(path),output_sha256=sha(path),
            sketch_sha256=row['sketch_sha256'],caption=row['caption'],reference_texture_used=False,GT_used=False,
            native_size=list(image.size),seconds=time.perf_counter()-before,peak_gib=torch.cuda.max_memory_allocated()/2**30,
            code_commit=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
            official_pipeline_sha256=sha(PROJECT/'src/lots/lots_pipeline.py'),
            adapter_sha256=assets['adapter']['files']['ckpts/lots/lots.bin']['sha256'],
            inference_steps=30,scale=.8,scheduler=pipe.pipe.scheduler.__class__.__name__,
            global_prompt='High quality photo of a model, artistic, 4k',**latent_info))
        write(OUT/'smoke/manifest.json',results)
        write(EXPERIMENT/'lots/smoke/manifest.json',results)
        print('LOTS_SMOKE',row['id'],len(results),flush=True)
    status=dict(images=4,network_changed=False,official_pipeline=True,
        adapter_sha256=sha(adapter),compatibility='pending actual image review; no expansion before task compatibility confirmed',
        reference_margin_eligible=False,training=0)
    write(OUT/'smoke/status.json',status)
    write(EXPERIMENT/'lots/smoke/status.json',status)


def import_assets():
    """导入用户授权本地下载、网页上传的权重包；源代码仍通过git pull传递。"""
    folder=PROJECT/'transfer';manifest=read(folder/'assets_bundle_manifest.json')
    assert manifest['source_code_included'] is False
    archive=folder/'lots_assets.tar'
    with archive.open('wb') as out:
        for row in manifest['parts']:
            path=folder/row['name']
            assert path.name==row['name'] and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            with path.open('rb') as source:shutil.copyfileobj(source,out,8*1024*1024)
            print('IMPORT_PART_VERIFIED',path.name,flush=True)
    assert archive.stat().st_size==manifest['archive_bytes'] and sha(archive)==manifest['archive_sha256']
    expected={row['path']:row for row in manifest['files']}
    with tarfile.open(archive,'r:') as bundle:
        members=bundle.getmembers()
        assert len(members)==len(expected)
        for member in members:
            relative=Path(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            assert relative.parts[0]=='assets' and member.name in expected
            target=PROJECT/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out,8*1024*1024)
            row=expected[member.name]
            assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
            print('IMPORT_FILE_VERIFIED',member.name,flush=True)
    assets=manifest['assets']
    for name,item in assets.items():item['path']=str(PROJECT/item['path'])
    # 官方pipeline固定用该公共processor ID，填充标准离线Hub缓存，不改官方网络代码。
    processor=assets['processor'];source=Path(processor['path'])/'preprocessor_config.json'
    cache=Path(os.environ.get('HF_HOME',str(PROJECT/'cache/huggingface')))/'hub/models--facebook--dinov2-base'
    snapshot=cache/'snapshots'/processor['revision']/'preprocessor_config.json';snapshot.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,snapshot);(cache/'refs').mkdir(exist_ok=True);(cache/'refs/main').write_text(processor['revision'])
    from huggingface_hub import try_to_load_from_cache
    cached=try_to_load_from_cache('facebook/dinov2-base','preprocessor_config.json',cache_dir=str(cache.parent))
    assert isinstance(cached,str) and sha(cached)==sha(source)
    assets['processor'].update(path=cached,sha256=sha(source))
    write(PROJECT/'assets.json',assets)
    write(OUT/'asset_transfer_verification.json',dict(archive_sha256=manifest['archive_sha256'],
        parts=len(manifest['parts']),files=len(expected),all_file_hashes_match=True,
        source='official pinned assets downloaded locally and uploaded via school portal; no source code in package'))
    print('IMPORTED_OFFICIAL_ASSETS_COMPLETE',len(expected),flush=True)


def environment():
    import torch,transformers,diffusers,huggingface_hub
    sys.path.insert(0,str(PROJECT/'src'))
    from lots.lots_pipeline import LOTSPipeline
    info=dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,
        diffusers=diffusers.__version__,huggingface_hub=huggingface_hub.__version__,
        official_pipeline_imported=LOTSPipeline.__name__,cuda_available=torch.cuda.is_available())
    write(OUT/'runtime_environment.json',info);print('ENVIRONMENT_READY',info,flush=True)


def package_results():
    if (PROJECT/'assets.json').is_file():shutil.copy2(PROJECT/'assets.json',OUT/'assets_provenance.json')
    code=OUT/'code';code.mkdir(parents=True,exist_ok=True)
    tracked=subprocess.check_output(['git','-C',str(PROJECT),'ls-files'],text=True).splitlines()
    for name in tracked:
        source=PROJECT/name
        if source.is_file():
            target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for source in (PROJECT/'logs').glob('*.log'):
        target=OUT/'logs'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.gz','.tgz'] and p.name not in ['bundle_manifest.json','bundle_info.json']]
    write(OUT/'bundle_manifest.json',[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in paths])
    archive=PROJECT/'transfer/smoke_review.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for path in paths+[OUT/'bundle_manifest.json']:bundle.add(path,arcname=str(path.relative_to(OUT)),recursive=False)
    write(PROJECT/'transfer/smoke_bundle_info.json',dict(sha256=sha(archive),bytes=archive.stat().st_size,members=len(paths)+1,weights_included=False))
    print('LOTS_RESULT_BUNDLE',read(PROJECT/'transfer/smoke_bundle_info.json'),flush=True)


def audit():
    """保存真实准备状态与失败证据，不把下载失败称为模型任务不兼容。"""
    mirror=EXPERIMENT/'lots'
    errors={p.name:read(p) for p in sorted((OUT/'asset_errors').glob('*.json'))}
    assets=read(PROJECT/'assets.json') if (PROJECT/'assets.json').is_file() else None
    smoke_status=read(OUT/'smoke/status.json') if (OUT/'smoke/status.json').is_file() else None
    status=dict(upstream=read(PROJECT/'UPSTREAM.json'),
        reproduction_commit=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
        project_server=str(PROJECT),project_local='F:/fuxian/LOTS',
        state='SMOKE_AWAITS_TASK_REVIEW' if smoke_status else 'ASSETS_READY_SMOKE_NOT_RUN' if assets else 'NOT_REPRODUCED_ASSET_DOWNLOAD_FAILED',
        official_assets_complete=bool(assets),actual_smoke_images=smoke_status.get('images',0) if smoke_status else 0,
        task_compatibility='not assessed without outputs',reference_margin_eligible=False,
        network_changed=False,training=0,errors=errors,
        limitation='asset access failure is an infrastructure result; it does not show that LOTS is incompatible or inferior')
    write(OUT/'candidate_status.json',status);write(mirror/'candidate_status.json',status)
    for name in ['run_experiment.py','run_experiment.sbatch','UPSTREAM.json','LICENSE']:
        destination=mirror/'code'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/name,destination)
    for folder,source in [('logs',PROJECT/'logs'),('asset_errors',OUT/'asset_errors')]:
        for path in source.glob('*'):
            if path.is_file():
                destination=mirror/folder/path.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    if assets:write(mirror/'assets_provenance.json',assets)
    print('LOTS_AUDIT',status['state'],status['actual_smoke_images'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fetch','download','smoke','audit','import_assets','environment','package_results'])
    args=parser.parse_args();globals()[args.action]()
