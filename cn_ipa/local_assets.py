"""只打包官方权重；实验代码经 GitHub 传输，不放入上传包。"""
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

OUT=Path('pretrained_upload'); OUT.mkdir(exist_ok=True)
CN_REV='8a158f547e031c5b8fbca19ead09a74767ff4db0'
IP_REV='018e402774aeeddd60609b4ecdb7e298259dc729'


def download(repo,rev,name,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): return
    temp=target.with_suffix(target.suffix+'.part')
    for attempt in range(10):
        try:
            url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true'
            offset=temp.stat().st_size if temp.exists() else 0
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=90) as response:
                mode='ab' if response.status==206 and offset else 'wb'
                with temp.open(mode) as stream:
                    while block:=response.read(8*1024*1024): stream.write(block)
            temp.replace(target); print('DOWNLOADED',name,target.stat().st_size,flush=True); return
        except Exception as error:
            print('RETRY',attempt,repr(error),flush=True); time.sleep(3)
    raise RuntimeError(name)


def main():
    stage=OUT/'weights'; stage.mkdir(exist_ok=True)
    download('lllyasviel/control_v11p_sd15_lineart',CN_REV,'config.json',stage/'controlnet/config.json')
    download('lllyasviel/control_v11p_sd15_lineart',CN_REV,'diffusion_pytorch_model.safetensors',stage/'controlnet/diffusion_pytorch_model.safetensors')
    # 下载小权重以核验本地缓存；正确的 ViT-H 模型在用户已有官方仓库中。
    download('h94/IP-Adapter',IP_REV,'models/ip-adapter_sd15.bin',stage/'ip_adapter/ip-adapter_sd15.bin')
    download('h94/IP-Adapter',IP_REV,'models/image_encoder/config.json',stage/'image_encoder/config.json')
    source=Path('F:/fuxian/ControlNet/IP-Adapter/models/image_encoder/model.safetensors')
    destination=stage/'image_encoder/model.safetensors'
    if not destination.exists(): shutil.copyfile(source,destination)
    provenance={'controlnet':dict(repo='lllyasviel/control_v11p_sd15_lineart',revision=CN_REV),
        'ipa':dict(repo='h94/IP-Adapter',revision=IP_REV),
        'image_encoder':dict(source=str(source),revision='既有官方缓存；以文件SHA及官方LFS SHA核验，不凭目录名认定')}
    hashes={}
    for p in stage.rglob('*'):
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            hashes[p.relative_to(stage).as_posix()]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
    (stage/'assets.json').write_text(json.dumps(dict(provenance=provenance,files=hashes),indent=2),encoding='utf-8')
    archive=OUT/'cnipa_weights.tar'
    if not archive.exists():
        with tarfile.open(archive,'w') as tar:
            for p in stage.rglob('*'):
                if p.is_file():tar.add(p,arcname=p.relative_to(stage).as_posix())
    with archive.open('rb') as f:
        index=0
        while block:=f.read(512*1024*1024):
            part=OUT/f'cnipa_weights.tar.part{index:02d}'
            if not part.exists():part.write_bytes(block)
            index+=1
    print('ASSETS_READY',index,hashes,flush=True)


if __name__=='__main__':main()
