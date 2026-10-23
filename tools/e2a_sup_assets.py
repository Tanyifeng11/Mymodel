"""从官方来源准备冻结依赖；失败保留文件并由下次跟进重试。"""
import subprocess
import shutil
import urllib.parse
import urllib.request
import torch
from tools.e2a_sup_common import *


def official_checkout(repo,url):
    """空 clone 可能成功退出；仅从同一官方 origin 恢复 HEAD，保留原目录。"""
    if not (repo/'.git').exists():
        subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True,timeout=180)
    origin=subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip()
    assert origin==url,('OFFICIAL_ORIGIN_CHANGED',origin)
    head=subprocess.run(['git','-C',str(repo),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
    if head.returncode:
        subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin','HEAD'],check=True,timeout=180)
        subprocess.run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'],check=True,timeout=30)
    return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()


def official_download(url,path,hash_prefix=None):
    """沿已配置的 Git 出口下载官方权重；验证后才发布缓存，不改全局代理。"""
    parsed=urllib.parse.urlsplit(url)
    assert parsed.scheme=='https' and parsed.hostname in ['download.pytorch.org','dl.fbaipublicfiles.com']
    path=Path(path)
    if path.exists():
        assert path.stat().st_size>0,('EMPTY_CACHED_WEIGHT',str(path))
        if hash_prefix:assert sha(path).startswith(hash_prefix),('CACHED_WEIGHT_SHA_MISMATCH',str(path))
        return
    proxy=subprocess.run(['git','config','--get','http.proxy'],capture_output=True,text=True).stdout.strip()
    handler=urllib.request.ProxyHandler({'http':proxy,'https':proxy}) if proxy else urllib.request.ProxyHandler()
    opener=urllib.request.build_opener(handler)
    print('OFFICIAL_DOWNLOAD_PROXY',urllib.parse.urlsplit(proxy).hostname if proxy else 'environment/default',flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,4):
        temporary=path.with_name(path.name+'.partial_%s_%d'%(os.environ.get('SLURM_JOB_ID','local'),attempt))
        assert not temporary.exists(),('PARTIAL_DOWNLOAD_PRESERVED',str(temporary))
        print('OFFICIAL_DOWNLOAD_ATTEMPT',attempt,url,flush=True)
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'torch.hub'})
            with opener.open(request,timeout=60) as source,temporary.open('wb') as target:
                shutil.copyfileobj(source,target,length=1024*1024)
                assert target.tell()>0,'EMPTY_OFFICIAL_WEIGHT'
                expected=source.headers.get('Content-Length')
                assert expected is None or target.tell()==int(expected),'INCOMPLETE_OFFICIAL_WEIGHT'
        except OSError as error:
            print('OFFICIAL_DOWNLOAD_FAILED',attempt,repr(error),flush=True)
            if attempt==3:raise
            continue
        digest=sha(temporary)
        if hash_prefix:assert digest.startswith(hash_prefix),('OFFICIAL_WEIGHT_SHA_MISMATCH',str(temporary))
        temporary.replace(path)
        print('OFFICIAL_DOWNLOAD_VERIFIED',path.name,digest,flush=True)
        return


def assets(arm):
    root=ROOT/'assets';root.mkdir(parents=True,exist_ok=True)
    registry=read(root/'registry.json') if (root/'registry.json').exists() else {}
    if arm in registry:
        assert all(sha(p)==digest for p,digest in registry[arm]['files'].items())
        return
    name,url=('DISTS','https://github.com/dingkeyan93/DISTS.git') if arm=='T1' else ('dino','https://github.com/facebookresearch/dino.git')
    repo=root/name
    code_commit=official_checkout(repo,url)
    record=dict(source=url,commit=code_commit,license=str(repo/'LICENSE'),pretraining_data_exposure_unknown=True,
        preparation_source_commit=commit())
    files=list(repo.glob('*.py'))+list(repo.glob('LICENSE*'))
    if arm=='T1':
        from torchvision.models import vgg16,VGG16_Weights
        vgg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
        official_download(VGG16_Weights.IMAGENET1K_V1.url,vgg,'397923af')
        vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        assert sha(vgg).startswith('397923af')
        candidates=list(repo.rglob('weights.pt'));assert candidates, 'OFFICIAL_DISTS_WEIGHTS_NOT_FOUND'
        path=candidates[0];record.update(alpha_beta=str(path),backbone=str(vgg),
            backbone_source=VGG16_Weights.IMAGENET1K_V1.url,license_name='MIT')
        files+=list((repo/'DISTS_pytorch').glob('*.py'))+[path,vgg]
    else:
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
        path=root/'dino_vits16.pth'
        official_download(url,path)
        record.update(weights=str(path),weights_source=url,license_name='Apache-2.0',
            architecture='DINO ViT-S/16 official; strict load; last CLS and block6 patch keys')
        files.append(path)
    record['files']={str(p):sha(p) for p in files}
    registry[arm]=record;write(root/'registry.json',registry)
    print('OFFICIAL_ASSETS_FROZEN',arm,code_commit,flush=True)


if __name__=='__main__':
    import sys
    assets(sys.argv[1])
