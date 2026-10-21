"""从官方来源准备冻结依赖；失败保留文件并由下次跟进重试。"""
import subprocess
import urllib.request
import torch
from tools.e2a_sup_common import *


def assets(arm):
    root=ROOT/'assets';root.mkdir(parents=True,exist_ok=True)
    registry=read(root/'registry.json') if (root/'registry.json').exists() else {}
    if arm in registry:
        assert all(sha(p)==digest for p,digest in registry[arm]['files'].items())
        return
    name,url=('DISTS','https://github.com/dingkeyan93/DISTS.git') if arm=='T1' else ('dino','https://github.com/facebookresearch/dino.git')
    repo=root/name
    if not (repo/'.git').exists(): subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True,timeout=180)
    code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    record=dict(source=url,commit=code_commit,license=str(repo/'LICENSE'),pretraining_data_exposure_unknown=True)
    files=list(repo.glob('*.py'))+list(repo.glob('LICENSE*'))
    if arm=='T1':
        from torchvision.models import vgg16,VGG16_Weights
        vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
        assert sha(vgg).startswith('397923af')
        candidates=list(repo.rglob('weights.pt'));assert candidates, 'OFFICIAL_DISTS_WEIGHTS_NOT_FOUND'
        path=candidates[0];record.update(alpha_beta=str(path),backbone=str(vgg),license_name='MIT')
        files+=list((repo/'DISTS_pytorch').glob('*.py'))+[path,vgg]
    else:
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
        path=root/'dino_vits16.pth'
        if not path.exists():
            temporary=path.with_suffix('.partial')
            with urllib.request.urlopen(url,timeout=120) as source,temporary.open('wb') as target:
                import shutil
                shutil.copyfileobj(source,target)
            temporary.replace(path)
        record.update(weights=str(path),weights_source=url,license_name='Apache-2.0',
            architecture='DINO ViT-S/16 official; strict load; last CLS and block6 patch keys')
        files.append(path)
    record['files']={str(p):sha(p) for p in files}
    registry[arm]=record;write(root/'registry.json',registry)
    print('OFFICIAL_ASSETS_FROZEN',arm,code_commit,flush=True)


if __name__=='__main__':
    import sys
    assets(sys.argv[1])
