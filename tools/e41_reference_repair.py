"""主生成完成后修正真实旋转边界并重提取辅助参考特征；生成预算不变。"""
import shutil
import os
import subprocess
import numpy as np
from PIL import Image
from tools.e41_protocol import *
from tools.e41_runtime import Runner


def run():
    if (OUT/'audit/reference_repair_complete.json').exists():return
    assert all((OUT/('shard%d_complete.json'%i)).exists() for i in [0,1])
    rows=read(OUT/'references.json');before={str(p):sha(p) for p in (OUT/'generated').glob('*/Rplus.png')}
    assert len(before)==160
    backup=OUT/'audit/pre_rotation_features';backup.mkdir(exist_ok=True)
    for r in rows:
        if r['domain']!='real':continue
        path=OUT/'features'/(r['id']+'.npz')
        if not (backup/path.name).exists():shutil.copyfile(path,backup/path.name)
    repair_real_rotations();g=Runner(2)
    try:
        rows=[r for r in read(OUT/'references.json') if r['domain']=='real']
        for i,r in enumerate(rows):
            np.savez_compressed(OUT/'features'/(r['id']+'.npz'),**g.reference_features(Image.open(r['path']).convert('RGB')))
            if i%30==0:print('E41 REAL REFERENCE REPAIR',i,len(rows),flush=True)
    finally:g.close()
    assert all(sha(path)==h for path,h in before.items())
    write(OUT/'audit/reference_repair_complete.json',dict(complete=True,reference_views=len(rows),
        generated_pngs_unchanged=160,new_generations=0,job=os.environ.get('SLURM_JOB_ID'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


if __name__=='__main__':run()
