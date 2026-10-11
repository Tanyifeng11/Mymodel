"""统一两分片颜色累加精度；保留原始 NPZ，验证其他表示和全部图像不变。"""
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from tools.e40_protocol import OUT, OLD, read, write, sha
from tools.e40_runtime import appearance, eval_mask


def load(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}


def run():
    marker=OUT/'audit/color_precision_repair.json'
    if marker.exists() and read(marker)['complete']:return
    previous=OUT/'audit/pre_color_repair_comparison.json'
    if not previous.exists():shutil.copyfile(OUT/'E40_pattern_representation_comparison.json',previous)
    tasks={};images={}
    for r in read(OUT/'references.json'):
        tasks[OUT/'features'/('ref_'+r['id']+'_'+r['variant']+'.npz')]=(Path(r['path']),None)
    for r in read(OUT/'generation.json'):
        folder=OUT/'generated'/r['id'];png=folder/'image.png';images[str(png)]=sha(png)
        mask,valid,_=eval_mask(r,Image.open(png).size)
        tasks[folder/'features.npz']=(png,mask)
        tasks[Path(read(folder/'evaluation.json')['reference_features'])]=(Path(r['reference']),None)
    for r in read(OLD/'pairs.json'):
        source=OLD/'cases'/r['id'];dst=OUT/'e39_reuse'/r['id']
        mask,valid,_=eval_mask(r,Image.open(source/'Rplus.png').size)
        for arm in read(source/'complete.json')['arms']:
            png=source/(arm+'.png');images[str(png)]=sha(png)
            tasks[dst/(arm+'_generated.npz')]=(png,mask)
            if arm!='Rzero':tasks[dst/(arm+'_reference.npz')]=(source/(arm+'_reference.png'),None)
    audit=[]
    for path,(image,mask) in tasks.items():
        backup=OUT/'audit/pre_color_repair_features'/path.relative_to(OUT)
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copyfile(path,backup)
        old=load(backup);data=load(path);colors=appearance(Image.open(image),mask)
        changed=any(not np.array_equal(old[k],colors[k]) for k in colors)
        data.update(colors);np.savez_compressed(path,**data)
        after=load(path)
        assert all(np.array_equal(v,after[k]) for k,v in old.items() if k not in colors)
        audit.append(dict(path=str(path.relative_to(OUT)),before_sha256=sha(backup),after_sha256=sha(path),
            numerical_color_change=changed,max_LAB_mean_change=float(np.max(abs(old['lab_mean']-after['lab_mean'])))))
    assert all(sha(path)==value for path,value in images.items())
    write(marker,dict(complete=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cause='shard0 began at 9db6c4f before float64 color summation repair; shard1 began at eb161b7',
        feature_files=len(tasks),changed_files=sum(r['numerical_color_change'] for r in audit),
        non_color_arrays_bitwise_unchanged=True,generated_and_reused_images_unchanged=len(images),
        training_updates=0,new_generations=0,records=audit))
    print('E40 COLOR PRECISION REPAIRED',len(tasks),flush=True)


if __name__=='__main__':run()
