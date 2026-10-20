"""科学输出冻结后，仅在 CPU 复算既有 PNG，诊断跨设备 LPIPS 数值差异。"""
import hashlib
import platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image, __version__ as pillow_version
from tools.m12_gain_experiment import RUN, ARMS, image, rows, CORE
from tools.m54_protocol import read, write, sha, commit
from tools.e33tmoc_appearance_eval import frozen_lpips


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    assert read(RUN/'audit/server_complete.json')['passed']
    protocol = read(RUN/'protocol/protocol.json')
    assert all(sha(path)==expected for path,expected in protocol['code_sha256'].items())
    records=read(RUN/'metrics/per_image.json')
    index={(r['variant'],r['id']):r for r in records}
    metric,information=frozen_lpips()
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None]
    results=[];inputs={}
    for row in rows():
        gt=Image.open(row['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        inputs[row['id']]=hashlib.sha256(gt.tobytes()).hexdigest()
        for arm in ARMS:
            path=image(arm,row['id'])
            meta=read(path.with_suffix('.json'))
            assert sha(path)==meta['png_sha256']
            value=float(metric(tensor(Image.open(path).convert('RGB')),tensor(gt)).mean())
            results.append(dict(id=row['id'],arm=arm,cpu_lpips=value,
                frozen_gpu_lpips=index[arm,row['id']]['lpips_gt'],
                cpu_minus_gpu=value-index[arm,row['id']]['lpips_gt']))
        print('CPU_PARITY_PROBE',row['id'],flush=True)
    write(RUN/'resources/precision_probe.json',dict(
        purpose='cross-device numerical diagnosis only; no new generation or scientific rescoring replacement',
        scientific_commit=protocol['implementation_commit'],probe_commit=commit(),
        probe_sha256=sha(__file__),python=platform.python_version(),pytorch=torch.__version__,
        pillow=pillow_version,cuda_build=torch.version.cuda,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        mkldnn_enabled=torch.backends.mkldnn.enabled,lpips=information,
        resized_gt_pixel_sha256=inputs,results=results,
        max_abs_cpu_minus_gpu=max(abs(r['cpu_minus_gpu']) for r in results),
        original_local_cpu_gpu_atol=protocol['local_verification']['lpips_cpu_atol'],
        tolerance_unchanged=True,new_scientific_images=0))
    print('CPU_PARITY_PROBE_COMPLETE',max(abs(r['cpu_minus_gpu']) for r in results),flush=True)


if __name__=='__main__':
    main()
