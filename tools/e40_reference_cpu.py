"""GPU 排队期间先做设备无关的 FFT/颜色参考检验，不能代替生成实验。"""
import numpy as np
from PIL import Image
import cv2
from tools.e40_protocol import OUT, read, write
from tools.e40_runtime import patches, spectrum
from tools import e40_analyze as analysis


def run():
    rows=read(OUT/'references.json');features={}
    for r in rows:
        image=Image.open(r['path']).convert('RGB');a=np.asarray(image)
        lab=cv2.cvtColor(a.astype(np.float32)/255,cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float64)
        hist=np.histogramdd(a.reshape(-1,3)/255,bins=4,range=((0,1),)*3)[0].ravel()
        fft,valid=spectrum(patches(image))
        color=np.concatenate([lab.mean(0)/100,lab.std(0)/100,hist/hist.sum()]).astype(np.float32)
        features[(r['id'],r['variant'])]=dict(fft=fft,fft_valid=valid,color=color,mask_valid=True)
    analysis.REPS=['fft','color']
    classification,_=analysis.classify(rows,features)
    reference=analysis.reference_tests(rows,features)
    write(OUT/'cpu_reference_analysis.json',dict(classification=classification,reference_tests=reference,
        input_color_drift=analysis.input_drift(rows),status='reference FFT and color only; generation still pending'))
    print('E40 CPU REFERENCE COMPLETE',flush=True)


if __name__=='__main__':run()
