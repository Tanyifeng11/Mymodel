"""补齐旧C0/C1审计状态；验证旧PNG，不改变生成结果或评测。"""
import cv2
import numpy as np
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from models.e33tm_generation_wrapper import spatial_carrier
from data.e32_target_pseudogt import image_at, masks
from data.e33rf_real_rotation_dataset import rotate
from tools.e33tmoc_protocol import *

def main():
    cohort=prepare(); ids=read(OUT/'splits/diagnostic64.json')
    locked=freeze_carrier_inputs(cohort,ids)
    for row in cohort:
        sid=row['id']
        if sid not in ids: continue
        reference=image_at(DATASET/row['reference']); sketch=image_at(DATASET/row['sketch'])
        mask=Image.fromarray(masks(sketch)[0].astype(np.uint8)*255)
        with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:
            fields=z['orientation'][:3].copy(); confidences=z['confidence'][:3].copy()
        for ai,(arm,ref) in enumerate(zip(('R0','R90','R180'),(reference,rotate(reference,90),rotate(reference,180)))):
            fg=np.asarray(estimate_cloth_foreground_mask(ref,*ref.size)[0])>127
            carrier,coords=spatial_carrier(ref,fields[ai],mask)
            for name in ('C0_current','C1_analytic'):
                dest=OUT/'candidates'/name/'seed42'/sid
                with np.load(dest/(arm+'_inputs.npz')) as z: inputs={k:z[k].copy() for k in z.files}
                inputs['confidence']=confidences[ai]
                np.savez_compressed(dest/(arm+'_inputs.npz'),**inputs)
                state=dict(RF_orientation=fields[ai],RF_confidence=confidences[ai],target_support=np.asarray(mask)>0)
                if name=='C0_current':
                    assert np.array_equal(np.asarray(carrier),np.asarray(Image.open(dest/'S2'/(arm+'.png'))))
                    state.update(source_foreground=fg,source_valid_support=fg,UV=coords.uv,
                        sampled_source_foreground=cv2.remap(fg.astype(np.uint8),coords.uv[...,0],coords.uv[...,1],
                            cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)>0)
                np.savez_compressed(dest/(arm+'_construction.npz'),**state)
                write(dest/(arm+'_construction_metadata.json'),dict(
                    method=name,selected_source_patches=None,
                    source_support_usage='informational foreground only; C0 remap does not enforce source support' if name=='C0_current' else 'N/A: analytic diagnostic has no reference sampling',
                    source_patch_usage='N/A: C0 uses dense UV sampling' if name=='C0_current' else 'N/A: fixed procedural cosine',
                    result_images_modified=False,git_commit=commit()))
    assert {p:sha(p) for p in locked}==locked
    frozen_check();print('[OC C0/C1 artifacts complete; PNG unchanged]',flush=True)

if __name__=='__main__':main()
