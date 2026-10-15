"""用训练域局部遮挡验证 Writer token 来源；失败时禁止伪造 oracle。"""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e52_protocol import *
from tools.e52_runtime import runtime,release

def run():
    from tools.e45_eval import foreground_at
    assert read(OUT/'audit/p0_complete.json')['passed']
    path=OUT/'audit/token_provenance.json'
    if path.exists():return read(path)
    r=runtime('audit/provenance/runtime');records=[];entropies=[]
    try:
        for row in read(OUT/'splits/calibration.json'):
            image=Image.open(row['reference']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            pixels=to_tensor(image)[None].to(r.pipe.device)*2-1
            foreground=foreground_at(image,18,235)
            usable=cv2.resize(foreground.astype('float32'),(16,16),interpolation=cv2.INTER_AREA).ravel()>=.5
            points=np.stack(np.meshgrid((np.arange(16)+.5)/16,(np.arange(16)+.5)/16),-1).reshape(-1,2)
            centers=np.array([[(x+.5)/4,(y+.5)/4] for y in range(4) for x in range(4)])
            responses=[];base_features=[]
            for t in CONFIG['provenance']['timesteps']:
                r.writer.set_reference(pixels,r.empty_text);base=r.writer.write(torch.tensor(t,device=r.pipe.device)).float()
                base_features.append(base.cpu());changes=[]
                for y in range(4):
                    for x in range(4):
                        changed=pixels.clone();roi=changed[:,:,y*128:(y+1)*128,x*96:(x+1)*96]
                        roi[:]=pixels[:,:,y*128:(y+1)*128,x*96:(x+1)*96].mean((2,3),keepdim=True)
                        r.writer.set_reference(changed,r.empty_text)
                        feature=r.writer.write(torch.tensor(t,device=r.pipe.device)).float()
                        changes.append((feature-base).square().mean(-1).sqrt()[0].cpu().numpy())
                response=np.stack(changes,1);responses.append(response)
            response=np.stack(responses);mass=response/np.maximum(response.sum(-1,keepdims=True),1e-12)
            top=mass.argmax(-1);distance=np.linalg.norm(centers[top]-points[None],axis=-1)*16
            local_mass=mass.max(-1)
            stable=(distance<=CONFIG['provenance']['tolerance_cells']).all(0)&(local_mass>=.15).all(0)
            n=int(usable.sum());fraction=float(stable[usable].mean()) if n else 0.
            records.append(dict(id=row['id'],usable_tokens=n,reproducible_tokens=int((stable&usable).sum()),
                valid_fraction=fraction,mean_distance_cells=float(distance[:,usable].mean()) if n else None,
                response_shape=list(response.shape),centers=centers.tolist(),
                response_center=centers[top].tolist(),distance_cells=distance.tolist(),
                local_mass=local_mass.tolist(),valid_tokens=(stable&usable).tolist()))
            torch.save(dict(response=torch.from_numpy(response),base_features=base_features),OUT/'audit'/(row['id']+'_provenance.pt'))
            print('E52 TOKEN',row['id'],fraction,flush=True)
        valid=sum(x['reproducible_tokens'] for x in records)/max(sum(x['usable_tokens'] for x in records),1)
        positions=np.mean([np.asarray(x['valid_tokens']) for x in records],axis=0)>=.70
        result=dict(oracle_valid=valid>=.70,valid_fraction=valid,threshold=.70,records=records,position_valid=positions.tolist(),
            spatial_pooling='verified source code adaptive_avg_pool2d(16,16); not sufficient alone',
            local_origin='4x4 controlled local mean occlusion, 3 timesteps, <=2 cells and >=15% response mass at all times',
            status='PASS' if valid>=.70 else 'TOKEN_ORACLE_NOT_VALID',mapping_sigma=None,
            intervention_temperature_calibration='not_run',code_commit=commit())
        # 空间来源合格后仍须在 pilot 校准映射熵，不能在确认集调温度。
        if valid>=.70:
            from tools.e52_runtime import case
            from tools.e52_generate import mapping,pooled_uv
            entropy=[]
            for row in read(OUT/'splits/calibration.json'):
                value=case(r,row,'B1','calibration',diagnose=True)
                entropy.extend(h for s in value['steps'] if 'softmax' in s for h in s['softmax']['reference_entropy_by_head'])
            target=float(np.mean(entropy));temperatures=[]
            for sigma in [.025,.05,.075,.10,.15,.25,.5,1.,2.]:
                values=[];distances=[];wrong_distances=[];coverages=[]
                for row in read(OUT/'splits/pilot.json'):
                    m=mapping(row,'ORACLE',sigma,positions).numpy();valid_rows=m.sum(-1)>0
                    values.extend((-(m*np.log(np.maximum(m,1e-20))).sum(-1))[valid_rows].tolist())
                    data=np.load(row['geometry']);tu,_,tv=pooled_uv(data['target_uv'],data['target_labels'],(24,32))
                    ru,_,_=pooled_uv(data['reference_uv0'],data['target_labels'],(16,16))
                    distance=np.linalg.norm(tu[:,None]-ru[None],axis=-1)
                    wrong=mapping(row,'WRONG',sigma,positions).numpy()
                    distances.extend((m*distance).sum(-1)[valid_rows].tolist())
                    wrong_distances.extend((wrong*distance).sum(-1)[valid_rows].tolist())
                    coverages.append(float(valid_rows.sum())/max(int(tv.sum()),1))
                error=float(np.mean(distances));gap=float(np.mean(wrong_distances))-error
                temperatures.append(dict(sigma=sigma,entropy=float(np.mean(values)),distance=abs(float(np.mean(values))-target),
                    uv_error=error,wrong_uv_gap=gap,coverage_foreground=float(np.mean(coverages)),
                    mapping_valid=bool(error<=.12 and gap>=.025 and np.mean(coverages)>=.70)))
            candidates=[x for x in temperatures if x['mapping_valid']]
            selected=min(candidates,key=lambda x:(x['distance'],x['sigma'])) if candidates else None
            result.update(mapping_sigma=selected['sigma'] if selected else None,
                oracle_valid=selected is not None,
                intervention_temperature_calibration='train8 original entropy; pilot UV accuracy constraint; no outcome optimization',
                entropy_calibration=dict(target_entropy=target,grid=temperatures,selected=selected,
                    max_uv_error=.12,min_wrong_uv_gap=.025,min_foreground_coverage=.70,
                    entropy_unmatched=selected['distance']>.25 if selected else None))
            if selected is None:result['status']='TOKEN_ORACLE_NOT_VALID_MAPPING_QUALITY'
        write(path,result)
        decision(P3='token_provenance_pass' if result['oracle_valid'] else 'NOT_IDENTIFIABLE_WITH_CURRENT_WRITER_TOKENS',
            H_C='NOT_RUN' if result['oracle_valid'] else 'NOT_IDENTIFIABLE')
        return result
    finally:release(r)
