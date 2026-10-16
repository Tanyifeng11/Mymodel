"""仅在模型筛选输出前激活已实际目视核验的输入标签；R0原文件不覆盖。"""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from tools.m54_protocol import *
from tools.m54_prepare import order,panels


def activate():
    assert not (OUT/'round1/images').exists() and not (OUT/'training').exists(),'NO_INPUT_RESELECTION_AFTER_MODEL_SCREENING'
    annotation=Path('configs/m54_input_labels_r1.json')
    labels=read(annotation)
    assert labels['reviewer']=='assistant' and labels['actual_reference_crop_visual_review']
    assert labels['independent_human_review']=='NOT_RUN'
    scores=read(OUT/'audit/input_candidates_r1/scores.json');pool={r['id']:r for r in scores}
    verified=[]
    for item in labels['labels']:
        row=pool[item['id']]
        assert item['reference_pattern'] in ['stripe','grid','symbol','non-periodic']
        assert item['geometry_groups'] and item.get('actual_reference_pattern_visible')
        verified.append(dict(row,pattern_group=item['reference_pattern'],geometry_groups=item['geometry_groups'],
            label_source='assistant actual reference crop and garment input visual review',label_note=item.get('note','')))
    assert len({r['id'] for r in verified})==len(verified)
    used=set();splits={};prepared={}
    patterns=['stripe','grid','symbol','non-periodic'];groups=['sleeve','hem','collar/shoulder','pant-leg']
    slots=[(stage,k,groups[k%4],patterns[(k+k//4)%4]) for stage,n in [('pilot',24),('lock',64)] for k in range(n)]
    verified=sorted(verified,key=lambda r:order(r['id']))
    assert len(verified)>=88,'VERIFIED_INPUT_TOTAL_SHORTAGE'
    allowed=np.array([[kind in r['geometry_groups'] and r['pattern_group']==pattern for r in verified] for _,_,kind,pattern in slots])
    # 同时满足两个split配额，避免贪心把稀少可用衣服提前耗尽而误报无解。
    rr,cc=linear_sum_assignment((~allowed)*1000000+np.arange(len(verified))[None,:])
    assert allowed[rr,cc].all(),'VERIFIED_INPUT_QUOTA_SHORTAGE'
    allocation={slots[i][:2]:dict(verified[j],geometry_group=slots[i][2]) for i,j in zip(rr,cc)}
    for stage,n in [('pilot',24),('lock',64)]:
        rows=[allocation[(stage,k)] for k in range(n)];prepared[stage]=rows;used.update(r['id'] for r in rows)
        splits[stage]='splits/'+stage+'_r1.json'
    natural=sorted([r for r in scores if r['id'] not in used],key=lambda r:order(r['id']))[:96]
    assert len(natural)==96
    splits['natural']='splits/natural_r1.json';prepared['natural']=natural
    allrows=prepared['pilot']+prepared['lock']+natural
    assert len({r['id'] for r in allrows})==184
    hashes={r[k]:sha(r[k]) for r in allrows for k in ['gt','sketch','reference']}
    historical=read(HISTORY/'holdout_audit/validation_signatures.json')
    assert all(historical[path]['file_sha256']==value for path,value in hashes.items())
    # 全部配额/哈希已通过后才落盘，失败不留下半个冻结版本。
    for stage,rows in prepared.items():freeze(OUT/splits[stage],rows);panels(rows,stage+'_r1')
    freeze(OUT/'manifests/input_hashes_r1.json',hashes)
    active=dict(revision=1,splits=splits,annotation_sha256=sha(annotation),original_splits_preserved=True,
        labels_from_actual_input_review=True,model_outputs_used=False,scientific_training_updates=0,commit=commit())
    freeze(OUT/'active_inputs.json',active)
    write(OUT/'audit/input_revision_r1.json',dict(**active,pilot=24,lock=64,natural=96,
        independent_human_review='NOT_RUN',input_files=len(hashes)))
    print('M54 INPUT REVISION R1 ACTIVATED',flush=True)


if __name__=='__main__':activate()
