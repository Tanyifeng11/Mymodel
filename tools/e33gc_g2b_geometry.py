"""同支持上的零训练F0/F1/F2；E26目标是代理真值，不声称RF2绝对正确。"""
import cv2,numpy as np,torch
from tools.e33gc_g2b_protocol import *
from tools.e33rf_common import build
from data.e33rf_real_rotation_dataset import reference_group,rotate
from data.e32_target_pseudogt import image_at
from models.local_pattern_field import patch_geometry
from tools.e33tmif_metrics import pair,axial

def constant_geometry(image,shape):
    pg=patch_geometry(image.crop(CONFIG['crop']));theta=np.deg2rad(2*pg['orientation'])
    ori=np.broadcast_to(np.array([np.cos(theta),np.sin(theta)],np.float32)[:,None,None],(2,*shape)).copy()
    return np.concatenate([ori,np.zeros((1,*shape),np.float32),np.full((1,*shape),pg['confidence'],np.float32)])

def run():
    init();torch.set_num_threads(2);cv2.setNumThreads(1)
    model,_=build(42,checkpoint=RF/'seed42/RF2/checkpoint_final.pt',device='cpu');model.eval().requires_grad_(False)
    rows=read(OUT/'G0a_real_input_annotations/rows.json');donors=read(TM/'manifests/intervention_data.json')
    records=[]
    for row in rows:
        sid=row['id'];path=IF/'reproduction/seed42/fields'/(sid+'.npz')
        with np.load(path) as z:gt=z['gt'].copy();support=z['support'].astype(bool);rf0=z['orientation'][:3].copy();conf=z['confidence'][:3].copy()
        refs=[image_at(DATASET/row['reference'])];refs +=[rotate(refs[0],a) for a in [90,180]]
        simple=np.stack([constant_geometry(r,support.shape) for r in refs])
        case=reference_group(row);structure=case['structure'][None]
        with torch.no_grad():prior=model.prior(structure)['orientation'].numpy()[0]
        prior_geo=np.concatenate([prior,np.zeros((1,*support.shape),np.float32),np.ones((1,*support.shape),np.float32)])
        field=np.concatenate([rf0,np.zeros_like(conf),conf],1)
        # 主三组相同支持：冻结旧target支持，并共同要求F0三臂输入可读及RF2三臂可读。
        common=support.copy()
        for k in range(3):common &= (simple[k,3]>=.25)&(field[k,3]>=.25)
        item=dict(id=sid,fixed_supported_cells=int(support.sum()),common_readable_cells=int(common.sum()),methods={})
        for name,geometry in [('F0_simple_E26',simple),('F1_RF2',field),('F2_sketch_prior',np.stack([prior_geo]*3))]:
            item['methods'][name]=dict(**pair(dict(zip(ARMS,geometry)),common,gt[3]),
                R0_proxy_absolute_error=float(np.average(axial(geometry[0,:2],gt[:2])[common],weights=gt[3][common])) if common.any() else None)
        for setting,donor_id in [('wrong',donors['donors'].get(sid,{}).get('wrong_texture')),
            ('near',donors['text_compatible_near'].get(sid,{}).get('donor'))]:
            if not donor_id:continue
            donor=donors['donor_rows'][donor_id];dref=image_at(DATASET/donor['reference'])
            donor_geo=constant_geometry(dref,support.shape);dc=reference_group(donor)
            with torch.no_grad():pred=model(dc['reference'][:1],structure)
            o=pred['orientation'].numpy()[0];q=pred['confidence_logits'].sigmoid().numpy()[0,0]
            valid=common&(donor_geo[3]>=.25)&(q>=.25)
            vals={}
            for name,correct,other in [('F0_simple_E26',simple[0,:2],donor_geo[:2]),('F1_RF2',field[0,:2],o),('F2_sketch_prior',prior,prior)]:
                vals[name]=float(np.average(axial(other,gt[:2])[valid]-axial(correct,gt[:2])[valid],weights=gt[3][valid])) if valid.any() else None
            item[setting]=dict(donor_id=donor_id,supported_cells=int(valid.sum()),correct_vs_donor_proxy_error_advantage=vals)
        records.append(item)
    summary={}
    for name in ['F0_simple_E26','F1_RF2','F2_sketch_prior']:
        values=[r['methods'][name]['R0_proxy_absolute_error'] for r in records if r['methods'][name]['R0_proxy_absolute_error'] is not None]
        summary[name]=dict(all_fixed_N=64,readable_N=len(values),R0_proxy_error=float(np.mean(values)) if values else None,
            R90_success_n=sum(r['methods'][name]['r90_success'] for r in records),R180_success_n=sum(r['methods'][name]['r180_success'] for r in records))
    write(OUT/'G0a_simple_geometry/comparison.json',dict(records=records,summary=summary,training_updates=0,
        independent_ground_truth=False,analysis_type='proxy_self_consistency_E26_target_and_source',
        F0_projection='identity axes on common384x512; constant source parsed tangent, no learned reference routing',
        F2='actual frozen RF2.backbone.prior(structure); reference-free',RF2_necessity_demonstrated=None,
        near_note='small frozen donor support; descriptive only'))
    verify_frozen();bundle('geometry')

if __name__=='__main__':run()
