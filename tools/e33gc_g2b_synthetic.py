"""A不足4例时唯一允许的B工程控制；新caption与B0成对，不混算真实参考结论。"""
import argparse,hashlib
from PIL import Image
from tools.e33gc_g2b_protocol import *
from data.e33gc_g2b_renderer import construct,input_dir
from tools.e33gc_g2b_prepare import panels
from tools.e33gc_g2b_eval import dft_pair
from tools.e33tm_protocol import category

def run(repair=False):
    init();assert read(OUT/'decision_summary.json')['main_reference_derived_G2b']=='not_run'
    suffix='_v2' if repair else '';root=OUT/('G1a_synthetic_reference_control'+suffix)
    if repair:
        reviews=read('tools/e33gc_g2b_synthetic_reviews_v1.json')
        write(OUT/'G1a_synthetic_reference_control/two_AI_reviews.json',dict(reviews=reviews,eligible_n=14,
            input_contract_pass=False,formal_training_updates=0,reason='two jointly rejected sketch-mask artifacts'))
        decision(synthetic_reference_control_v1='invalid_input_contract',routeB_mask_repair_before_training=True)
    original=read(OUT/'G1a_reference_target_audit/candidate16_rows.json');rows=[];materials=[];audit=[]
    for row in original:
        cat=category(row['caption']);cat={'trousers':'pants','knit':'sweater','other':'garment'}.get(cat,cat)
        r=dict(row,route='B',original_caption=row['caption'],caption='a '+cat+' with a striped fabric pattern.')
        if repair:r['renderer_revision']=2
        value=construct(r);folder=input_dir(r);folder.mkdir(parents=True,exist_ok=True)
        value['source_crop'].save(folder/'P_source.png');value['sketch'].save(folder/'sketch.png')
        for arm,ref,target in zip(ARMS,value['references'],value['targets']):
            ref.save(folder/(arm+'_reference.png'));target.save(folder/(arm+'_target.png'))
        metric=dft_pair(dict(zip(ARMS,value['targets'])),r['dft_boxes'],periodic=True)
        audit.append(dict(id=r['id'],DFT=metric,shared_P_source=True,source_sha256=sha(folder/'P_source.png'),
            renderer_sha256=sha('data/e33gc_g2b_renderer.py'),files={p.name:sha(p) for p in folder.glob('*.png')}))
        rows.append(r);materials.append((r['id'],r['caption'],[('P source',value['source_crop']),('sketch',value['sketch'])]+
            [(a+' ref',v) for a,v in zip(ARMS,value['references'])]+[(a+' truth',v) for a,v in zip(ARMS,value['targets'])]))
    write(root/'candidate16_rows.json',rows)
    write(root/'automatic_audit.json',dict(records=audit,N=16,
        independent_DFT_pass_n=sum(a['DFT']['r90_success'] and a['DFT']['r180_success'] for a in audit),human_pass=None))
    write(OUT/'protocol'/('routeB_frozen_config'+suffix+'.json'),dict(route='B',main_route_A='not_run',reason='A fewer than4 jointly AI-qualified inputs',
        source='independent procedural128 RGB P; same P tiled into reference and target',
        periods=[16,32],axes=[0,90],seed_rule='SHA256 E33GC-G2b/synthetic/identity first16hex',
        color_range=[65,190],contrast_range=[35,55],phase_range=[0,6.283185307179586],
        caption='new neutral category+striped fabric, paired all arms and freshB0',
        reference_geometry='fixed sketch-derived silhouette and center128 support across all arms',
        train_configuration=CONFIG,real_reference_claim_allowed=False,renderer_revision=2 if repair else 1,
        mask_repair='uniform sketch-only17px close then external-contour fill; old16 identities, P, captions, DFT boxes unchanged' if repair else None))
    panels(materials,OUT/'visual_audit'/('G1a_routeB_candidate16'+suffix))
    decision(synthetic_reference_control='await_two_AI_review',next_route='routeB_input_contract_review')
    verify_frozen();bundle('routeB_inputs'+suffix)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repair-mask',action='store_true');a=p.parse_args();run(a.repair_mask)
