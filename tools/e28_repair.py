"""仅在 B 触发 C2 后运行规则衣片修复；未过 pilot Gate 不扩大确认。"""

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from models.confidence_local_correspondence import build_scaffold
from models.e28_panel_parser import infer_regions, VERSION
from models.panel_correspondence import PanelRegion, source_panels, semantic_match, _semantic
from tools.e27_correspondence import file_sha
from tools.e27_experiment import stats, write


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def scaffold(root, out):
    from tools.e28_experiment import _boundary_f1
    decision = load(out/'decision_summary.json')
    assert decision['B_oracle_reproduced'] and decision['component_check_pass']
    assert decision['selected_repair'] == 'panel_parser', '仅允许 B 阶段选择的一个主分支'
    cases = load(out/'cases.json'); groups = load(root/'data/e28_panel_cases.json')
    mask = np.asarray(Image.open(out/cases['sketches'][0]['mask']).convert('L')) > 0
    folder = out/'C_repair/pilot/C2_panel_repair'; folder.mkdir(parents=True,exist_ok=True)
    rows = []
    for case in cases['references']:
        cid=case['id']; cg=groups[str(cid)]
        smask=np.asarray(Image.open(out/'A_audit/inputs'/f'c{cid:02d}_mask.png'))>0
        for variant in ('original','rot90'):
            prefix=f'c{cid:02d}_{variant}'
            reference=Image.open(out/'A_audit/inputs'/f'{prefix}.png').convert('RGB')
            image,arrays,info=build_scaffold(reference,mask,'full_CALPC',region_proposer=infer_regions)
            image.save(folder/f'{prefix}_scaffold.png')
            np.savez_compressed(folder/f'{prefix}_fields.npz',**arrays)
            write(folder/f'{prefix}_scaffold.json',{'case_id':cid,'variant':variant,'arm':'C2_panel_repair',
                   'source':VERSION,'assignments':{r['name']:r['name'] for r in info['regions']},
                   'info':info,'scaffold_sha256':file_sha(folder/f'{prefix}_scaffold.png'),
                   'changed_component':'source/target panel proposal only; E27 warp/confidence/composition unchanged'})
            # 人工信息只在此评测分支读取，完全不传给 proposal / scaffold。
            manual=source_panels(reference,case['panels'],'manual',smask,cg)
            auto=[PanelRegion(r['name'],a,tuple(r['source_box']),r['coverage'],None,_semantic(r['name']))
                  for r,a in zip(info['regions'],arrays['automatic_source_masks'])]
            mapping=semantic_match(auto,manual)
            for m in manual:
                selected=[a.mask for a in auto if mapping[a.panel_id]==m.panel_id]
                a=np.logical_or.reduce(selected) if selected else np.zeros_like(m.mask)
                inter=int((a&m.mask).sum())
                rows.append({'case':cid,'variant':variant,'manual_panel':m.panel_id,
                            'iou':inter/max(int((a|m.mask).sum()),1),
                            'coverage':inter/max(int(m.mask.sum()),1),
                            'boundary_f1_3px':_boundary_f1(a,m.mask,3),'boundary_f1_5px':_boundary_f1(a,m.mask,5)})
            overlay=reference.copy(); d=ImageDraw.Draw(overlay)
            for a in auto: d.rectangle(a.bbox,outline='red',width=2); d.text(a.bbox[:2],a.panel_id,fill='yellow')
            overlay.save(folder/f'{prefix}_panel_overlay.png')
    write(out/'C_repair/panel_proposal.json',{'rows':rows})
    write(out/'C_repair/selected_route.json',{'route':'C2_panel_parser','trigger':decision,
          'parser_version':VERSION,'training_steps':0,'parameter_search':False,
          'one_component_change':True,'source_access':'RGB only'})
    print('[E28 C2] scaffold complete',flush=True)


def report(root,out):
    from tools.e20_utilization import case_stat
    folder=out/'C_repair/pilot/C2_panel_repair'
    rows=[load(p) for p in sorted(folder.glob('c*_s*_*.json'))]
    assert len(rows)==32
    assert load(out/'C_repair/pilot/frozen_check.json')['pass']
    baseline=[load(p) for p in sorted((out/'B_decomposition/B6_full_auto').glob('c*_s*_*.json'))]
    anchor=[load(p) for p in sorted((out/'B_decomposition/B0_global_anchor').glob('c*_s*_*.json'))]
    ids=sorted({r['case'] for r in rows})
    bykey={(r['case'],r['seed'],r['variant']):r for r in baseline}
    for r in rows:
        other=bykey[r['case'],r['seed'],r['variant']]
        for key in ('noise_sha256','condition_sha256','checkpoint_sha256','target_mask_sha256','sketch_sha256'):
            assert r[key]==other[key],key
    current,base,global_anchor=stats(rows),stats(baseline),stats(anchor)
    proposal=load(out/'C_repair/panel_proposal.json')['rows']
    miou=case_stat([(r['case'],r['iou']) for r in proposal])
    coverage=case_stat([(r['case'],r['coverage']) for r in proposal])
    lr=[r for r in proposal if r['manual_panel'] in ('body_left','body_right')]
    lr_iou=case_stat([(r['case'],r['iou']) for r in lr])
    delta={}
    for metric in ('follow','identity','theta_error','contour_f1','leakage'):
        pairs=[]
        for cid in ids:
            a=stats([r for r in rows if r['case']==cid])[metric]
            b=stats([r for r in baseline if r['case']==cid])[metric]
            if a and b:pairs.append((cid,a['mean']-b['mean']))
        delta[metric]=case_stat(pairs) if pairs else None
    structure=(current['contour_f1']['mean']>=global_anchor['contour_f1']['mean']-.02
               and current['leakage']['mean']<=global_anchor['leakage']['mean']+.01
               and delta['contour_f1']['mean']>=-.02 and delta['leakage']['mean']<=.01)
    gates={'panel_miou':miou['mean']>=.80,'body_left_right_miou':lr_iou['mean']>=.70,
           'panel_coverage':coverage['mean']>=.95,'structure_safe':structure,
           'follow_noninferior':delta['follow']['mean']>=0,'identity_noninferior':delta['identity']['mean']>=0,
           'orientation_improved':delta['theta_error'] is not None and delta['theta_error']['mean']<0}
    passed=all(gates.values())
    rotation_n=sum(stats([r for r in rows if r['case']==cid])['rotation_follow'] is not None for cid in ids)
    rotation=bool(current['rotation_follow'] and current['rotation_pair_coverage']>=2/3 and current['rotation_follow']['mean']>.75) if rotation_n>=4 else None
    stage_metrics={}
    for stage in ('scaffold_metrics','vae_metrics','final_metrics'):
        stage_metrics[stage]={k:case_stat([(r['case'],r[stage][k]) for r in rows if r[stage][k] is not None])
                          for k in ('contour_f1','sketch_iou','leakage','foreground_occupancy','outer_boundary_damage')}
    result={'selected_branch':'C2_panel_parser','parser_version':VERSION,'current':current,'B6_baseline':base,
            'panel_miou':miou,'body_left_right_miou':lr_iou,'panel_coverage':coverage,
            'panel_by_name':{n:case_stat([(r['case'],r['iou']) for r in proposal if r['manual_panel']==n])
                              for n in sorted({r['manual_panel'] for r in proposal})},
            'paired_delta_vs_B6':delta,'structure_stages':stage_metrics,'gates':gates,'pilot_pass':passed,
            'conditions_shared':True,'model_frozen':True,'rotation_case_n':rotation_n,'rotation_pass':rotation,
            'note':'一次固定规则 pilot；人工区域只作评测。缺失口袋/层级区域计零，没有去掉困难 panel。'}
    write(out/'C_repair/report.json',result)
    decision=load(out/'decision_summary.json')
    decision.update(repair_pilot_pass=bool(passed),structure_safe=bool(structure),rotation_pass=rotation,
                    rotation_case_n=rotation_n,confirmation_pass=None,
                    next_route='multi_target_confirmation' if passed else 'panel_parser',
                    stopping_reason=None if passed else 'C2 固定规则未通过全部 pilot Gate；D 不运行，后续应单独研究 learned panel parser。')
    write(out/'decision_summary.json',decision)
    if not passed:
        write(out/'D_confirmation/report.json',{'status':'not_run','reason':'C2 pilot gate failed',
              'failed_gates':[k for k,v in gates.items() if not v],'confirmation_pass':None})
    review=out/'review_images'
    for cid in (13,6,7,12,17):
        tiles=[(out/'A_audit/inputs'/f'c{cid:02d}_original.png','reference'),
               (folder/f'c{cid:02d}_original_panel_overlay.png','C2 proposal')]
        for label,directory in [('B6',out/'B_decomposition/B6_full_auto'),('C2',folder)]:
            for suffix,title in [('original_scaffold','S0'),('original_vae','S1'),('s42_original','S2'),('s42_rot90','rot90')]:
                tiles.append((directory/f'c{cid:02d}_{suffix}.png',label+' '+title))
        sheet=Image.new('RGB',(5*256,2*364),'white');d=ImageDraw.Draw(sheet)
        for i,(path,label) in enumerate(tiles):
            im=Image.open(path).convert('RGB');im.thumbnail((256,340))
            x,y=(i%5)*256,(i//5)*364;sheet.paste(im,(x,y));d.text((x+3,y+344),label,fill='black')
        sheet.save(review/f'c{cid:02d}_C2.jpg',quality=92)
    index=load(out/'artifact_index.json')
    index.update(repair='C_repair/report.json',confirmation='D_confirmation/report.json')
    write(out/'artifact_index.json',index)
    # 小结果包供本地目视审阅；完整图像仍留在服务器本次目录。
    with zipfile.ZipFile(out/'e28_review_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for pattern in ('review_images/*.jpg','*.json','B_decomposition/*report.json','C_repair/*.json','D_confirmation/*.json'):
            for path in out.glob(pattern):archive.write(path,path.relative_to(out))
    print('[E28 C2]',json.dumps({'pass':passed,'gates':gates,'next_route':decision['next_route']}),flush=True)
