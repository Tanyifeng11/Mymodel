"""五组输入预选名单的固定全候选图例；绝不按候选结果换身份。"""
import tarfile
import numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from data.e33rf_real_rotation_dataset import rotate
from tools.e33tmoc_geometry_audit import overlay
from tools.e33tmoc_protocol import *

NAMES=['C0_current','C1_analytic','C2_appearance','C3_support_aware','C4_confidence_blend']

def case_panel(row,selected):
    sid=row['id'];reference=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch'])
    with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:fields=z['orientation'][:2].copy()
    canvas=Image.new('RGB',(9*128,400),'white');draw=ImageDraw.Draw(canvas)
    draw.text((4,2),sid+' | '+row['caption'][:170],fill='black')
    for ai,arm in enumerate(('R0','R90')):
        ref=reference if ai==0 else rotate(reference,90)
        tiles=[('Ref '+arm,ref),('RF tangent '+arm,overlay(sketch,fields[ai]))]
        tiles += [(name+' S2',Image.open(OUT/'candidates'/name/'seed42'/sid/'S2'/(arm+'.png'))) for name in NAMES]
        for stage,label in [('S2','Selected S2'),('S9','Selected Final')]:
            path=OUT/'candidates'/selected/'seed42'/sid/stage/(arm+'.png') if selected else None
            if path is not None and path.exists():tiles.append((label,Image.open(path)))
            else:
                missing=Image.new('RGB',(384,512),(235,235,235));ImageDraw.Draw(missing).text((10,210),'not run / not frozen',fill='black')
                tiles.append((label,missing))
        for i,(label,image) in enumerate(tiles):
            y=22+ai*188;draw.text((i*128+2,y),label,fill='black');image=image.convert('RGB');image.thumbnail((124,166))
            canvas.paste(image,(i*128+2,y+18))
    return canvas

def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    initial=read(OUT/'candidates/initial_summary.json');methods={**initial,**read(OUT/'candidates/appearance_summary.json')}
    app=read(OUT/'appearance_metrics/summary.json')
    items=[('S1_R90','S1','r90_success'),('S2_R90','S2','r90_success'),
        ('S2_R180','S2','r180_success'),('S2_readable','S2','r90_readable'),('S2_support','S2','r90_coverage')]
    for title,stage,key in items:
        stats=[methods[n][stage]['statistics'][key] for n in NAMES]
        means=np.array([s['mean'] for s in stats])*100;cis=np.array([s['ci95'] for s in stats])*100
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(range(5),means,color=['gray','silver','#4c78a8','#54a24b','#f58518'])
        ax.errorbar(range(5),means,yerr=np.vstack([means-cis[:,0],cis[:,1]-means]),fmt='none',ecolor='black',capsize=4)
        ax.set_xticks(range(5),['C0','C1 diagnostic','C2','C3','C4']);ax.set_ylabel('Percent (fixed N=64)');ax.set_ylim(0,100);ax.set_title(title)
        fig.tight_layout();folder=OUT/'curves';folder.mkdir(exist_ok=True);fig.savefig(folder/(title+'.png'),dpi=160);plt.close(fig)
    for key in ('Lab_color_distance','color_histogram_similarity','texture_score','patch_lpips'):
        stats=[app[n]['S2']['statistics'][key] for n in NAMES]
        fig,ax=plt.subplots(figsize=(7,4))
        for i,s in enumerate(stats):
            if s['mean'] is None:continue
            ax.errorbar(i,s['mean'],yerr=np.array([[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]]),fmt='o',capsize=4)
        ax.set_xticks(range(5),['C0','C1 diagnostic','C2','C3','C4']);ax.set_ylabel(key);ax.set_title('S2 '+key+' (case bootstrap, NA reported)')
        fig.tight_layout();fig.savefig(OUT/'curves'/('S2_'+key+'.png'),dpi=160);plt.close(fig)

def main():
    cohort=read(TM/'manifests/cohorts.json')['primary'];lookup={r['id']:r for r in cohort}
    sets=read(OUT/'splits/visual_sets.json')['sets'];selected=read(OUT/'decision_summary.json').get('selected_candidate')
    for name,ids in sets.items():
        folder=OUT/'visual_audit/fixed_sets'/name;folder.mkdir(parents=True,exist_ok=True)
        panels=[case_panel(lookup[sid],selected) for sid in ids]
        for sid,p in zip(ids,panels):p.save(folder/(sid+'.png'))
        for i in range(0,len(panels),4):
            page=Image.new('RGB',(panels[0].width,panels[0].height*4),'white')
            for j,p in enumerate(panels[i:i+4]):page.paste(p,(0,j*p.height))
            page.save(folder/('page%d.png'%(i//4)))
        write(folder/'manifest.json',dict(ids=ids,source_selection=sha(OUT/'splits/visual_sets.json'),selected_candidate=selected,
            columns=['reference','RF tangent']+NAMES+['selected S2','selected final'],arms=['R0','R90']))
    curves()
    with tarfile.open(OUT/'carrier_gate_review.tar.gz','w:gz') as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and (p.suffix=='.json' or 'visual_audit' in p.parts or 'curves' in p.parts):archive.add(p,arcname=str(p.relative_to(OUT)))
    print('[OC carrier visual complete]',len(sets),'fixed groups, bundle SHA256',sha(OUT/'carrier_gate_review.tar.gz'),flush=True)

if __name__=='__main__':main()
