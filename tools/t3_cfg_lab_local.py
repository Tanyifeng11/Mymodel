"""离线校验归档、复算统计并生成256平方展示；不改服务器原始产物。"""
import argparse
import html
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image, ImageDraw
from tools.t3_cfg_lab import ARMS, CONDITIONS, read, write, sha, summarize


def verify_extract(root):
    archive=root/'review_results.tar.gz';info=read(root/'audit/server_bundle_info.json')
    assert archive.stat().st_size==info['bytes'] and sha(archive)==info['sha256']
    dest=root/'server_results'
    if not dest.exists():
        with tarfile.open(archive) as tar:
            members=tar.getmembers();assert len(members)==info['members']
            for member in members:
                assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            tar.extractall(dest,filter='data')
    manifest=read(dest/'bundle_manifest.json')
    for name,value in manifest.items():
        path=dest/name;assert path.stat().st_size==value['bytes'] and sha(path)==value['sha256'],name
    assert len(manifest)+1==info['members']
    return dest,dict(passed=True,archive_sha256=sha(archive),members_verified=len(manifest))


def decide(analysis):
    stats=read(analysis/'metrics/paired_statistics.json');records=read(analysis/'metrics/per_image.json')
    result=[]
    for condition in CONDITIONS:
        for a,b in read(analysis/'protocol.json')['statistics']['comparisons']:
            rows={r['metric']:r for r in stats if r['condition']==condition and r['comparison']==a+' minus '+b}
            pair={arm:{r['id']:r for r in records if r['condition']==condition and r['arm']==arm} for arm in [a,b]}
            f1=[pair[a][i]['edge_f1']-pair[b][i]['edge_f1'] for i in pair[a]
                if pair[a][i]['edge_f1'] is not None and pair[b][i]['edge_f1'] is not None]
            color_ok=all(rows[k]['ci95'] is not None and rows[k]['ci95'][1]<0 for k in ['ref_rgbuv_h','ref_ab_w1'])
            guard_ok=all(rows[k]['ci95'] is not None for k in ['edge_f1','iou','leak']) and rows['edge_f1']['ci95'][0]>=-.02 and rows['iou']['ci95'][0]>=-.01 and rows['leak']['ci95'][1]<=.005 and min(f1)>=-.15
            result.append(dict(condition=condition,comparison=a+' minus '+b,color_rule_passed=color_ok,
                structure_guard_passed=guard_ok,worst_identity_f1_delta=min(f1),exploratory_combined_pass=color_ok and guard_ok))
    write(analysis/'decision.json',result)
    return result


def gallery(root,source):
    rows=read(source/'evaluation.json');old={r['id']:r for r in rows if r['condition']=='neutral_original'}
    def input_path(row,key):return source/'inputs'/row[key+'_sha256']
    assets=root/'display256';assets.mkdir(exist_ok=True)
    blocks=[];links=[]
    labels=['面料参考','草图','GT原图（换参考时非目标）']+ARMS
    fixed=[]
    for condition in ['original','neutral_swapped']:
        for category in ['top','outwear','pants','dress']:
            fixed.append(sorted([r for r in rows if r['condition']==condition and r['category']==category],key=lambda r:r['sketch_sha256'])[0])
    for row in rows:
        key=row['condition']+'_'+row['id'];paths=[input_path(row,'reference'),input_path(row,'sketch'),input_path(row,'gt')]
        paths += [source/'outputs'/arm/row['condition']/(row['id']+'.png') for arm in ARMS]
        cells=[]
        for col,path in enumerate(paths):
            dest=assets/(key+'_'+str(col)+'.png')
            Image.open(path).convert('RGB').resize((256,256),Image.Resampling.LANCZOS).save(dest)
            assert Image.open(dest).size==(256,256)
            relative=dest.relative_to(root).as_posix();links.append(relative)
            cells.append('<td><img loading="lazy" width="256" height="256" src="'+relative+'"></td>')
        donor='；换参考供体：'+row['reference_donor_id'] if row['condition']=='neutral_swapped' else ''
        blocks.append('<tbody><tr><th colspan="7">'+html.escape(row['condition']+' · '+row['category']+' · '+row['id']+donor+' · '+row['caption'])+'</th></tr><tr>'+''.join(cells)+'</tr></tbody>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>T3 CFG与Lab 192组完整对比</title><style>body{font-family:Arial;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:3px}thead{position:sticky;top:0;background:#e9f1f6}img{width:256px;height:256px;object-fit:fill}tbody th{text-align:left;background:#f4f4f4}</style><h1>T3 CFG×Lab颜色诊断</h1><p>完整192条件；四组同输入与latent。展示统一拉伸到256×256，原始结果保持384×512。Lab为后处理；GT不用于生成或校准。换参考GT只表示原服装，不是换参考目标。</p><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in labels)+'</tr></thead>'+''.join(blocks)+'</table></html>'
    (root/'index.html').write_text(page,encoding='utf-8')
    assert len(links)==192*7 and all((root/p).is_file() for p in links)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    for start in [0,4]:
        panel=Image.new('RGB',(7*256,4*294),'white');draw=ImageDraw.Draw(panel)
        for line,row in enumerate(fixed[start:start+4]):
            for col,label in enumerate(['Reference','Sketch','GT(original)']+ARMS):
                draw.text((col*256+3,line*294+2),label,fill='black')
                pic=Image.open(assets/(row['condition']+'_'+row['id']+'_'+str(col)+'.png'))
                panel.paste(pic,(col*256,line*294+20))
            draw.text((3,line*294+278),row['condition']+' '+row['category']+' '+row['id'],fill='black')
        panel.save(figures/('fixed_'+('original' if start==0 else 'swapped')+'.png'))
    return dict(rows=192,display_images=len(links),size=[256,256],links_passed=True,fixed_case_selection='input SHA first per category; no output-based selection')


def plots(root,analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats=read(analysis/'metrics/paired_statistics.json')
    contrasts=read(analysis/'protocol.json')['statistics']['comparisons']
    for name,metrics in [('color',['ref_rgbuv_h','ref_ab_w1']),('guardrails',['edge_f1','iou','leak','lpips'])]:
        fig,axes=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4.5),squeeze=False)
        for ax,metric in zip(axes[0],metrics):
            for y,(a,b) in enumerate(contrasts):
                row=next(r for r in stats if r['condition']=='original' and r['comparison']==a+' minus '+b and r['metric']==metric)
                if row['ci95'] is None:continue
                lo,hi=row['ci95'];mean=row['mean']
                ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4)
            ax.set_yticks(range(4),[a+' - '+b for a,b in contrasts]);ax.axvline(0,color='gray',linestyle='--')
            ax.set_title(metric+' / original128');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(root/'figures'/(name+'_paired_ci.png'),dpi=160);plt.close(fig)


def main(root):
    source,verification=verify_extract(root)
    analysis=root/'analysis';(analysis/'metrics').mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','metrics/per_image.json']:shutil.copy2(source/name,analysis/name)
    summarize(analysis)
    for name in ['summary.json','paired_statistics.json']:
        assert read(analysis/'metrics'/name)==read(source/'metrics'/name),'服务器统计复算不一致: '+name
    rows=read(source/'evaluation.json');records=read(source/'metrics/per_image.json')
    for row in rows:
        metas=[]
        for arm in ARMS:
            path=source/'outputs'/arm/row['condition']/(row['id']+'.png');meta=read(path.with_suffix('.json'))
            assert sha(path)==meta['output_sha256'] and Image.open(path).size==(384,512)
            assert meta['seed']==row['seed'] and meta['caption']==row['caption'] and not meta['GT_used_in_inference']
            assert meta['cfg']==(7 if arm.startswith('CFG7') else 5)
            metas.append(meta)
        assert len({m['initial_latent_sha256'] for m in metas})==1
        soft=np.load(source/'masks'/(row['id']+'.npy'))
        for a,b in [('CFG7_LAB','CFG7'),('CFG5_LAB','CFG5')]:
            va=np.asarray(Image.open(source/'outputs'/a/row['condition']/(row['id']+'.png')))
            vb=np.asarray(Image.open(source/'outputs'/b/row['condition']/(row['id']+'.png')))
            assert np.array_equal(va[soft==0],vb[soft==0])
    verification.update(statistics_recomputed=True,outputs_verified=768,latent_input_consistency=True,outside_exact=True)
    decisions=decide(analysis);verification['gallery']=gallery(root,source);plots(root,analysis)
    write(root/'audit/local_verification.json',verification)
    print(json.dumps(dict(verified=verification,decision=decisions),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
