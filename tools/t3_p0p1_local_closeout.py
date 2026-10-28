"""本地收尾：只读校验后的服务器包，重排科学图表标签并归档可读导出。"""
import csv
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from tools.t3_p0p1 import read, write, sha, csv_write
from tools.t3_p0p1_review import sources, METRICS, paired_stats, holm


def closeout(project):
    root=project/'output_eval/t3_p0_p1_20261028';server=root/'server_results'
    verified=read(root/'audit/download_integrity.json')
    assert verified['all_member_sha256_and_sizes_match']
    # 服务器包保持原样；根目录提供最新表格，避免用户误读早期pending台账。
    for name in ['model_registry.csv','external_model_registry.csv','external_input_manifest.csv',
                 'external_generation_manifest.csv','external_metrics.csv','external_comparison_table.csv',
                 'automatic_results_status.json']:
        shutil.copy2(server/name,root/name)
    for folder in ['tables','code']:
        shutil.copytree(server/folder,root/folder,dirs_exist_ok=True)
    shutil.copy2(__file__,root/'code/tools'/Path(__file__).name)
    # 外部统一表的LPIPS与旧P0有微小数值差异，配对统计也明确使用同一次重算。
    data={a:{p.stem:read(p) for p in (server/'metrics'/a/'original').glob('*.json')}
          for a in ['B','T1','T2','C','T3']}
    contrasts=[]
    for baseline in ['B','T1','T2','C']:
        for metric,direction in METRICS.items():
            ids=sorted(data['T3']);assert set(ids)==set(data[baseline]) and len(ids)==96
            values=[data['T3'][i][metric]-data[baseline][i][metric] for i in ids]
            contrasts.append(dict(baseline=baseline,metric=metric,favorable_direction=direction,
                protocol='uniform CUDA float32; TF32 disabled; retrospective development',
                **paired_stats(values,baseline+'/'+metric)))
    for family in [[r for r in contrasts if r['baseline']!='C'],[r for r in contrasts if r['baseline']=='C']]:holm(family)
    csv_write(root/'tables/internal_uniform_paired_statistics.csv',contrasts)
    # 从经过哈希验证的真实输入/输出重绘，不合成、不修饰模型图像。
    lookup={row['sha256']:server/row['path'] for row in read(server/'bundle_manifest.json')
            if Path(row['path']).suffix.lower() in ['.png','.jpg','.jpeg']}
    sup,_,_=sources(project)
    for path in (sup/'evaluation').rglob('*.png'):
        lookup.setdefault(sha(path),path)
    with (server/'provenance/final_panel_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    panels=[]
    for row in rows:
        entries=json.loads(row['source_sha256'])
        labels=list(entries)
        # JSON键序原本为哈希审计排序；恢复展示顺序，避免字母序打乱输入/方法。
        if 'E0' in entries:order=['Sketch','Ref','E0','E1','B','T3']
        elif 'Old ref' in entries:order=['Sketch','Old ref','New ref']+[a+' '+c for a in ['T3','BF-Fashion','ControlNet+IP-Adapter'] for c in ['original','swapped']]
        else:order=['Sketch','Ref','T3','BF-Fashion','ControlNet+IP-Adapter']
        assert set(order)==set(labels)
        canvas=Image.new('RGB',(len(order)*176,280),'white');draw=ImageDraw.Draw(canvas)
        draw.text((6,4),'ID '+row['id']+' | qualitative development examples',fill='black')
        for column,label in enumerate(order):
            display=label.replace('ControlNet+IP-Adapter','CN+IPA').replace('original','old ref').replace('swapped','new ref')
            if 'old ref' in display:display=display.replace(' old ref','\nold ref')
            if 'new ref' in display:display=display.replace(' new ref','\nnew ref')
            draw.text((column*176+6,22),display,fill='black')
            path=lookup[entries[label]];assert sha(path)==entries[label]
            image=ImageOps.contain(Image.open(path).convert('RGB'),(168,222))
            canvas.paste(image,(column*176+(176-image.width)//2,54+(222-image.height)//2))
        relative=Path(row['panel'].split('/t3_p0_p1_20261028/',1)[1])
        out=root/'figures/readable_panels'/relative;out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out)
        panels.append(dict(**row,local_panel=str(out),local_panel_sha256=sha(out),
                           presentation_change='labels abbreviated and separated; image pixels only proportionally resized'))
    csv_write(root/'provenance/local_panel_manifest.csv',panels)
    for folder in ['figures/stage_comparison','external_qualitative_grid','external_failure_cases','figures/reference_swap']:
        files=sorted((root/'figures/readable_panels'/folder).glob('*.png'))
        width=max(Image.open(path).width for path in files)
        canvas=Image.new('RGB',(width,len(files)*280),'white')
        for index,path in enumerate(files):canvas.paste(Image.open(path),(0,index*280))
        canvas.save(root/'figures'/('QA_'+Path(folder).name+'.png'))
    write(root/'audit/local_closeout.json',dict(immutable_server_archive=True,readable_panels=len(panels),
        source_server_package_sha256=verified['archive_sha256'],human_reviewers=0,
        note='visual review notes are AI observations; no human scores'))
    print('LOCAL_CLOSEOUT',len(panels),'readable panels; original server package preserved')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    closeout(parser.parse_args().project)
