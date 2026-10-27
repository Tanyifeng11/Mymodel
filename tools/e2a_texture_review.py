"""把冻结计算结果整理成表格和按固定极值规则选取的案例页；不改指标。"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from e2a_texture_prepare import ROOT, NEW, ARMS, read, write, csv_write, sha
from e2a_texture_metrics import csv_read

FONT='C:/Windows/Fonts/msyh.ttc'
def font(n):return ImageFont.truetype(FONT,n)

def summary_table():
    stats=read(ROOT/'statistics/paired_bootstrap.json');rows=[]
    for arm in ARMS:
        a=stats['summary']['original_'+arm];b=stats['summary']['swap_'+arm]
        rows.append(dict(arm=arm,original_DISTS_mean=a['dists']['mean'],original_DISTS_n=a['dists']['n'],swap_DISTS_mean=b['dists']['mean'],swap_DISTS_n=b['dists']['n'],ResponseMargin_mean=b['response_margin']['mean'],original_orientation_success=a['orientation_detection']['success_n'],original_orientation_eligible=a['orientation_detection']['eligible_n'],swap_orientation_success=b['orientation_detection']['success_n'],swap_orientation_eligible=b['orientation_detection']['eligible_n'],original_period_success=a['period_detection']['success_n'],original_period_eligible=a['period_detection']['eligible_n'],swap_period_success=b['period_detection']['success_n'],swap_period_eligible=b['period_detection']['eligible_n'],Lab_color_ResponseMargin_reused=b['color_response_margin_reused']['mean']))
    csv_write(ROOT/'statistics/six_model_summary.csv',rows)
    page=Image.new('RGB',(1900,1120),'white');d=ImageDraw.Draw(page)
    d.text((36,26),'六模型参考纹理补充评价',font=font(40),fill='#112033')
    d.text((36,88),'E2A-TEXTURE-EVAL-01  |  复用768张既有输出  |  seed42开发集，AI参考标注已曝光',font=font(24),fill='#475569')
    xs=[36,170,480,790,1100,1370,1640]
    headers=['模型','原参考 DISTS ↓','换参考 DISTS ↓','换参考 Margin ↑','方向检出 原/换','周期可估 原/换','颜色 Margin ↑']
    d.rectangle((24,143,1876,223),fill='#e9eef5')
    for x,h in zip(xs,headers):d.text((x,159),h,font=font(25),fill='#1d3550')
    for i,(r,arm) in enumerate(zip(rows,ARMS)):
        y=238+i*96;a=stats['summary']['original_'+arm];b=stats['summary']['swap_'+arm]
        if i%2==0:d.rectangle((24,y-6,1876,y+85),fill='#f7f9fc')
        values=[arm,f"{r['original_DISTS_mean']:.5f}",f"{r['swap_DISTS_mean']:.5f}",f"{r['ResponseMargin_mean']:+.5f}",f"{r['original_orientation_success']}/11  ·  {r['swap_orientation_success']}/4",'0/10  ·  0/4',f"{r['Lab_color_ResponseMargin_reused']:+.3f}"]
        for x,v in zip(xs,values):d.text((x,y),v,font=font(28),fill='#15283e')
        for x,key,stage in [(xs[1],'dists',a),(xs[2],'dists',b),(xs[3],'response_margin',b)]:
            ci=stage[key]['CI95'];d.text((x,y+41),f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",font=font(19),fill='#64748b')
    foot=[
        'DISTS：双向最近邻patch集合距离，越低越近；Margin = 到旧参考距离 − 到新参考距离，越高越响应新参考。',
        '原参考有效87/96；换参考有效27/32。其余由冻结mask下内部patch不足导致，全部保留在失败清单。',
        '方向/周期数字为可估成功数，不能当作纹样正确率。换参考方向4/4、周期4/4均在参考端已检测失败。',
        '原参考周期10/10也在参考端已失败，故本轮没有有效周期误差，不能据此断言模型周期迁移全部失败。',
        '颜色Margin来自旧Lab统计，仅作颜色代理；DISTS也不是经过真人验证的纹样身份准确率。',
        '主比较：C−T3M Margin −0.00223，Holm p=0.0644；C−T3 −0.00709，Holm p=0.0202。未支持C优势。']
    for j,line in enumerate(foot):d.text((36,839+j*40),line,font=font(23),fill='#334155')
    page.save(ROOT/'six_model_texture_comparison.png')

def cases():
    manifest=csv_read(ROOT/'input_manifest.csv');index={(r['stage'],r['arm'],r['identity']):r for r in manifest}
    swap=read(ROOT/'swapped_reference/per_identity_metrics.json');lookup={(r['arm'],r['identity']):r for r in swap}
    selected=[]
    for arm in ['T3M','T3']:
        values=[(lookup[('C',r['identity'])]['response_margin']-r['response_margin'],r['identity']) for r in swap if r['arm']==arm and r.get('response_margin') is not None]
        for label,record in [('worst',min(values)),('best',max(values))]:selected.append(dict(stage='swap',identity=record[1],selection=f'C-{arm} ResponseMargin {label}',difference=record[0]))
    original=read(ROOT/'original_reference/per_identity_metrics.json')
    r=next(r for r in original if r['arm']=='C' and r['period_applicable'] and r['period_status']=='reference_detection_failure')
    selected.append(dict(stage='original',identity=r['identity'],selection='first reference-period estimator failure in frozen identity order'))
    r=next(r for r in original if r['arm']=='C' and r['insufficient_interior_patches'])
    selected.append(dict(stage='original',identity=r['identity'],selection='first insufficient-patch identity in frozen identity order'))
    folder=ROOT/'failure_cases/review_pages';folder.mkdir(parents=True,exist_ok=True)
    for n,case in enumerate(selected,1):
        identity=case['identity'];stage=case['stage'];r=index[(stage,'C',identity)]
        old=index[('original','C',identity)]
        paths=[('Sketch',r['sketch_path']),('Old/reference',old['reference_path']),('New reference' if stage=='swap' else 'Frozen mask',r['reference_path'] if stage=='swap' else r['mask_path'])]
        for arm in ['B','C','T3M','T1','T2','T3']:paths.append((arm,index[(stage,arm,identity)]['output_path']))
        canvas=Image.new('RGB',(1152,1690),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,4),f"{n:02} {stage} ID {identity} | {case['selection']}",font=font(17),fill='black')
        for j,(title,path) in enumerate(paths):
            x=(j%3)*384;y=34+(j//3)*552;img=Image.open(path).convert('RGB')
            assert img.size in [(384,512),(256,256)]
            draw.text((x+6,y+2),title,font=font(23),fill='black')
            canvas.paste(img,(x+(384-img.width)//2,y+34+(512-img.height)//2))
        target=folder/f'case_{n:02}_{identity}.png';canvas.save(target);case.update(page=str(target),page_sha256=sha(target),native_model_pixels=True,AI_review_pending=True,human=False)
    write(ROOT/'failure_cases/selected_case_manifest.json',selected)
    print(json.dumps(selected),flush=True)

if __name__=='__main__':summary_table();cases()
