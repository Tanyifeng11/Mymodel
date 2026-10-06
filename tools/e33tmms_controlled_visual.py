"""展示原 controlled dev128 全集；不选择或改变科学判定样本。"""
import numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from tools.e33rf_common import OUT as RF
from tools.e33tmoc_geometry_audit import overlay
from tools.e33tmms_protocol import *

def run():
    dest=OUT/'M2/controlled/dev128';assert (OUT/'M2/controlled/gate.json').exists()
    manifest=RF/'controlled_manifest.json'
    audit=read(OUT/'protocol/M2_training_identity_audit.json')
    assert audit['files'][str(manifest)]==sha(manifest)
    rows=read(manifest)['dev'];assert len(rows)==128
    assert {r['id'] for r in rows}=={p.parent.name for p in dest.glob('*/case.json')}
    visual=OUT/'visual_audit/M2_controlled_dev128';visual.mkdir(parents=True,exist_ok=True)
    panels=[]
    for row in rows:
        sid=row['id'];target=image_at(DATASET/row['target']);sketch=image_at(DATASET/row['sketch'])
        source=np.asarray(target.crop(tuple(row['box'])))
        with np.load(OUT/'frozen_fields/controlled_dev'/(sid+'.npz')) as z:field=z['orientation'].copy()
        panel=Image.new('RGB',(640,592),'white');draw=ImageDraw.Draw(panel);draw.text((3,2),sid,fill='black')
        for i,arm in enumerate(['R0','R90','R180']):
            rgb=Image.open(dest/sid/(arm+'.png')).convert('RGB')
            with np.load(dest/sid/(arm+'_orientation.npz')) as z:g=z['geometry'].copy();support=z['support_mask'].copy()
            ref=Image.fromarray(np.rot90(source,i).copy()).resize((384,512),Image.Resampling.BICUBIC)
            items=[('Reference '+arm,ref),('Frozen RF2',overlay(sketch,field[i])),('M2 carrier',rgb),
                ('E26 readout',overlay(rgb,g[:2])),('Readout support',Image.fromarray(np.uint8(support)*255))]
            for col,(label,img) in enumerate(items):
                y=22+i*190;draw.text((col*128+2,y),label,fill='black');img=img.convert('RGB');img.thumbnail((124,166));panel.paste(img,(col*128+2,y+18))
        panel.save(visual/(sid+'.png'));panels.append(panel)
    for i in range(0,128,4):
        page=Image.new('RGB',(640,2368),'white')
        for j,p in enumerate(panels[i:i+4]):page.paste(p,(0,j*592))
        page.save(visual/('page%02d.png'%(i//4)))
    write(visual/'audit.json',dict(ids=[r['id'] for r in rows],count=128,
        selection='all original controlled dev128; original manifest order; no output-based selection',
        manifest_sha256=sha(manifest),git_commit=commit(),final_image='not run: controlled carrier Gate only'))
    from tools.e33tmms_benchmark import bundle
    bundle('M2_stage_A')

if __name__=='__main__':run()
