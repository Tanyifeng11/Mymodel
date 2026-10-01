"""E32 scientific field plots；target仅用于评测可视化。"""

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data.e32_field_dataset import cache_path
from data.e32_target_pseudogt import image_at,TAU
from tools.e32_common import read,write


def geometry_plots(out,dataset):
    folder=out/'A_geometry/seed42'
    rows=read(folder/'dev/rows.json')
    readable=[r for r in rows if r['arms']['matched']['orientation_error'] is not None]
    worst=sorted(readable,key=lambda r:r['arms']['matched']['orientation_error'],reverse=True)[:16]
    used={r['id'] for r in worst}
    remaining=sorted([r for r in rows if r['id'] not in used],key=lambda r:hashlib.sha256(('E32 review/'+r['id']).encode()).hexdigest())[:16]
    selected=worst+remaining
    write(out/'audits/geometry_visual_selection.json',dict(seed=42,selected_ids=[r['id'] for r in selected],
          method='16 largest matched orientation errors plus16 fixed-hash cases; no manual filtering'))
    mapping={r['id']:r for r in read(out/'split_manifest.json')['dev']}
    destination=out/'audits/geometry_visuals';destination.mkdir(parents=True,exist_ok=True)
    def orientation(value,mask):
        degrees=(np.degrees(np.arctan2(value[1],value[0]))/2)%180
        return np.ma.array(degrees,mask=~mask)
    for row in selected:
        sid=row['id'];record=mapping[sid]
        with np.load(folder/'fields'/sid/'predictions.npz') as d:
            f={k:d[k] for k in d.files}
        with np.load(cache_path(out,sid)) as d:
            foreground=d['supervision_foreground']>=.95
        interior=f['interior']>=.95
        readable=interior&(f['gt_geometry'][3]>=TAU)
        fig,axes=plt.subplots(3,4,figsize=(16,13),constrained_layout=True)
        for ax in axes.flat:ax.axis('off')
        for x,(key,title) in enumerate([('reference','Reference crop'),('target','Paired target RGB'),('sketch','Target sketch')]):
            axes[0,x].imshow(image_at(dataset/record[key]));axes[0,x].set_title(title)
        artists=[]
        artists.append(axes[0,3].imshow(orientation(f['gt_geometry'][:2],readable),cmap='twilight',vmin=0,vmax=180))
        axes[0,3].set_title('GT orientation (E26 readable)')
        for x,arm in enumerate(('matched','color_near','zero','rot90')):
            artists.append(axes[1,x].imshow(orientation(f[arm+'_orientation'],interior),cmap='twilight',vmin=0,vmax=180))
            error=row['arms'][arm]['orientation_error']
            title='rot90 orientation (intervention)' if arm=='rot90' else '%s orientation; error %s'%(arm,'unreadable' if error is None else '%.1f deg'%error)
            axes[1,x].set_title(title)
        fig.colorbar(artists[0],ax=[axes[0,3]]+list(axes[1]),fraction=.025,label='Axial orientation (degrees)')
        period=axes[2,0].imshow(np.ma.array(f['gt_geometry'][2]/np.log(2),mask=~readable),vmin=-6,vmax=-2,cmap='viridis')
        axes[2,0].set_title('GT log2 frequency')
        axes[2,1].imshow(np.ma.array(f['matched_log_frequency'][0]/np.log(2),mask=~interior),vmin=-6,vmax=-2,cmap='viridis')
        axes[2,1].set_title('Matched log2 frequency')
        fig.colorbar(period,ax=list(axes[2,:2]),fraction=.025,label='log2(cycles/pixel)')
        conf=axes[2,2].imshow(np.ma.array(f['matched_confidence'][:2].mean(0),mask=~foreground),vmin=0,vmax=1,cmap='magma')
        axes[2,2].set_title('Matched geometry confidence')
        axes[2,3].imshow(np.ma.array(f['zero_confidence'][:2].mean(0),mask=~foreground),vmin=0,vmax=1,cmap='magma')
        axes[2,3].set_title('Zero geometry confidence')
        fig.colorbar(conf,ax=list(axes[2,2:]),fraction=.025,label='Confidence')
        response='unreadable' if row['rot90_response_error'] is None else '%.1f deg'%row['rot90_response_error']
        ratio='unavailable' if row['structural_edge_response_ratio'] is None else '%.3f'%row['structural_edge_response_ratio']
        fig.suptitle('E32 seed42 case %s | rot90 response error %s | structure response ratio %s'%(sid,response,ratio))
        fig.savefig(destination/(sid+'.png'),dpi=120);plt.close(fig)
