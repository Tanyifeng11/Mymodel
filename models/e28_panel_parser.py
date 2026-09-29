"""C2 规则 pilot：从白底前景的躯干截面、颜色断点产生互斥衣片。"""

import cv2
import numpy as np

from models.confidence_local_correspondence import foreground, select_crop


VERSION = 'e28_c2_centerline_color_graph_v1'


def _histogram(image, mask):
    pixels = np.asarray(image)[mask].astype(float)/255
    if not len(pixels): return np.zeros(24)
    vector = np.concatenate([np.histogram(pixels[:,c], bins=8, range=(0,1))[0] for c in range(3)]).astype(float)
    return vector/max(np.linalg.norm(vector),1e-9)


def infer_regions(image):
    mask = foreground(image)
    yy, xx = np.where(mask)
    x0,x1,y0,y1 = int(xx.min()),int(xx.max()+1),int(yy.min()),int(yy.max()+1)
    center = (x0+x1)//2
    gy,gx = np.indices(mask.shape)
    u = (gx-x0)/max(x1-x0,1); v = (gy-y0)/max(y1-y0,1)
    # 下半部穿过中心线的连通截面估计躯干宽度，避免将长袖整体当衣身。
    runs = []
    for y in range(y0+int(.40*(y1-y0)),y0+int(.85*(y1-y0))):
        if not mask[y,center]: continue
        xs = np.where(mask[y])[0]
        groups = np.split(xs, np.where(np.diff(xs)>1)[0]+1)
        run = next((q for q in groups if q[0]<=center<=q[-1]),None)
        if run is not None: runs.append((int(run[0]),int(run[-1]+1)))
    left = int(np.median([r[0] for r in runs])) if runs else x0+int(.25*(x1-x0))
    right = int(np.median([r[1] for r in runs])) if runs else x0+int(.75*(x1-x0))
    left = int(np.clip(left,x0+.20*(x1-x0),x0+.38*(x1-x0)))
    right = int(np.clip(right,x0+.62*(x1-x0),x0+.80*(x1-x0)))
    sleeves = [('left_sleeve',mask & (gx<left) & (v>.08)),
               ('right_sleeve',mask & (gx>=right) & (v>.08))]
    sleeves = [(n,a) for n,a in sleeves if a.sum()>256]
    body = mask.copy()
    for _,area in sleeves: body &= ~area
    center_collar = body & (u>.33) & (u<.67) & (v<.18)
    middle = body & (v>.30) & (v<.65)
    collar_similarity = float(_histogram(image,center_collar) @ _histogram(image,middle))
    extras = []
    if center_collar.sum()>256 and collar_similarity<.75:
        extras.append(('collar',center_collar,{})); body &= ~center_collar
    lbody,rbody = body & (gx<center),body & (gx>=center)
    vertical_similarity = float(_histogram(image,lbody) @ _histogram(image,rbody))
    records = []
    if vertical_similarity<.90 and min(lbody.sum(),rbody.sum())>512:
        records = [('body_left',lbody,{'target_u_range':[0.,.5]}),
                   ('body_right',rbody,{'target_u_range':[.5,1.]})]
        split = 'vertical_color_discontinuity'
        cuts = []
    else:
        # 固定阈值的一次规则试验；不按 case ID 或人工框搜索参数。
        candidates = []
        band = max(12,int(.10*(y1-y0)))
        for y in range(y0+int(.22*(y1-y0)),y0+int(.87*(y1-y0)),4):
            before = body & (gy>=y-band) & (gy<y)
            after = body & (gy>=y) & (gy<y+band)
            if min(before.sum(),after.sum())<256:continue
            score = 1-float(_histogram(image,before) @ _histogram(image,after))
            if score>.20: candidates.append((score,y))
        cuts = []
        for score,y in sorted(candidates,reverse=True):
            if all(abs(y-c)>2*band for c in cuts): cuts.append(y)
            if len(cuts)==2:break
        cuts.sort()
        if cuts:
            boundaries = [y0,*cuts,y1]
            names = ['body_upper','body_lower'] if len(cuts)==1 else ['body_upper','body_middle','body_lower']
            for name,lo,hi in zip(names,boundaries[:-1],boundaries[1:]):
                area = body & (gy>=lo) & (gy<hi)
                if area.sum()>256:
                    records.append((name,area,{'target_v_range':[(lo-y0)/(y1-y0),(hi-y0)/(y1-y0)]}))
            split = 'horizontal_color_discontinuity'
        else:
            records = [('body',body,{})]; split='single_body'
    records += [(n,a,{}) for n,a in sleeves]+extras
    # 将小片或未覆盖像素并入最大 body，保证图结构是前景的互斥覆盖。
    records = [(n,a,extra) for n,a,extra in records if a.any()]
    union = np.logical_or.reduce([a for _,a,_ in records])
    largest = max(range(len(records)),key=lambda i:records[i][1].sum() if records[i][0].startswith('body') else -1)
    records[largest][1][mask & ~union] = True
    packed = []
    for name,area,extra in records:
        box,geometry,coverage = select_crop(image,area)
        ys,xs = np.where(area)
        packed.append({'name':name,'area':area,'crop':box,'geometry':geometry,'coverage':coverage,
                       'source_box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],**extra})
    stacked = np.stack([r['area'] for r in packed])
    assert np.all(stacked.sum(0)[mask]==1) and not stacked[:,~mask].any()
    return mask,packed,{'version':VERSION,'body_split':split,'body_x_limits':[left,right],
                       'vertical_similarity':vertical_similarity,'horizontal_cuts':cuts,
                       'collar_similarity':collar_similarity,'graph_partition_pass':True,
                       'source_access':'RGB only; no case ID, manual panels, motif groups or source mask'}
