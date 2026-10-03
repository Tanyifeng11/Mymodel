"""固定train donor；nearest使用原E32 Lab histogram，排除相同图像hash。"""
import hashlib
import numpy as np
from scipy.spatial import cKDTree

def train_wrong_references(records,histograms,hashes):
    ids=[r['id'] for r in records];hist=np.asarray(histograms,np.float32);tree=cKDTree(hist)
    _,nearby=tree.query(hist,k=min(64,len(ids)),workers=4)
    valid=lambda i,j:i!=j and hashes[ids[i]]['reference']!=hashes[ids[j]]['reference'] and hashes[ids[i]]['target']!=hashes[ids[j]]['target']
    output={}
    for i,sid in enumerate(ids):
        candidates=[int(j) for j in np.atleast_1d(nearby[i]) if valid(i,int(j))]
        k=min(64,len(ids))
        while not candidates:
            assert k<len(ids),'无有效wrong donor'
            k=min(k*2,len(ids));_,neighbors=tree.query(hist[i],k=k)
            candidates=[int(j) for j in np.atleast_1d(neighbors) if valid(i,int(j))]
        distance=min(float(np.linalg.norm(hist[i]-hist[j])) for j in candidates)
        # 保持E32同距离按ID排序，不能让tree的任意tie顺序改变负样本。
        radius=max(distance,min(float(np.linalg.norm(hist[i].astype(np.float64)-hist[j])) for j in candidates))+1e-6
        tied=[int(j) for j in tree.query_ball_point(hist[i],radius) if valid(i,int(j))]
        near=min(tied,key=lambda j:(float(np.linalg.norm(hist[i]-hist[j])),ids[j]))
        rng=np.random.default_rng(int(hashlib.sha256(('E32 wrong/'+sid).encode()).hexdigest()[:8],16))
        random=int(rng.integers(len(ids)))
        while not valid(i,random) or random==near:random=int(rng.integers(len(ids)))
        output[sid]=dict(color_near=ids[near],random=ids[random],color_distance=float(np.linalg.norm(hist[i]-hist[near])))
    return output
