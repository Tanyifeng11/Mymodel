"""映射审阅留痕：数值复核可重跑，助手目视核验与独立人工核验分别标记。"""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tools.e53_protocol import OUT, CONFIG, read, write, sha, decision, commit
from tools.e53_mapping import matrices


def numeric_review(root=OUT, stage='pilot'):
    root=Path(root);rows=read(root/'splits'/(stage+'_r2.json'))
    audit=read(root/'audit'/('mapping_'+stage+'_r2.json'))
    records={(x['id'],x['mode']):x for x in audit['records']};checks=[]
    manifest=read(root/'manifest'/('synthetic_'+stage+'_r2_sha256.json'))
    def artifact(path):
        # 下载的真实服务器包保留原相对路径；一律基于指定root，不能误读本地预览。
        return root/str(path).replace('\\','/').split(OUT.name+'/',1)[1]
    assert all(sha(artifact(p))==h for p,h in manifest.items()),'DATA_MANIFEST_MISMATCH'
    assert len(rows)==CONFIG[stage+'_n'] and len(records)==len(rows)*3
    for original in rows:
        row=dict(original,geometry=str(artifact(original['geometry'])))
        stats={};points_checked=0
        for geometry in ['N','0','1']:
            oracle,allowed,t,r=matrices(row,geometry=geometry)
            valid=oracle.sum(1)>.99
            assert np.array_equal(valid,allowed.any(1)),'INVALID_MAPPING_COVERAGE'
            nearest_errors=[]
            for mode in ['ORACLE','SHIFT','FLIP']:
                m,support,_,_=matrices(row,geometry=geometry,mode=mode)
                assert np.array_equal(support,allowed)
                assert np.all(m[~allowed]==0) and np.all(m>=0)
                row_error=float(np.abs(m[valid].sum(1)-1).max());assert row_error<2e-6
                # 独立按每行所属part的支持区间求距离，验证argmax确为相应坐标的最近合法格。
                # 坐标变换采用协议实际的FP32；循环边界不能改用FP64后跨过另一端。
                query=t[0].copy()
                if mode!='ORACLE':
                    for part in np.unique(t[1][t[2]]):
                        q=(t[1]==part)&t[2];rr=(r[1]==part)&r[2]
                        if not rr.any():continue
                        low,high=float(r[0][rr,0].min()),float(r[0][rr,0].max());width=max(high-low,1e-6)
                        if mode=='SHIFT':query[q,0]=low+np.remainder((query[q,0]-low)/width+.25,1.)*width
                        else:query[q,0]=low+(1-(query[q,0]-low)/width)*width
                distances=np.linalg.norm(query[:,None].astype(np.float64)-r[0][None],axis=-1)
                distances[~allowed]=np.inf;chosen=m.argmax(1)
                error=distances[np.arange(len(m))[valid],chosen[valid]]-distances[valid].min(1)
                assert float(error.max())<1e-6,('NON_NEAREST_MAPPING',row['id'],geometry,mode,float(error.max()),int(np.where(valid)[0][error.argmax()]))
                if mode=='ORACLE':nearest_errors=distances[np.arange(len(m))[valid],chosen[valid]]
                tv=.5*np.abs(m-oracle).sum(1)
                changed=chosen[valid]!=oracle.argmax(1)[valid]
                stats[geometry+'_'+mode]=dict(valid_queries=int(valid.sum()),foreground_queries=int(t[2].sum()),
                    foreground_queries_without_reference_support=int((t[2]&~valid).sum()),row_sum_error=row_error,
                    argmax_changed=int(changed.sum()),distribution_changed=int((tv[valid]>1e-6).sum()),
                    mean_total_variation=float(tv[valid].mean()),max_nearest_distance_error=float(error.max()))
                if geometry=='N':
                    record=records[(row['id'],mode)]
                    assert len(record['tokens'])==8 and abs(record['coverage']-valid.sum()/t[2].sum())<1e-12
                    for q in record['tokens']:
                        point=record['points'][str(q)];j=point[mode]['reference_token']
                        assert valid[q] and j==int(chosen[q]) and r[1][j]==point['part']==t[1][q]
                        assert np.array_equal(np.asarray(point['target_uv']),t[0][q])
                        assert np.array_equal(np.asarray(point[mode]['reference_uv']),r[0][j])
                        # 图上蓝点按adaptive pooling支持中心标注；核验该中心仍落在同部位。
                        yy,xx=divmod(j,16);y0,y1=2*yy,2*yy+2
                        x0,x1=int(np.floor(xx*24/16)),int(np.ceil((xx+1)*24/16))
                        d=np.load(row['geometry']);py=int((y0+y1)/2*512/32);px=int((x0+x1)/2*384/24)
                        assert d['labelsN'][py,px]==point['part'],'MARKER_PART_MISMATCH'
                        points_checked+=1
            stats[geometry+'_oracle_quantization']=dict(mean=float(np.mean(nearest_errors)),maximum=float(np.max(nearest_errors)))
        checks.append(dict(id=row['id'],points_checked=points_checked,matrices=stats))
    transport=read(root/'audit'/('transport_calibration_'+('r2' if stage=='pilot' else 'lock_r2')+'.json'))
    assert transport['passed'],'RENDERER_TRANSPORT_GATE_FAIL'
    images={row['id']:sha(root/'figures'/(stage+'_r2')/'mapping_review'/(row['id']+'.png')) for row in rows}
    result=dict(passed=True,stage=stage,n=len(rows),manifest_files_verified=len(manifest),checks=checks,
        image_sha256=images,mapping_audit_sha256=sha(root/'audit'/('mapping_'+stage+'_r2.json')),
        semantic_oracle_proven=False,independent_human_review='NOT_RUN',commit=commit(),utc=datetime.now(timezone.utc).isoformat(),
        limitations=['coarse Writer grid quantizes UV; some SHIFT/FLIP argmax positions do not change',
                    'same-part support is retained; shifted U wraps within the part support interval',
                    'unsupported foreground queries retain original attention; coverage is reported per identity',
                    'geometry-coordinate audit does not establish semantic purity of mixed Writer features'])
    write(root/'audit'/('mapping_numeric_review_'+stage+'.json'),result)
    return result


def accept_assistant(stage, delegation, note, reviewed_ids):
    assert delegation and note,'必须记录用户委托原话及实际目视核验结果'
    numeric=numeric_review(OUT,stage)
    expected=list(numeric['image_sha256']);assert sorted(reviewed_ids)==sorted(expected),'必须逐图核验当前split全部身份'
    review=dict(approved=True,reviewer='assistant',user_delegation=delegation,note=note,
        image_sha256=numeric['image_sha256'],mapping_audit_sha256=numeric['mapping_audit_sha256'],
        numeric_review_sha256=sha(OUT/'audit'/('mapping_numeric_review_'+stage+'.json')),
        reviewed_ids=reviewed_ids,independent_human_review='NOT_RUN',semantic_oracle_proven=False,
        utc=datetime.now(timezone.utc).isoformat(),scope=stage+' actual mapping review images')
    write(OUT/'audit'/('mapping_review_'+stage+'.json'),review)
    decision(**{'mapping_review_'+stage:'APPROVED_BY_USER_DELEGATED_ASSISTANT','human_mapping_review_'+stage:'NOT_RUN'})
    print('E53 DELEGATED MAPPING REVIEW PASS',stage,len(expected),flush=True)


def review_approved(stage):
    path=OUT/'audit'/('mapping_review_'+stage+'.json')
    if not path.exists():
        old=OUT/'audit'/('human_mapping_review_'+stage+'.json')
        return old.exists() and read(old).get('approved',False) and read(old).get('reviewer')=='user'
    review=read(path)
    if not (review.get('approved') and review.get('reviewer')=='assistant' and review.get('user_delegation')):return False
    numeric=OUT/'audit'/('mapping_numeric_review_'+stage+'.json')
    return (numeric.exists() and read(numeric)['passed'] and sha(numeric)==review['numeric_review_sha256']
        and sha(OUT/'audit'/('mapping_'+stage+'_r2.json'))==review['mapping_audit_sha256']
        and all(sha(OUT/'figures'/(stage+'_r2')/'mapping_review'/(identity+'.png'))==h for identity,h in review['image_sha256'].items()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(OUT));p.add_argument('--stage',choices=['pilot','lock'],default='pilot');a=p.parse_args()
    result=numeric_review(a.root,a.stage);print('E53 NUMERIC MAPPING REVIEW PASS',result['n'],result['manifest_files_verified'])
