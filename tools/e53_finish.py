"""只打包结果和机读审计；中文报告始终在本地生成。"""
import argparse
import tarfile
from tools.e53_protocol import *

def mapping_package(stage='pilot'):
    suffix='' if stage=='pilot' else '_lock'
    root=OUT;paths=['synthetic/'+stage+'_r2','figures/'+stage+'_r2/mapping_review','splits/'+stage+'_r2.json','audit/renderer_'+stage+'_r2.json','audit/mapping_'+stage+'_r2.json','audit/texture_calibration'+suffix+'_r2.json','audit/transport_calibration'+suffix+'_r2.json','manifest/synthetic_'+stage+'_r2_sha256.json','data_revision.json']
    archive=OUT/('e53_mapping_review'+suffix+'_r2.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for name in paths:tar.add(root/name,arcname=name)
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print('MAPPING_ARCHIVE',archive.stat().st_size,sha(archive),flush=True)

def finish():
    from tools.e53_dispatch import status
    assert read(OUT/'metrics/lock/summary.json')['engineering_complete']
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    inputs={}
    for stage in ['pilot','lock']:
        source=read(OUT/'manifest'/('synthetic_'+stage+'_r2_sha256.json'));assert all(sha(p)==h for p,h in source.items()),'SYNTHETIC_INPUT_CHANGED';inputs.update(source)
    write(OUT/'audit/final_inputs.json',dict(unchanged=True,n=len(inputs),commit=commit()))
    from tools.e53_storage import verify_case,journal_records
    raw_checks=[];raw_index=journal_records()
    for stage in ['pilot','lock']:
        for path in (OUT/'images'/stage).rglob('*_trace.json'):
            value=read(path)
            if value['arm']!='E5':raw_checks.append(verify_case(stage,value['arm'],value['id'],stage=='pilot' and value['arm']!='BUDGET_BASELINE',raw_index))
    write(OUT/'audit/raw_storage_layout.json',dict(passed=True,conditions=raw_checks,format='lossless .pt.zst; deserialize with tools.e53_storage.load_raw; indexes preserve serialized-byte SHA256',commit=commit()))
    status();write(OUT/'complete.json',dict(computations_complete=True,local_delivery_complete=False,training_updates=0,pilot_identities=24,lock_identities=64,pilot_scientific_conditions=24*len(PILOT_ARMS),lock_scientific_conditions=64*len(LOCK_ARMS),budget_baselines=88,verdicts=read(OUT/'metrics/lock/summary.json')['verdicts'],raw_tensors='retained losslessly on server as .pt.zst with original serialized-byte SHA256; portable package excludes tensor archives',final_accounting='FINAL job still running; fetch sacct after exit for actual resources',commit=commit()))
    portable=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.psv','.png','.log','.err'] and p.name!='review_sha256.json']
    write(OUT/'manifest/review_sha256.json',{str(p.relative_to(OUT)):sha(p) for p in portable});portable.append(OUT/'manifest/review_sha256.json')
    archive=OUT/'e53_final_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in portable:tar.add(p,arcname=str(p.relative_to(OUT)))
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8');print('FINAL_ARCHIVE',archive.stat().st_size,sha(archive),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['mapping','finish']);p.add_argument('--stage',default='pilot',choices=['pilot','lock']);args=p.parse_args();mapping_package(args.stage) if args.action=='mapping' else finish()
