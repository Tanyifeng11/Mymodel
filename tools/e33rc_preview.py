"""仅归档已完成阶段供提前本地复核；不写最终Gate或完成声明。"""
import argparse
import io
import json
import tarfile
from tools.e33rc_common import *
from tools.e33rc_visualize import create_panels


def preview(name, phases):
    files=set(); required={}
    controlled=read(OUT/'controlled_manifest.json')['dev']
    cf_ids={r['id'] for r in controlled}
    fallback=hash_order(controlled,'E33RC/visual/controlled-probe')[0]['id']
    for phase in phases:
        folder=OUT/phase
        folder.resolve().relative_to(OUT.resolve())
        assert (folder/'phase_complete.json').exists(), '只归档已完成阶段'
        if not (folder/'visual_audit/selection.json').exists():create_panels(folder)
        ids=read(folder/'visual_audit/selection.json')['unique_ids'];required[phase]=ids
        files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.json','.png'])
        files.update(folder/'real/fields'/(sid+'.npz') for sid in ids)
        files.update(folder/'controlled/fields'/(sid+'.npz') for sid in ((set(ids)&cf_ids)|{fallback}))
    paths=sorted(files)
    manifest=dict(preview_only=True,phases=phases,
        files={str(p.relative_to(OUT)):sha(p) for p in paths})
    bundle=OUT/('preview_'+name+'.tar.gz')
    with tarfile.open(bundle,'w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(OUT)))
        for path,value in [('artifact_manifest.json',manifest),('visual_audit/required.json',required)]:
            data=json.dumps(value,ensure_ascii=False,indent=2).encode('utf-8')
            info=tarfile.TarInfo(path);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    result=dict(preview_only=True,bundle_sha256=sha(bundle),artifact_count=len(paths),phases=phases)
    write(bundle.with_suffix('.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True,choices=['initial','seed43','seed44','ablations'])
    parser.add_argument('--phases',nargs='+',required=True)
    args=parser.parse_args();preview(args.name,args.phases)
