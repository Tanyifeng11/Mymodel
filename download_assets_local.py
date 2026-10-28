"""本地下载已冻结的官方资产，核对Hub哈希；上传包不包含任何源代码。"""
import concurrent.futures
import hashlib
import json
import threading
import time
import tarfile
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
TRANSFER=ROOT/'transfer'
LOCK=threading.Lock()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def download(group,asset,row):
    name=row['rfilename'];target=ROOT/'assets'/group/name;target.parent.mkdir(parents=True,exist_ok=True)
    expected=row.get('lfs',{}).get('sha256')
    def verify(path):
        if path.stat().st_size!=row['size']:return False
        if expected:return sha(path)==expected
        value=hashlib.sha1(('blob '+str(row['size'])+'\0').encode()+path.read_bytes()).hexdigest()
        return value==row['blobId']
    if target.is_file() and verify(target):return dict(path=str(target.relative_to(ROOT)),bytes=row['size'],sha256=sha(target))
    partial=target.with_name(target.name+'.part')
    url=f"https://huggingface.co/{asset['repository']}/resolve/{asset['revision']}/{name}"
    for attempt in range(1,9):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            response=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,60))
            response.raise_for_status()
            if offset and response.status_code!=206:offset=0
            with response,partial.open('ab' if offset else 'wb') as stream:
                last=time.monotonic()
                for chunk in response.iter_content(4*1024*1024):
                    stream.write(chunk)
                    if time.monotonic()-last>25:
                        with LOCK:print('DOWNLOAD',group,name,stream.tell(),'/',row['size'],flush=True)
                        last=time.monotonic()
            assert verify(partial),('OFFICIAL_HASH_OR_SIZE_MISMATCH',name)
            partial.replace(target)
            with LOCK:print('VERIFIED',group,name,row['size'],flush=True)
            return dict(path=str(target.relative_to(ROOT)),bytes=row['size'],sha256=sha(target))
        except Exception as error:
            with LOCK:print('DOWNLOAD_RETRY',group,name,attempt,type(error).__name__,flush=True)
            if attempt==8:raise
            time.sleep(10)


def main():
    plan=json.loads((TRANSFER/'download_plan.json').read_text())
    jobs=[(group,asset,row) for group,asset in plan.items() for row in asset['files']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(lambda args:download(*args),jobs))
    assets={group:dict(repository=asset['repository'],revision=asset['revision'],path='assets/'+group,
        files={row['rfilename']:dict(size=row['size'],sha256=next(r['sha256'] for r in records if r['path']==f"assets/{group}/{row['rfilename']}")) for row in asset['files']})
        for group,asset in plan.items()}
    manifest=dict(source='official HuggingFace repositories, pinned revisions; LFS SHA256 or Git blob hash verified',assets=assets,files=records,source_code_included=False)
    archive=TRANSFER/'lots_assets.tar'
    with tarfile.open(archive,'w') as bundle:
        for row in records:bundle.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    parts=[]
    with archive.open('rb') as stream:
        index=0
        while True:
            first=stream.read(8*1024*1024)
            if not first:break
            index+=1;path=TRANSFER/f'lots_assets.part{index:03}'
            with path.open('wb') as out:
                out.write(first);remaining=1024**3-len(first)
                while remaining:
                    chunk=stream.read(min(8*1024*1024,remaining))
                    if not chunk:break
                    out.write(chunk);remaining-=len(chunk)
            parts.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path)))
            print('UPLOAD_PART_READY',path.name,path.stat().st_size,flush=True)
    manifest.update(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,parts=parts)
    (TRANSFER/'assets_bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'assets_local_verified.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('LOCAL_OFFICIAL_ASSETS_COMPLETE',len(records),'files',len(parts),'parts',archive.stat().st_size,flush=True)


if __name__=='__main__':main()
