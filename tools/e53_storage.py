"""只压缩E53诊断张量：标准Zstd，解压SHA256一致后才释放未压缩副本。"""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from tools.e53_protocol import OUT,read,write,sha,commit,freeze,KEY_STEPS
from functools import lru_cache

@lru_cache(maxsize=1)
def storage_commit():return commit()

def digest(data):return hashlib.sha256(data).hexdigest()

def encode(data):
    return subprocess.run(['zstd','-q','-3','--long=24','-T1','-c'],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout

def decode(data):
    return subprocess.run(['zstd','-q','-d','-c'],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout

def checked_path(path):
    path=Path(path);assert OUT.resolve() in path.resolve().parents,('OUTSIDE_E53_ROOT',str(path))
    return path

def archive_bytes(data,path):
    path=checked_path(path);dest=checked_path(str(path)+'.zst');temp=checked_path(str(dest)+'.tmp')
    packed=encode(data);source_hash=digest(data);archive_hash=digest(packed)
    with temp.open('wb') as stream:
        assert stream.write(packed)==len(packed);stream.flush();os.fsync(stream.fileno())
    # 核验实际落盘字节，而非只核验内存中压缩器返回值。
    actual=temp.read_bytes();assert digest(actual)==archive_hash and digest(decode(actual))==source_hash,'LOSSLESS_STORAGE_ROUNDTRIP_FAIL'
    os.replace(temp,dest)
    return dict(logical_source=str(path.relative_to(OUT)),archive=str(dest.relative_to(OUT)),
        source_bytes=len(data),archive_bytes=len(packed),source_sha256=source_hash,archive_sha256=archive_hash,
        roundtrip_verified=True,codec='zstd level3 long24; exact original serialized bytes',commit=storage_commit())

def append_record(path,record):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=True)+'\n');stream.flush();os.fsync(stream.fileno())

def save_raw(value,path):
    import torch
    buffer=io.BytesIO();torch.save(value,buffer)
    record=archive_bytes(buffer.getvalue(),path)
    append_record(Path(path).parent/'storage_index.jsonl',record)

def load_raw(path):
    import torch
    path=Path(path)
    if path.suffix=='.zst':data=decode(path.read_bytes())
    elif path.exists():data=path.read_bytes()
    else:data=decode(Path(str(path)+'.zst').read_bytes())
    return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)

def codec_check():
    import torch
    value=dict(half=torch.arange(1000).half(),single=torch.linspace(-1,1,333),metadata=dict(flag=True,missing=None,energy=.123))
    buffer=io.BytesIO();torch.save(value,buffer);original=buffer.getvalue();restored=decode(encode(original))
    assert restored==original
    obj=torch.load(io.BytesIO(restored),weights_only=False)
    assert obj['metadata']==value['metadata'] and torch.equal(obj['half'],value['half']) and torch.equal(obj['single'],value['single'])
    print('E53 ZSTD CODEC BYTE/TENSOR EXACT PASS',flush=True)

def journal_records():
    path=OUT/'resources/storage_compression_R1.jsonl'
    return {x['logical_source']:x for x in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())} if path.exists() else {}

def compress_existing():
    from tools.e53_dispatch import states
    codec_check();jobs,st=states()
    assert not any(j['action']=='generate' and st.get(j['id']) in ['RUNNING','COMPLETING'] for j in jobs.values()),'ACTIVE_RAW_WRITER'
    failed={name:j['id'] for name,j in jobs.items() if name.startswith('PILOT') and j['action']=='generate' and st.get(j['id'])=='FAILED'}
    assert failed and all('file write failed' in (OUT/'resources'/('gpu_%d.err'%identity)).read_text() for identity in failed.values())
    freeze(OUT/'resources/storage_repair_plan_R1.json',dict(failed_jobs=failed))
    paths=sorted([p for folder in ['traces','failures'] for p in (OUT/folder).rglob('*.pt')],key=lambda p:p.stat().st_size)
    inventory=OUT/'resources/storage_original_inventory_R1.json'
    if not inventory.exists():write(inventory,dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths)))
    journal=OUT/'resources/storage_compression_R1.jsonl';records=journal_records()
    def pack(path):return archive_bytes(path.read_bytes(),path)
    def finalize(record):
        # 日志与可恢复压缩副本均落盘后，才解除未压缩副本占用；原始全部字节仍保留。
        append_record(journal,record);records[record['logical_source']]=record
        source=checked_path(OUT/record['logical_source']);source.unlink()
        if len(records)%128==0:print('E53 LOSSLESS STORAGE',len(records),sum(x['source_bytes']-x['archive_bytes'] for x in records.values()),flush=True)
    # 配额最初只剩约14MiB：先串行从最小文件启动，释放空间后再并行。
    for path in paths[:32]:finalize(pack(path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(pack,p) for p in paths[32:]]):finalize(future.result())
    assert not [p for folder in ['traces','failures'] for p in (OUT/folder).rglob('*.pt')],'UNCOMPRESSED_RAW_REMAINS'
    assert len(records)==read(inventory)['files'] and sum(x['source_bytes'] for x in records.values())==read(inventory)['bytes']
    write(OUT/'audit/storage_compression_R1.json',dict(passed=True,files=len(records),source_bytes=sum(x['source_bytes'] for x in records.values()),
        archive_bytes=sum(x['archive_bytes'] for x in records.values()),all_original_bytes_roundtrip_verified=True,failed_jobs=failed,
        compression_job=int(os.environ['SLURM_JOB_ID']),zstd_version=subprocess.check_output(['zstd','--version'],text=True).strip(),
        scope='only E53 traces/failure tensor files; no tensor dtype, values, scientific samples, or thresholds changed',commit=commit()))
    print('E53 LOSSLESS STORAGE R1 PASS',len(records),flush=True)

def verify_case(stage,arm,identity,full,index=None):
    folder=OUT/'traces'/stage/arm/identity
    if index is None:index=journal_records()
    local=folder/'storage_index.jsonl'
    if local.exists():index.update({x['logical_source']:x for x in (json.loads(line) for line in local.read_text().splitlines())})
    for step in (range(50) if full else KEY_STEPS):
        source=folder/('%02d.pt'%step);record=index[str(source.relative_to(OUT))];archive=OUT/record['archive']
        assert record['roundtrip_verified'] and archive.exists() and archive.stat().st_size==record['archive_bytes'],'RAW_ARCHIVE_INCOMPLETE'
    return dict(stage=stage,arm=arm,id=identity,frames=50 if full else len(KEY_STEPS),lossless_index_complete=True)

if __name__=='__main__':codec_check()
