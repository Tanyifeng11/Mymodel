"""按用户清理指令移除淘汰权重/诊断张量；保留结果证据和冻结基底。"""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tools.m54_protocol import OUT, TRAIN_ARMS, E5, B1, E5_SHA, B1_SHA, read, write, sha

PLAN = OUT/'audit/cleanup_plan_20261017.json'
REPORT = OUT/'audit/cleanup_20261017.json'
TRACE_AUDIT = OUT/'audit/output_integrity_lightweight_20261017.json'
AUTHORIZATION = '先将服务器上不需要（失败的模块、或是诊断文件）的ckpt以及相关的文件先删除了'
TENSORS = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin'}


def safe_path(path, root, protected):
    path = Path(path)
    assert not path.is_symlink(), ('SYMLINK_NOT_ALLOWED', str(path))
    absolute = path.resolve()
    absolute.relative_to(root.resolve())
    assert absolute not in protected, ('PROTECTED_SOURCE', str(path))
    assert absolute.suffix in TENSORS, ('NOT_A_WEIGHT_OR_TRACE', str(path))
    return absolute


def idle():
    active = subprocess.check_output(['squeue', '-u', 'u2515283058', '-h', '-o', '%i %j %T'], text=True).strip()
    assert not active, ('ACTIVE_JOBS_MUST_FINISH', active)


def protections(root):
    protected = {E5.resolve(), B1.resolve()}
    references = []
    for pattern in ['tools/*.py', 'tools/*.sh', 'configs/*.json', 'submit/*.sh',
                    'output_eval/*/manifests/*.json', 'output_eval/*/config*.json']:
        references.extend(root.glob(pattern))
    text = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in references if p.is_file())
    # 显式使用的历史中间版本继续保留；分散拼接的路径也由step白名单保护。
    referenced_steps = set(re.findall(r'checkpoint-(\d+)', text))
    for manifest in OUT.glob('manifests/*hashes*.json'):
        value = read(manifest)
        if isinstance(value, dict):
            protected.update(Path(p).resolve() for p in value if Path(p).is_file())
    return protected, referenced_steps


def plan():
    assert not REPORT.exists(), 'CLEANUP_ALREADY_EXECUTED'
    idle()
    assert read(OUT/'selection/lock_candidates.json')['selected'] == []
    audit = read(TRACE_AUDIT)
    assert audit['passed'] and audit['conditions'] == 1032
    state = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '118357', '--format=State,ExitCode'], text=True).strip()
    assert state.split('|')[:2] == ['COMPLETED', '0:0'], ('TRACE_AUDIT_NOT_COMPLETED', state)
    root = Path.cwd().resolve()
    protected, referenced_steps = protections(root)
    selected = {}

    def add(path, reason, expected_sha=None):
        absolute = safe_path(path, root, protected)
        stat = absolute.stat()
        selected[str(absolute.relative_to(root))] = dict(path=str(absolute.relative_to(root)), reason=reason,
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino,
            allocated_bytes=stat.st_blocks*512, links=stat.st_nlink, sha256=expected_sha)

    for arm in TRAIN_ARMS:
        for p in (OUT/'training'/arm/'checkpoints').glob('*.pt'):
            add(p, 'M54 lightweight candidate retired after fixed Pilot/dev selection', read(p.with_suffix('.json'))['sha256'])
    for entry in audit['entries']:
        if entry['stage'] != 'oracle_diagnostic' and entry['arm'] in ['E5', 'B1']:
            continue  # 仍保留基线完整轨迹，供后备门禁精确复现。
        p = OUT/entry['stage']/'images'/entry['arm']/(entry['id']+'.pt')
        add(p, 'M54 diagnostic trace verified in full50 pre-cleanup audit', entry['trace_sha256'])
    for p in (OUT/'failures').rglob('*.pt'):
        add(p, 'M54 failed engineering diagnostic; logs and failure JSON retained', sha(p))

    # 仅清理历史训练的非最终中间权重；每个实验的final或latest始终保留。
    for run in sorted(Path('output').iterdir()):
        if not run.is_dir() or run.resolve() == E5.parent.parent.resolve():
            continue
        snapshots = [(int(p.name.split('-')[1]), p) for p in run.iterdir()
                     if p.is_dir() and re.fullmatch(r'checkpoint-\d+', p.name)]
        if not snapshots:
            continue
        final = run/'checkpoint-final'
        has_final = final.is_dir() and any(p.suffix in TENSORS for p in final.rglob('*') if p.is_file())
        latest = max(n for n, _ in snapshots)
        for step, folder in snapshots:
            if str(step) in referenced_steps or (not has_final and step == latest):
                continue
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix in TENSORS and p.resolve() not in protected:
                    add(p, 'unreferenced historical intermediate; final/latest retained')
    value = dict(user_authorization=AUTHORIZATION, utc=datetime.now(timezone.utc).isoformat(), root=str(root),
        trace_audit=str(TRACE_AUDIT), trace_audit_sha256=sha(TRACE_AUDIT), trace_audit_job=118357,
        frozen_sources={str(E5): E5_SHA, str(B1): B1_SHA},
        protected_referenced_steps=sorted(referenced_steps), entries=list(selected.values()),
        bytes=sum(x['bytes'] for x in selected.values()), complete=False)
    write(PLAN, value)
    print('CLEANUP_PLAN', len(selected), 'files', round(value['bytes']/2**30, 3), 'GiB', 'SHA', sha(PLAN), flush=True)
    return value


def apply(expected_plan_sha):
    assert sha(PLAN) == expected_plan_sha, 'PLAN_CHANGED'
    if REPORT.exists():
        assert read(REPORT)['complete']; print('CLEANUP_ALREADY_COMPLETE'); return
    idle()
    value = read(PLAN); root = Path.cwd().resolve()
    assert str(root) == value['root'] and value['user_authorization'] == AUTHORIZATION
    assert sha(TRACE_AUDIT) == value['trace_audit_sha256']
    protected, _ = protections(root)
    # 所有路径和文件身份检查通过后再开始；不递归删除目录，不碰PNG/JSON/日志。
    for entry in value['entries']:
        p = safe_path(root/entry['path'], root, protected); stat = p.stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (entry['bytes'], entry['mtime_ns'], entry['inode'])
        if entry['sha256']: assert sha(p) == entry['sha256'], ('CHANGED_ARTIFACT', entry['path'])
    value.update(plan_sha256=expected_plan_sha, deletion_started_utc=datetime.now(timezone.utc).isoformat())
    write(REPORT, value)
    for index, entry in enumerate(value['entries']):
        (root/entry['path']).unlink(); entry['deleted'] = True
        if index % 50 == 0: write(REPORT, value)
    assert not any((root/x['path']).exists() for x in value['entries'])
    for source, expected in value['frozen_sources'].items(): assert sha(source) == expected, ('FROZEN_SOURCE_CHANGED', source)
    value.update(complete=True, deletion_completed_utc=datetime.now(timezone.utc).isoformat(),
        deleted_files=len(value['entries']), deleted_logical_bytes=value['bytes'], frozen_sources_unchanged=True,
        metrics_png_json_logs_preserved=True, raw_traces_recheck='deleted traces only attested by pre-cleanup full50 audit; never reported as present')
    write(REPORT, value)
    print('CLEANUP_COMPLETE', value['deleted_files'], 'files', round(value['bytes']/2**30, 3), 'GiB logical; quota must be measured separately', flush=True)


def retired_traces():
    if not REPORT.exists(): return {}
    value = read(REPORT)
    assert value['complete'] and value['user_authorization'] == AUTHORIZATION
    audit_path = Path(value['trace_audit']); assert sha(audit_path) == value['trace_audit_sha256']
    audit = read(audit_path); assert audit['passed']
    retired = {x['path']: x for x in value['entries'] if x.get('deleted')}
    return {(x['stage'], x['arm'], x['id']): x for x in audit['entries']
            if str(OUT/x['stage']/'images'/x['arm']/(x['id']+'.pt')) in retired}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'apply']); parser.add_argument('--plan-sha')
    args = parser.parse_args()
    if args.action == 'plan': plan()
    else: apply(args.plan_sha)
