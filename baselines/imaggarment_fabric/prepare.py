"""复制公共初始权重并审计固定数据；既有模型权重始终只读。"""
import os
import shutil
import subprocess
import time
from pathlib import Path
from .common import CONFIG, ROOT, PROJECT, OLD, HERE, read, write, sha, output, identity, training_row, generation_row


def protected_weights():
    records = {}
    for folder in [PROJECT / 'weight', PROJECT / 'output', OLD / 'training']:
        if folder.exists():
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix in {'.bin', '.pt', '.pth', '.ckpt', '.safetensors'}:
                    stat = path.stat()
                    records[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return records


def copy_public(source, target):
    source, target = Path(source), output(target)
    if not target.exists():
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.git', '.cache'))
        else:
            shutil.copy2(source, target)
    records = {}
    files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file() and '.cache' not in p.parts)
    for original in files:
        if '__pycache__' in original.parts or '.git' in original.parts:
            continue
        dest = target if source.is_file() else target / original.relative_to(source)
        a, b = sha(original), sha(dest)
        assert a == b, (original, dest)
        assert original.stat().st_ino != dest.stat().st_ino, '公共权重必须独立复制，不能硬链接'
        records[str(dest.relative_to(ROOT))] = dict(source=str(original), sha256=b, bytes=dest.stat().st_size)
    return records


def verify_public_weights():
    destination = ROOT / 'audit/public_pretrain_verified.json'
    if destination.exists():
        assert read(destination)['passed']
        return
    expected = read(HERE / 'public_weights.json')
    actual = {}
    for component in ['unet', 'vae', 'text_encoder']:
        stem = 'model' if component == 'text_encoder' else 'diffusion_pytorch_model'
        names = [f'sd15/{component}/{stem}.safetensors',
                 f'sd15/{component}/' + ('pytorch_model.bin' if component == 'text_encoder' else stem + '.bin')]
        selected = next(name for name in names if (ROOT / 'initialization' / name).is_file())
        actual[selected] = sha(ROOT / 'initialization' / selected)
    for name in ['image_encoder/model.safetensors', 'ip-adapter_sd15.bin']:
        actual[name] = sha(ROOT / 'initialization' / name)
    assert all(value == expected['sha256'][name] for name, value in actual.items()), actual
    write(destination, dict(passed=True, actual=actual, public_sources=expected))


def main():
    import torch
    started = time.time()
    if (ROOT / 'audit/prepare_complete.json').exists():
        verify_public_weights()
        print('PREPARE_ALREADY_COMPLETE', flush=True)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / 'audit/protected_before.json').exists():
        write(ROOT / 'audit/protected_before.json', protected_weights())
    for folder in ['initialization', 'training', 'cache', 'generation', 'outputs', 'metrics', 'logs', 'audit', 'figures', 'datasets']:
        output(ROOT / folder / '.path').parent.mkdir(parents=True, exist_ok=True)
    provenance = read(HERE / 'vendor/provenance.json')
    for record in provenance['files'].values():
        assert sha(HERE / 'vendor' / record['file']) == record['sha256']
    write(ROOT / 'audit/official_source.json', provenance)
    train = [training_row(r, PROJECT.parent / 'datasets/BF/training', 'train')
             for r in read(PROJECT / 'data/train_bf_texture.json')]
    assert len(train) == 45384
    evaluation = read(OLD / 'evaluation.json')
    assert len(evaluation) == 192
    train_ids = {r['sample_id'] for r in train}
    test_ids = {r['id'].split('_')[0] for r in evaluation}
    assert not train_ids & test_ids
    counts = {}
    for row in evaluation:
        key = row['split'] + '/' + row['condition']
        counts[key] = counts.get(key, 0) + 1
        for field in ['sketch', 'reference', 'gt']:
            assert sha(row[field]) == row[field + '_sha256']
    assert counts == {'heldout128/original': 128, 'swap32/original': 32, 'swap32/swapped': 32}
    originals = {r['id']: r for r in evaluation if r['condition'] == 'original'}
    for row in evaluation:
        if row['condition'] == 'swapped':
            before = originals[row['id']]
            assert row['sketch_sha256'] == before['sketch_sha256'] and row['caption'] == before['caption']
    validation = [training_row(r, PROJECT.parent / 'datasets/BF/validation', 'validation')
                  for r in read(PROJECT / 'data/bf_validation_e9.json')]
    validation = [r for r in validation if r['sample_id'] not in train_ids | test_ids]
    assert validation, '需要至少一个与训练/测试分离的验证身份'
    for row in train + validation:
        assert isinstance(row['caption'], str) and row['caption'].strip()
        for field in ['sketch_path', 'fabric_path', 'target_path']:
            assert Path(row[field]).is_file(), row[field]
    write(ROOT / 'datasets/train.json', train)
    write(ROOT / 'datasets/validation.json', validation)
    write(ROOT / 'evaluation.json', evaluation)
    write(ROOT / 'datasets/generation.json', [generation_row(r) for r in evaluation])
    assets = {}
    public = Path(CONFIG['public_ip_assets'])
    assets.update(copy_public(PROJECT / 'models/stable-diffusion-v1-5', ROOT / 'initialization/sd15'))
    assets.update(copy_public(public / 'image_encoder', ROOT / 'initialization/image_encoder'))
    assets.update(copy_public(public / 'ip_adapter/ip-adapter_sd15.bin', ROOT / 'initialization/ip-adapter_sd15.bin'))
    assets.update(copy_public(public / 'assets.json', ROOT / 'initialization/public_ip_source_assets.json'))
    ip = torch.load(ROOT / 'initialization/ip-adapter_sd15.bin', map_location='cpu', weights_only=True)
    assert set(ip) == {'image_proj', 'ip_adapter'}
    assert tuple(ip['image_proj']['proj.weight'].shape) == (3072, 1024)
    assert not any('bf' in k.lower() or 't3' in k.lower() for k in ip)
    write(ROOT / 'audit/initialization_assets.json', assets)
    verify_public_weights()
    source = {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts}
    protocol = dict(CONFIG, official_commit=provenance['commit'], samples=len(train), validation_samples=len(validation),
                    cases=counts, source_sha256=source,
                    source_commit=subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip(),
                    initialization='independently copied public SD1.5 + h94 IP-Adapter; no Mymodel trained checkpoint',
                    fabric='raw RGB fabric -> official CLIPImageProcessor -> pooled image_embeds -> 4 tokens',
                    objective='official epsilon MSE only; separate Color Adapter and GAM training',
                    native_size=[384, 512], total_sample_exposures=45384 * (CONFIG['color_epochs'] + CONFIG['gam_epochs']),
                    equal_budget_to_T3=False, previous_test_exposure=True, inference_GT=False)
    write(ROOT / 'protocol.json', protocol)
    # CPU 数据接口检查包含训练、验证、换参考，图像原文件只读。
    from PIL import Image
    sample = train[:4] + validation[:1]
    inspected = []
    for row in sample:
        sizes = {field: list(Image.open(row[field]).convert('RGB').size)
                 for field in ['sketch_path', 'fabric_path', 'target_path']}
        inspected.append(dict(sample_id=row['sample_id'], split=row['split'], caption=row['caption'], sizes=sizes))
    write(ROOT / 'audit/data_interface.json', dict(passed=True, train_test_overlap=0, validation_test_overlap=0,
                                                  inspected=inspected, swap_condition_equivalence=True, logo=False))
    assert protected_weights() == read(ROOT / 'audit/protected_before.json')
    write(ROOT / 'audit/prepare_complete.json', dict(passed=True, seconds=time.time() - started,
                                                   torch=torch.__version__, weight_isolation=True))
    print('PREPARED', len(train), len(evaluation), 'validation', len(validation), flush=True)


if __name__ == '__main__':
    main()
