"""冻结新样本并打包第一批15张原图；未通过前置门槛不扩标。"""
import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from tools.e34_stage0 import read, write, sha
from tools.e34_protocol import OUT

PILOT = OUT / 'evidence_pilot_v2'


def archive(out, name):
    files = [p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ('.zip', '.log', '.err') and p.name != 'artifact_manifest.json']
    write(out / 'artifact_manifest.json', {'artifacts': [{'path': str(p.relative_to(out)), 'sha256': sha(p)} for p in files]})
    with ZipFile(out / name, 'w', ZIP_DEFLATED) as package:
        for path in files + [out / 'artifact_manifest.json']:
            package.write(path, str(path.relative_to(out)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    args = parser.parse_args()
    selection = read(Path('assets/e34_pilot_selection.json'))
    assert sha(OUT / 'protocol/split_manifest.json') == selection['split_sha256']
    previous = read(Path('assets/e34_visual_annotations.json'))['records']
    previous_hashes = {r['reference_sha256'] for r in previous}
    split = read(OUT / 'protocol/split_manifest.json')
    heldout = {sha(args.dataset / r['target']) for group in ('causal_test', 'independent_confirmation') for r in split[group]}
    sets = {'train': set(), 'validation': set()}
    for row in selection['records']:
        path = args.dataset / row['reference']
        row['reference_sha256'] = sha(path)
        assert row['reference_sha256'] not in previous_hashes | heldout
        assert row['reference_sha256'] not in sets[row['group']]
        sets[row['group']].add(row['reference_sha256'])
        if row['schema_pilot']:
            destination = PILOT / 'images' / row['group'] / (row['id'] + '.jpg')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    assert not sets['train'] & sets['validation']
    write(PILOT / 'protocol.json', selection)
    write(PILOT / 'recruitment_audit.json', dict(new_reference_count=40, schema_pilot_count=15,
          prior_annotation_byte_overlap=False, causal_independent_byte_overlap=False,
          train_validation_byte_overlap=False, protocol_source_sha256=sha(Path('assets/e34_pilot_selection.json'))))
    archive(PILOT, 'e34_pilot_schema_review.zip')
    print('E34 pilot: 40 recruited, 15 packed; no training', flush=True)


if __name__ == '__main__':
    main()
