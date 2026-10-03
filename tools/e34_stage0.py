"""只运行 E34-0：真实文件审计、冻结协议、待人工标注包与明确停止记录。"""
import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw

from tools.e34_protocol import (OUT, E32, E29, E30, E33R, GROUPS, CAUSAL_IDS,
                                FAMILIES, PROTOCOL, METRICS)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(item):
    dataset, relative = item
    path = dataset / relative
    return relative, dict(exists=path.is_file(), sha256=sha(path) if path.is_file() else None)


def hash_order(rows, tag):
    return sorted(rows, key=lambda r: hashlib.sha256((tag + '/' + str(r['id'])).encode()).hexdigest())


def source_audit(root, dataset, out, workers):
    source_split = root / E32 / 'split_manifest.json'
    split = read(source_split)
    shutil.copy2(source_split, out / 'protocol/split_manifest.json')
    checks = dict(exact_counts=[len(split[g]) for g in GROUPS] == [45126, 256, 8, 10],
                  exact_split_copy=sha(source_split) == sha(out / 'protocol/split_manifest.json'))
    identities = {g: {r['id'] for r in split[g]} for g in GROUPS}
    checks['unique_ids_within_groups'] = all(len(identities[g]) == len(split[g]) for g in GROUPS)
    checks['disjoint_identity_groups'] = all(not identities[g] & identities[h]
                                           for i, g in enumerate(GROUPS) for h in GROUPS[i + 1:])
    paths = sorted({r[k] for g in GROUPS for r in split[g] for k in ('reference', 'target', 'sketch')})
    hashes = {}
    # 保存每个真实输入的SHA，不仅继承旧报告的布尔值。
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (relative, record) in enumerate(pool.map(file_record, [(dataset, p) for p in paths]), 1):
            hashes[relative] = record
            if i % 4096 == 0 or i == len(paths):
                print('[E34 input hashes]', i, '/', len(paths), flush=True)
    checks['all_inputs_exist'] = all(r['exists'] for r in hashes.values())
    held = GROUPS[1:]
    for role in ('reference', 'target'):
        a = {hashes[r[role]]['sha256'] for r in split['train']}
        b = {hashes[r[role]]['sha256'] for g in held for r in split[g]}
        overlap = sorted((a & b) - {None})
        checks['no_train_heldout_' + role + '_byte_overlap'] = not overlap
        write(out / ('audits/' + role + '_overlap.json'), dict(overlap_sha256=overlap))
    known = read(root / 'data/e27_c3_cases.json')['references']
    checks['exact_causal8'] = sorted(r['case_id'] for r in split['causal_test']) == sorted(CAUSAL_IDS)
    legacy = []
    for row in known:
        actual = dict(source=sha(dataset / row['source']), mask=sha(dataset / row['source_mask']))
        legacy.append(dict(case_id=row['id'], reference=row['source'], actual_sha256=actual,
                           matches_E27=actual['source'] == row['source_sha256'] and actual['mask'] == row['mask_sha256']))
    checks['E27_sources_and_masks_unchanged'] = all(r['matches_E27'] for r in legacy)
    write(out / 'audits/input_hashes.json', hashes)
    write(out / 'audits/legacy_annotations.json', dict(total=18, causal=8, independent=10,
          note='18个均属于既有heldout，不加入E34训练；未补齐E34新schema', rows=legacy))
    write(out / 'audits/split_audit.json', dict(checks=checks, **{'pass': all(checks.values())}))
    return split, hashes, checks


def freeze_sources(root, out):
    paths = [E32 / 'split_manifest.json', E32 / 'audits/target_group_split_audit.json',
             E29 / 'protocol.json', E29 / 'completion_check.json',
             E30 / 'A21_embedding/summary.json', E33R / 'R0_integrity/summary.json',
             E33R / 'protocol.json',
             Path('output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth'),
             Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')]
    paths += [E30 / ('A21_embedding/seed%d/adapter.pt' % s) for s in (42, 43, 44)]
    paths += [Path(p) for p in ('models/local_pattern_field.py', 'models/pattern_geometry.py',
        'models/pattern_coordinate_field.py', 'models/panel_correspondence.py',
        'models/e29_crop_selector.py', 'models/apacc_affinity_adapter.py',
        'tools/e22_spatial.py', 'tools/e22_o4_generation.py', 'tools/e22_o4_metrics.py',
        'tools/e28_experiment.py', 'tools/e29_experiment.py', 'tools/e34_protocol.py', 'tools/e34_stage0.py')]
    frozen = {str(p): dict(exists=(root / p).is_file(),
                         sha256=sha(root / p) if (root / p).is_file() else None) for p in paths}
    write(out / 'protocol/frozen_hashes.json', dict(before=frozen, after=None, unchanged=None))
    return frozen


def review_package(root, dataset, out, split, hashes):
    folder = out / 'annotation_review'
    captions = {Path(r['cloth']).stem: r['caption'].lower() for r in read(root / 'data/train_bf_texture.json')}
    cues = ('stripe', 'plaid', 'floral', 'print', 'pattern', 'geometric', 'polka', 'graphic', 'checkered')
    chosen = []
    for source_group, group, count in [('train', 'train', 100), ('dev', 'validation', 20)]:
        ranked = hash_order(split[source_group], 'E34/evidence/' + group)
        recruits = [r for r in ranked if any(c in captions.get(r['id'], '') for c in cues)]
        recruits += [r for r in ranked if r not in recruits]
        for row in recruits:
            digest = hashes[row['target']]['sha256']
            if digest is not None and digest not in {r['reference_sha256'] for r in chosen}:
                chosen.append(dict(id=row['id'], group=group, reference=row['target'],
                                   reference_sha256=digest, inherited_identity_group=source_group))
            if sum(r['group'] == group for r in chosen) == count:
                break
    banned = {v['sha256'] for v in hashes.values()}
    banned_ids = {r['id'] for g in GROUPS for r in split[g]}
    candidates = [dict(id=p.stem, reference=str(p.relative_to(dataset)))
                  for p in (dataset / 'validation/gt').glob('*.jpg') if p.stem not in banned_ids]
    for row in hash_order(candidates, 'E34/independent_reference'):
        digest = sha(dataset / row['reference'])
        if digest not in banned and digest not in {r['reference_sha256'] for r in chosen}:
            chosen.append(dict(row, group='independent_test', reference_sha256=digest,
                               inherited_identity_group=None))
        if sum(r['group'] == 'independent_test' for r in chosen) == 20:
            break
    records = []
    for row in chosen:
        path = folder / 'images' / row['group'] / (row['id'] + '.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / row['reference'], path)
        assert sha(path) == row['reference_sha256']
        with Image.open(path) as image:
            size = list(image.size)
        records.append(dict(row, image_size_wh=size, review_image=str(path.relative_to(folder)),
                            reviewed_by_human=False, reviewer=None, garment_foreground=None,
                            pattern_support=None, evidence=[], pattern_family=None,
                            orientation_readable=None, identity_group=None, status='pending_human_annotation'))
    write(folder / 'annotations.json', dict(schema=PROTOCOL['annotation_schema'], records=records))
    write(folder / 'recruitment_manifest.json', dict(
        selection=PROTOCOL['evidence_selection'], records=chosen,
        test_independent_of_inherited_split_by_id_and_all_input_byte_hash=True,
        unseen_pattern_family_claim=False, human_labels_available=False))
    for group in ('train', 'validation', 'independent_test'):
        rows = [r for r in records if r['group'] == group]
        for offset in range(0, len(rows), 20):
            sheet = Image.new('RGB', (1000, 4 * 260), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, row in enumerate(rows[offset:offset + 20]):
                image = Image.open(folder / row['review_image']).convert('RGB')
                image.thumbnail((190, 230))
                x, y = j % 5 * 200, j // 5 * 260
                sheet.paste(image, (x, y + 24))
                draw.text((x + 4, y + 4), group + '/' + row['id'], fill='black')
            sheet.save(folder / ('contact_%s_%02d.png' % (group, offset // 20)))
    return records


def annotation_audit(path, dataset, out, split, hashes):
    records = read(path).get('records', []) if path and path.is_file() else []
    counts = dict(train=0, validation=0, independent_test=0)
    errors = []
    seen = set()
    group_ids = {g: {r['id'] for r in split[g]} for g in GROUPS}
    held_hashes = {hashes[r[k]]['sha256'] for g in GROUPS[1:] for r in split[g] for k in ('reference', 'target', 'sketch')}
    train_hashes = {hashes[r[k]]['sha256'] for r in split['train'] for k in ('reference', 'target', 'sketch')}
    all_hashes = {h['sha256'] for h in hashes.values()}
    family = []
    for row in records:
        failures = []
        group, identity = row.get('group'), row.get('id')
        if group not in counts:
            failures.append('unknown_group')
        if not row.get('reviewed_by_human') or not row.get('reviewer'):
            failures.append('not_human_reviewed')
        source = dataset / row.get('reference', '')
        digest = sha(source) if source.is_file() else None
        if digest is None or digest != row.get('reference_sha256'):
            failures.append('reference_hash_or_file_invalid')
        if digest in seen:
            failures.append('duplicate_reference')
        seen.add(digest)
        if group == 'train' and (identity not in group_ids['train'] or digest in held_hashes):
            failures.append('train_holdout_leak')
        if group == 'validation' and (identity not in group_ids['dev'] or digest in train_hashes):
            failures.append('validation_train_leak')
        if group == 'independent_test' and (identity in set.union(*group_ids.values()) or digest in all_hashes):
            failures.append('independent_test_leak')
        if row.get('pattern_family') not in FAMILIES or not row.get('identity_group'):
            failures.append('missing_family_or_identity_group')
        if not isinstance(row.get('orientation_readable'), bool):
            failures.append('missing_orientation_readability')
        evidence = row.get('evidence', [])
        if not 1 <= len(evidence) <= 3:
            failures.append('requires_1_to_3_evidence')
        masks = {}
        for key in ('garment_foreground', 'pattern_support'):
            relative = row.get(key)
            mask_path = path.parent / relative if relative else None
            if mask_path is None or not mask_path.is_file():
                failures.append('missing_' + key)
            else:
                mask = np.asarray(Image.open(mask_path).convert('L'))
                masks[key] = mask > 0
                if source.is_file() and mask.shape != (Image.open(source).height, Image.open(source).width):
                    failures.append('wrong_mask_shape_' + key)
                if not set(np.unique(mask)) <= {0, 255}:
                    failures.append('non_binary_mask_' + key)
        if len(masks) == 2 and masks['garment_foreground'].shape == masks['pattern_support'].shape:
            if np.any(masks['pattern_support'] & ~masks['garment_foreground']):
                failures.append('pattern_support_outside_garment')
            if not masks['pattern_support'].any():
                failures.append('empty_pattern_support')
        for e in evidence:
            box = e.get('box', [])
            if len(box) != 4 or not all(isinstance(v, int) for v in box):
                failures.append('invalid_box')
            elif source.is_file():
                with Image.open(source) as im:
                    w, h = im.size
                x, y, x1, y1 = box
                if not (0 <= x < x1 <= w and 0 <= y < y1 <= h):
                    failures.append('box_outside_reference')
                elif 'pattern_support' in masks and masks['pattern_support'].shape == (h, w):
                    if masks['pattern_support'][y:y1, x:x1].mean() < .90:
                        failures.append('evidence_support_purity_below_0.90')
            if e.get('confidence') not in ('high', 'medium', 'invalid'):
                failures.append('missing_confidence')
            if row.get('orientation_readable') and not isinstance(e.get('orientation_deg'), (int, float)):
                failures.append('missing_readable_orientation')
            if not isinstance(e.get('structural_contamination'), dict) or set(e.get('structural_contamination', {})) != set(PROTOCOL['annotation_schema']['structural_contamination']):
                failures.append('missing_contamination_labels')
        if failures:
            errors.append(dict(id=identity, group=group, errors=sorted(set(failures))))
        else:
            counts[group] += 1
        family.append(dict(id=identity, group=group, pattern_family=row.get('pattern_family'),
                           identity_group=row.get('identity_group'), human_schema_valid=not failures))
    minimum = PROTOCOL['annotation_minimum']
    passed = all(counts[g] >= minimum[g] for g in counts) and not errors
    result = dict(input_path=str(path) if path else None, input_sha256=sha(path) if path and path.is_file() else None,
                  total_submitted=len(records), valid_human_counts=counts, required=minimum,
                  deficit={g: max(0, minimum[g] - counts[g]) for g in counts},
                  errors=errors, **{'pass': passed}, legacy_E27_not_counted=True)
    write(out / 'protocol/annotation_manifest.json', result)
    write(out / 'protocol/reference_family_manifest.json', dict(records=family, family_split_verified=False,
          status='pending_human_labels' if not passed else 'human_annotations_validated'))
    return result


def legacy_rot90(root, dataset, out):
    from tools.e27_correspondence import pattern_variant, source_labels
    from models.local_pattern_field import patch_geometry
    from models.pattern_geometry import axial_distance
    rows = []
    for row in read(root / 'data/e27_c3_cases.json')['references']:
        if row['id'] not in CAUSAL_IDS:
            continue
        image = Image.open(dataset / row['source']).convert('RGB')
        mask = np.asarray(Image.open(dataset / row['source_mask']).convert('L')) > 0
        changed = pattern_variant(image, mask, row['panels'], 'rot90')
        labels = source_labels(mask, row['panels'])
        support = np.zeros(mask.shape, bool)
        measurements = []
        for i, panel in enumerate(row['panels']):
            if panel.get('solid', False):
                continue
            support |= labels == i
            before = patch_geometry(image.crop(tuple(panel['crop'])))
            # 只做source patch估计器的旋转单测；不是最终生成Follow。
            patch = np.rot90(np.asarray(image.crop(tuple(panel['crop'])))).copy()
            after = patch_geometry(Image.fromarray(patch))
            readable = min(before['confidence'], after['confidence']) >= .25
            measurements.append(dict(panel=panel['name'], readable=readable,
                error_deg=float(axial_distance(after['orientation'], before['orientation'] + 90)),
                before=before, after=after))
        unchanged = bool(np.array_equal(np.asarray(image)[~support], np.asarray(changed)[~support]))
        Image.fromarray((support * 255).astype('uint8')).save(out / ('audits/c%02d_pattern_support.png' % row['id']))
        changed.save(out / ('audits/c%02d_rot90.png' % row['id']))
        rows.append(dict(case_id=row['id'], outside_support_unchanged=unchanged, patch_evaluator=measurements))
    passed = len(rows) == 8 and all(r['outside_support_unchanged'] for r in rows)
    write(out / 'audits/legacy_rot90.json', dict(rows=rows, outside_support_pass=passed,
          scope='E29 causal8既有manual panel干预只读回放；不包含E34新增标注或生成',
          E34_new_reference_rot90_integrity=None))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    (out / 'protocol').mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    protocol = dict(protocol=PROTOCOL, code_sha=commit, protocol_sha256=hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(), execution_scope='E34-0 only', training_steps=0)
    write(out / 'protocol/protocol.json', protocol)
    write(out / 'protocol/metric_definition.json', METRICS)
    frozen = freeze_sources(root, out)
    split, hashes, checks = source_audit(root, args.dataset, out, args.workers)
    checks['all_frozen_dependencies_available'] = all(r['exists'] for r in frozen.values())
    checks['E30_embedding_gate'] = read(root / E30 / 'A21_embedding/summary.json')['gate_pass']
    checks['legacy_rot90_nonpattern_unchanged'] = legacy_rot90(root, args.dataset, out)
    records = review_package(root, args.dataset, out, split, hashes)
    annotations = annotation_audit(args.annotations, args.dataset, out, split, hashes)
    checks['human_annotations_ready'] = annotations['pass']
    # 新参考的干预与paired nuisance必须在人工pattern_support可用后验证，不能继承self-reference结果冒充通过。
    checks['new_reference_rot90_integrity_ready'] = False
    write(out / 'protocol/intervention_manifest.json', dict(causal_case_ids=list(CAUSAL_IDS),
          seeds=[42, 43], variants=['original', 'rot90'], source_E29_protocol_sha256=sha(root / E29 / 'protocol.json'),
          outside_pattern_support_legacy_unchanged=checks['legacy_rot90_nonpattern_unchanged'],
          new_reference_integrity_status='not_run_requires_human_pattern_support',
          same_target_and_noise_status='inherits_E29_for_generation; generation_not_run',
          scale_hard_gate=False))
    after = {p: dict(exists=(root / p).is_file(), sha256=sha(root / p) if (root / p).is_file() else None) for p in frozen}
    checks['frozen_dependencies_unchanged'] = frozen == after
    write(out / 'protocol/frozen_hashes.json', dict(before=frozen, after=after, unchanged=frozen == after))
    decision = dict(experiment='E34', execution_scope='E34-0', checks=checks,
          stage0_protocol_locked=True, stage0_gate_pass=all(checks.values()), training_steps=0, generated_diffusion_images=0,
          next_route='complete_human_evidence_annotations_then_new_reference_integrity',
          missing_human_annotation_counts=annotations['deficit'],
          historical_metrics_reused_as_E34_results=False)
    write(out / 'decision_summary.json', decision)
    for stage in ('A_evidence', 'B_canonical', 'C_field_ablation', 'D_injection', 'E_end2end', 'F_causal', 'G_generalization', 'final_review'):
        write(out / stage / 'status.json', dict(status='not_run', reason='E34-0 human annotation/new-reference integrity gate unmet'))
    expected = ['protocol/' + name + '.json' for name in ('protocol', 'split_manifest', 'annotation_manifest',
                'reference_family_manifest', 'intervention_manifest', 'metric_definition', 'frozen_hashes')]
    completion = dict(stage0_audit_complete=True, experiment_complete=False, code_sha=commit,
          required_missing=[p for p in expected if not (out / p).is_file()],
          review_reference_counts={g: sum(r['group'] == g for r in records) for g in ('train', 'validation', 'independent_test')},
          stopped_before_training=True, training_steps=0)
    write(out / 'completion_check.json', completion)
    write(out / 'artifact_manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
          if p.is_file() and p.name not in ('artifact_manifest.json', 'e34_stage0_review.zip')})
    with ZipFile(out / 'e34_stage0_review.zip', 'w', ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in ('.zip', '.log', '.err'):
                archive.write(p, str(p.relative_to(out)))
    print('[E34 decision]', json.dumps(decision, ensure_ascii=False), flush=True)
    print('[E34 completion]', json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
