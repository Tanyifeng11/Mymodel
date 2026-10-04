"""物化助手视觉弱标注。正区域外是未知，RGB前景不冒充人工像素GT。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.confidence_local_correspondence import foreground
from models.local_pattern_field import patch_geometry
from models.pattern_geometry import axial_distance
from data.e33_interventions import nuisance
from tools.e34_stage0 import read, write, sha
from tools.e34_protocol import OUT


def image_path(row, dataset, review_root=None):
    return (review_root / row['review_image']) if review_root else dataset / row['reference']


def materialize(source, dataset, out, review_root=None):
    document = read(source)
    records, errors, counts, valid_counts = document['records'], [], {}, {}
    destination = out / 'visual_annotations'
    frozen = out / 'annotation_review/annotations.json'
    recruitment = read(frozen)['records'] if frozen.is_file() else records
    if frozen.is_file() and document['source_manifest_sha256'] != sha(frozen):
        errors.append('recruitment manifest SHA mismatch')
    locked = {(r['group'], r['id']): r['reference_sha256'] for r in recruitment}
    submitted = {(r['group'], r['id']): r['reference_sha256'] for r in records}
    if locked != submitted:
        errors.append('recruitment changed')
    observed_hashes = {g: set() for g in ('train', 'validation', 'independent_test')}
    updated = []
    for row in records:
        row = dict(row)
        key = row['group'] + '/' + row['id']
        source_image = image_path(row, dataset, review_root)
        if sha(source_image) != row['reference_sha256']:
            errors.append(key + ': RGB SHA mismatch')
        observed_hashes[row['group']].add(row['reference_sha256'])
        if row['reviewed_by_human'] or not row['visual_review_completed']:
            errors.append(key + ': wrong annotation provenance')
        image = Image.open(source_image).convert('RGB')
        fg = foreground(image)
        support = np.zeros(fg.shape, bool)
        for x, y, x1, y1 in row['pattern_regions']:
            support[y:y1, x:x1] = True
        support &= fg
        # 未标区域的污染只可给上界，不可当成确定负例。
        row['garment_foreground'] = 'masks/' + key + '_foreground.png'
        row['pattern_support'] = 'masks/' + key + '_positive.png'
        for name, mask in (('garment_foreground', fg), ('pattern_support', support)):
            path = destination / row[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8)).save(path)
            row[name + '_sha256'] = sha(path)
        for evidence in row['evidence']:
            x, y, x1, y1 = evidence['box']
            purity = float(support[y:y1, x:x1].mean())
            evidence['positive_region_purity'] = purity
            if purity < .90:
                errors.append(key + ': evidence purity < .90')
        counts[row['group']] = counts.get(row['group'], 0) + 1
        valid_counts[row['group']] = valid_counts.get(row['group'], 0) + int(row['valid_evidence'])
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for region in row['pattern_regions']:
            draw.rectangle(region, outline='lime', width=2)
        for evidence in row['evidence']:
            draw.rectangle(evidence['box'], outline='cyan', width=2)
        path = destination / 'overlays' / (key + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(path)
        updated.append(row)
    for g, values in observed_hashes.items():
        for other, theirs in observed_hashes.items():
            if g < other and values & theirs:
                errors.append('cross-split RGB overlap: ' + g + '/' + other)
    minimum = dict(train=100, validation=20, independent_test=20)
    for group, count in minimum.items():
        if counts.get(group, 0) < count:
            errors.append('annotation deficit ' + group)
    audit = dict(annotation_source='assistant_visual_review', source_sha256=sha(source),
                 user_authorization=document['authorization'], reviewed_counts=counts,
                 valid_evidence_counts=valid_counts, errors=errors, exploratory_ready=not errors,
                 original_human_annotation_gate_pass=False, limitations=document['limitations'])
    write(destination / 'annotations.json', dict(document, records=updated))
    write(destination / 'audit.json', audit)
    for start in range(0, len(updated), 12):
        sheet = Image.new('RGB', (1024, 870), '#dddddd')
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(updated[start:start + 12]):
            x, y = j % 4 * 256, j // 4 * 290
            image = Image.open(destination / 'overlays' / (row['group'] + '/' + row['id'] + '.png'))
            sheet.paste(image, (x, y + 26))
            draw.text((x + 3, y + 4), '%03d %s' % (start + j, row['id']), fill='black')
        sheet.save(destination / ('review_%03d.jpg' % start), quality=96)
    if errors:
        raise ValueError(errors)
    return updated, audit


def intervention_audit(rows, dataset, out, review_root=None):
    """只旋转第一个正方形evidence，参考的其他像素保持原样。不是全纹样区域旋转。"""
    measurements = []
    destination = out / 'visual_annotations/localized_rot90'
    destination.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        if not row['valid_evidence']:
            continue
        original = np.asarray(Image.open(image_path(row, dataset, review_root)).convert('RGB'))
        x, y, x1, y1 = row['evidence'][0]['box']
        patch = original[y:y1, x:x1].copy()
        rotated = np.rot90(patch).copy()
        changed = original.copy()
        changed[y:y1, x:x1] = rotated
        support = np.zeros(original.shape[:2], bool)
        support[y:y1, x:x1] = True
        measurement = dict(id=row['id'], group=row['group'], box=[x, y, x1, y1],
                           outside_unchanged=bool(np.array_equal(original[~support], changed[~support])),
                           inverse_exact=bool(np.array_equal(np.rot90(rotated, -1), patch)),
                           orientation_readable=row['orientation_readable'])
        if row['orientation_readable']:
            before = patch_geometry(Image.fromarray(patch))
            after = patch_geometry(Image.fromarray(rotated))
            noisy_base, _ = nuisance(patch, np.ones(patch.shape[:2], bool), 34042 + index)
            noisy_rot, _ = nuisance(rotated, np.ones(patch.shape[:2], bool), 34042 + index)
            # 去除共同边缘，JPEG/插值产生的边界不参加局部测量。
            b = patch_geometry(Image.fromarray(noisy_base[3:-3, 3:-3]))
            r = patch_geometry(Image.fromarray(noisy_rot[3:-3, 3:-3]))
            measurement.update(clean_error_deg=float(axial_distance(after['orientation'], before['orientation'] + 90)),
                               noisy_error_deg=float(axial_distance(r['orientation'], b['orientation'] + 90)),
                               clean_confidence=min(before['confidence'], after['confidence']),
                               noisy_confidence=min(b['confidence'], r['confidence']),
                               visual_orientation_error_deg=float(axial_distance(before['orientation'], row['evidence'][0]['orientation_deg'])))
        Image.fromarray(changed).save(destination / (row['group'] + '_' + row['id'] + '.png'))
        measurements.append(measurement)
    readable = [r for r in measurements if r['orientation_readable']]
    summary = dict(scope='localized evidence-square rotation only; full-reference support intervention not established',
                   rows=measurements, rotated_references=len(measurements), readable_references=len(readable),
                   outside_unchanged=all(r['outside_unchanged'] for r in measurements),
                   inverse_exact=all(r['inverse_exact'] for r in measurements),
                   clean_follow=float(np.mean([r['clean_error_deg'] <= 15 and r['clean_confidence'] >= .25 for r in readable])),
                   noisy_follow=float(np.mean([r['noisy_error_deg'] <= 15 and r['noisy_confidence'] >= .25 for r in readable])),
                   full_reference_intervention_gate_pass=False)
    write(destination / 'audit.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=Path('assets/e34_visual_annotations.json'))
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--review-root', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    rows, audit = materialize(args.annotations, args.dataset, args.out, args.review_root)
    intervention = intervention_audit(rows, args.dataset, args.out, args.review_root)
    print(audit)
    print({k: v for k, v in intervention.items() if k != 'rows'})


if __name__ == '__main__':
    main()
