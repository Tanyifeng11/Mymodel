"""用户批准的五对参考修订；原清单、身份和训练预算保持冻结。"""
import argparse
import copy
import io
import numpy as np
from PIL import Image, ImageDraw
from tools.e2a_sup_common import ROOT, read, freeze, sha, commit


ORIGINAL_SHA = '348770342ea8f4043b1a1535694f9e911aeefca1201e46725a1af26b8dbe0cee'
PROPOSAL_SHA = 'f83881f89bd9b71eb643b2fc9f4d32982593d91604376391e29fa8ff7a7aa9f0'
CHANGES = {'103342923': '210630782', '157345960': '120663574',
    '118140191': '143842215', '120663574': '157345960', '179871811': '208830381'}
MANIFEST = 'splits/reference_swap_v2.json'
AMENDMENT = 'audit/reference_swap_amendment_v2.json'
RELEASE = 'audit/reference_swap_release.json'
PANELS = 'audit/swap_reference_review_v2'
REFERENCE_FIELDS = {'alternative_id', 'reference_alternative', 'reference_category',
    'references', 'input_rgb_difference'}


def rgb_difference(first, second):
    a = np.asarray(Image.open(first).convert('RGB').resize((32, 32)), float) / 255
    b = np.asarray(Image.open(second).convert('RGB').resize((32, 32)), float) / 255
    return float(np.abs(a - b).mean())


def validate_revision(original, revised, pool):
    assert len(original) == len(revised) == 32
    assert [r['id'] for r in original] == [r['id'] for r in revised]
    changed = set()
    for before, after in zip(original, revised):
        identity = before['id']
        if identity not in CHANGES:
            assert before == after, ('UNAPPROVED_PAIR_CHANGED', identity)
            continue
        changed.add(identity)
        assert before.keys() == after.keys()
        assert all(before[k] == after[k] for k in before if k not in REFERENCE_FIELDS)
        alternative = pool[CHANGES[identity]]
        assert after['alternative_id'] == alternative['id']
        assert after['reference_alternative'] == alternative['reference']
        assert after['reference_category'] == alternative['stratum']['category']
        assert after['references'] == [before['reference'], alternative['reference']]
        assert after['input_rgb_difference'] >= .05
    assert changed == set(CHANGES)


def build_revision(original, pool):
    revised = copy.deepcopy(original)
    for row in revised:
        if row['id'] not in CHANGES:
            continue
        alternative = pool[CHANGES[row['id']]]
        row.update(alternative_id=alternative['id'], reference_alternative=alternative['reference'],
            reference_category=alternative['stratum']['category'],
            references=[row['reference'], alternative['reference']],
            input_rgb_difference=rgb_difference(row['reference'], alternative['reference']))
    validate_revision(original, revised, pool)
    return revised


def original_inputs():
    path = ROOT / 'splits/reference_swap.json'
    assert sha(path) == ORIGINAL_SHA
    return read(path), {r['id']: r for r in read(ROOT / 'splits/P2b.json')}


def before_generation():
    # 修订和图像复核必须先于五臂 P2b/换参考生成，不能根据输出改配对。
    assert not (ROOT / 'evaluation/P2b').exists()
    assert not (ROOT / 'evaluation/reference_swap').exists()


def amend(approval):
    assert approval.strip()
    before_generation()
    original, pool = original_inputs()
    review_path = ROOT / 'audit/reference_swap_input_review_20261022.json'
    review = read(review_path)
    assert review['original_manifest_sha256'] == ORIGINAL_SHA
    assert sha(ROOT / 'audit/reference_swap_amendment_proposal.png') == PROPOSAL_SHA
    assert {v['id']: v['proposed_alternative_id'] for v in review['proposed_changes']} == CHANGES
    for item in review['proposed_changes']:
        assert sha(pool[CHANGES[item['id']]]['reference']) == item['proposed_reference_sha256']
    for name, digest in review['panels'].items():
        assert sha(ROOT / 'audit/swap_reference_review' / name) == digest
    revised = build_revision(original, pool)
    freeze(ROOT / MANIFEST, revised)
    directory = ROOT / PANELS
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, 32, 8):
        canvas = Image.new('RGB', (384, 1560), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(revised[start:start + 8]):
            for k, key in enumerate(['sketch', 'reference', 'reference_alternative']):
                canvas.paste(Image.open(row[key]).convert('RGB').resize((128, 171)), (k * 128, j * 195 + 24))
            # 不把冻结的 caption 正则分层冒充人工语义标签。
            draw.text((2, j * 195 + 3), row['id'] + ' -> ' + row['alternative_id'], fill='black')
        data = io.BytesIO()
        canvas.save(data, format='PNG')
        path = directory / ('page_%02d.png' % start)
        if path.exists():
            assert path.read_bytes() == data.getvalue()
        else:
            path.write_bytes(data.getvalue())
    paths = sorted({row[key] for row in revised for key in ['sketch', 'reference', 'reference_alternative']})
    evidence = dict(approved=True,approval=approval,source_commit=commit(),
        original_manifest_sha256=ORIGINAL_SHA,manifest=MANIFEST,manifest_sha256=sha(ROOT / MANIFEST),
        P2b_sha256=sha(ROOT / 'splits/P2b.json'),input_review_sha256=sha(review_path),
        approved_proposal_sha256=PROPOSAL_SHA,changes=CHANGES,
        unchanged_pairs=27,identities_unchanged=True,original_references_unchanged=True,
        training_budget_changed=False,selection_used_generated_images=False,
        caption_regex_strata_unchanged=True,caption_categories_are_semantic_proxies=True,
        input_files_sha256={p:sha(p) for p in paths},
        panels={p.name:sha(p) for p in sorted(directory.glob('page_*.png'))})
    freeze(ROOT / AMENDMENT, evidence)
    print('REFERENCE_V2_FROZEN', evidence['manifest_sha256'], 'VISUAL_REVIEW_REQUIRED', flush=True)


def release_reviewed():
    before_generation()
    original, pool = original_inputs()
    amendment = read(ROOT / AMENDMENT)
    assert amendment['approved'] and amendment['changes'] == CHANGES
    assert amendment['manifest_sha256'] == sha(ROOT / MANIFEST)
    assert amendment['P2b_sha256'] == sha(ROOT / 'splits/P2b.json')
    revised = read(ROOT / MANIFEST)
    validate_revision(original, revised, pool)
    assert len(amendment['panels']) == 4
    for name, digest in amendment['panels'].items():
        assert sha(ROOT / PANELS / name) == digest
    for source, digest in amendment['input_files_sha256'].items():
        assert sha(source) == digest, ('REVIEWED_INPUT_CHANGED', source)
    freeze(ROOT / RELEASE, dict(passed=True,manifest=MANIFEST,manifest_sha256=sha(ROOT / MANIFEST),
        original_manifest_sha256=ORIGINAL_SHA,amendment_sha256=sha(ROOT / AMENDMENT),
        input_files_sha256=amendment['input_files_sha256'],panels=amendment['panels'],
        reviewer='AI visual input-legality review of all four v2 pages; human approved five corrections',
        human_blind_scores=None,identities=32,references_per_identity=2,
        caption_regex_strata_unchanged=True,semantic_proxy_limitation_disclosed=True,
        selection_used_generated_images=False,counterfactual_GT_available=False))
    print('REFERENCE_V2_RELEASED', sha(ROOT / MANIFEST), flush=True)


def approved_reference_manifest():
    release = read(ROOT / RELEASE)
    assert release['passed'] and release['manifest'] == MANIFEST
    path = ROOT / MANIFEST
    assert sha(path) == release['manifest_sha256']
    assert sha(ROOT / 'splits/reference_swap.json') == release['original_manifest_sha256'] == ORIGINAL_SHA
    assert sha(ROOT / AMENDMENT) == release['amendment_sha256']
    amendment = read(ROOT / AMENDMENT)
    assert sha(ROOT / 'splits/P2b.json') == amendment['P2b_sha256']
    for source, digest in release['input_files_sha256'].items():
        assert sha(source) == digest, ('REFERENCE_INPUT_CHANGED', source)
    rows = read(path)
    original, pool = original_inputs()
    validate_revision(original, rows, pool)
    return path, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['amend', 'release'])
    parser.add_argument('--approval', default='')
    options = parser.parse_args()
    if options.action == 'amend':
        amend(options.approval)
    else:
        release_reviewed()
