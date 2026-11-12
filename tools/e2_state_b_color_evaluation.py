"""用原 400 步 B 的逐图结果做配对颜色与纹样评价。"""
import cv2
import numpy as np
from PIL import Image

from tools.bf_t3_three_arm_report import paired
from tools.e2_fft_experiment import read, write


ARMS = ('original_b', 'paired_color_b')
SWITCHES = {'direction': {'v16': 'h16', 'v32': 'h32'}, 'period': {'v16': 'v32', 'h16': 'h32'}}


def lab_fields(path, mask):
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    mean = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].mean(0)
    local = cv2.cvtColor(cv2.GaussianBlur(rgb, (0, 0), 12), cv2.COLOR_RGB2LAB)[mask, 1:]
    return mean, local


def compare(out, prior, identities):
    all_cases = []
    for case in identities:
        folder = out / 'cases' / case['case_id']
        source = prior / 'cases' / case['case_id']
        original = read(source / 'result.json')
        assert read(folder / 'generation.json') == read(source / 'generation.json')
        mask = np.asarray(Image.open(folder / 'mask.png')) > 0
        anchors = {seed: lab_fields(folder / f'anchor_{seed}.png', mask)[1] for seed in ('s0', 's1')}
        rows = {}
        for suffix in case['rows']:
            goal, local_goal = lab_fields(folder / (suffix + '_target.png'), mask)
            local_anchor = anchors[suffix.rsplit('_', 1)[1]]
            rows[suffix] = {}
            for arm, images in zip(ARMS, (source, folder)):
                mean, local = lab_fields(images / (suffix + '_b400.png'), mask)
                rows[suffix][arm] = dict(mean_ab=mean[1:].tolist(),
                    chroma_error=float(np.linalg.norm((mean - goal)[1:])),
                    luminance_error=float(abs(mean[0] - goal[0])),
                    local_chroma_error=float(np.linalg.norm(local - local_goal, axis=1).mean()),
                    local_chroma_to_common_anchor=float(np.linalg.norm(local - local_anchor, axis=1).mean()))
        pair_rows = {}
        for key, values in case['switches'].items():
            kind, first = key.split('_')[:2]
            second = SWITCHES[kind][first]
            phase = '_shift' if '_shift_' in key else ''
            suffix = phase + '_' + key.rsplit('_', 1)[1]
            a, b = rows[first + suffix], rows[second + suffix]
            pair_rows[key] = {}
            for arm, result in zip(ARMS, (original['switches'][key], values)):
                color_difference = np.asarray(a[arm]['mean_ab']) - np.asarray(b[arm]['mean_ab'])
                pair_rows[key][arm] = dict(switch=result['b400']['strict'],
                    chroma_difference=float(np.linalg.norm(color_difference)))
        aggregate = {}
        for arm, result in zip(ARMS, (original, case)):
            aggregate[arm] = {k: v for k, v in result['aggregate']['b400'].items() if k != 'valid_templates'}
            for metric in ('chroma_error', 'luminance_error', 'local_chroma_error', 'local_chroma_to_common_anchor'):
                aggregate[arm][metric] = float(np.mean([r[arm][metric] for r in rows.values()]))
            for kind in SWITCHES:
                aggregate[arm][kind + '_chroma_difference'] = float(np.mean(
                    [r[arm]['chroma_difference'] for k, r in pair_rows.items() if k.startswith(kind)]))
        result = dict(case_id=case['case_id'], split=case['split'], aggregate=aggregate,
                      color=rows, switches=pair_rows)
        write(folder / 'paired_color_comparison.json', result)
        all_cases.append(result)

    splits = {}
    for split in ('train', 'holdout'):
        cases = [c for c in all_cases if c['split'] == split]
        metrics = cases[0]['aggregate']['original_b']
        arms = {arm: {k: float(np.mean([c['aggregate'][arm][k] for c in cases])) for k in metrics}
                for arm in ARMS}
        differences = {k: paired([c['aggregate']['paired_color_b'][k] - c['aggregate']['original_b'][k]
                                 for c in cases]) for k in metrics}
        counts = {}
        for kind, patterns in SWITCHES.items():
            for first in patterns:
                group = [r for c in cases for k, r in c['switches'].items() if k.startswith(f'{kind}_{first}_')]
                counts[f'{kind}_{first}'] = dict(n=len(group), **{a: sum(r[a]['switch'] for r in group) for a in ARMS})
        splits[split] = dict(n=len(cases), arms=arms, paired_difference=differences, strict_switch_counts=counts)
    return splits
