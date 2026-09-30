"""E30-A2 停止规则与结果完整性汇总；只写服务器 JSON，不生成文档。"""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def ci(values, seed=20260930):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main(out):
    protocol = read(out / 'protocol.json')
    split = read(out / 'split_manifest.json')
    frozen = read(out / 'frozen_check.json')
    front = read(out / 'A20_frozen_adaptive/summary.json')
    embedding = read(out / 'A21_embedding/summary.json')
    pilot = read(out / 'A22_segmentation/seed42/summary.json')
    arms = pilot['arms']
    fixed = read(out / 'A22_segmentation/seed42/frozen_fixed/rows.json')
    learned = read(out / 'A22_segmentation/seed42/learned_adaptive/rows.json')
    dev_pairs = [(a, b) for a, b in zip(fixed, learned) if a['split'] == b['split'] == 'dev']
    region_delta = [b['region_count'] - a['region_count'] for a, b in dev_pairs]
    coverage_delta = [int(b['valid_crop']) - int(a['valid_crop']) for a, b in dev_pairs]
    errors = {'F2_over_fragmentation': [r['image'] for r in learned if r['region_count'] > 10],
              'F8_no_valid_crop': [r['image'] for r in learned if not r['valid_crop']]}
    error_counts = {key: len(value) for key, value in errors.items()}
    pilot_stop = arms['learned_adaptive']['dev']['median_region_count'] > 10
    decision = {'stage': 'E30-A2', 'embedding_gate_pass': embedding['gate_pass'],
                'adaptive_segmentation_pass': pilot['gate_pass'],
                'canonicality_compatibility_pass': None,
                'effective_rank': {s: v['effective_rank'] for s, v in embedding['seeds'].items()},
                'valid_crop_coverage': arms['learned_adaptive']['dev']['valid_crop_coverage'],
                'stress_valid_crop_coverage': arms['learned_adaptive']['stress_dev']['valid_crop_coverage'],
                'orientation_consistency': arms['learned_adaptive']['dev']['orientation_consistency'],
                'period_consistency': arms['learned_adaptive']['dev']['period_consistency'],
                'identity': arms['learned_adaptive']['dev']['identity'],
                'homogeneity': arms['learned_adaptive']['dev']['homogeneity'],
                'contamination': arms['learned_adaptive']['dev']['contamination'],
                'median_region_count': arms['learned_adaptive']['dev']['median_region_count'],
                'region_stability': arms['learned_adaptive']['dev']['region_stability_iou'],
                'feature_bottleneck': 'embedding proxy improved; region discovery remains unresolved',
                'segmentation_bottleneck': True, 'pilot_stop_over_10_regions': pilot_stop,
                'next_route': 'segmentation_revision' if pilot_stop else 'A22_full_three_seed',
                'A23_status': 'not_run_gate_failed', 'E5_status': 'not_run_gate_failed',
                'interpretation': 'Positive/hard-negative pairs are automatic proxies, not pattern GT.'}
    write(out / 'decision_summary.json', decision)
    analysis = {'reference_unit': 'BF training reference image', 'bootstrap_resamples': 2000,
                'dev_count': len(dev_pairs), 'frozen_fixed_to_learned_adaptive': {
                    'region_count_mean_delta': float(np.mean(region_delta)),
                    'region_count_mean_delta_ci95': ci(region_delta),
                    'valid_crop_coverage_delta': float(np.mean(coverage_delta)),
                    'valid_crop_coverage_delta_ci95': ci(coverage_delta)},
                'failure_counts': error_counts, 'failure_images': errors,
                'ablation_status': {
                    'A_frozen_DINO_only': 'not_run_pilot_stop',
                    'B_no_geometry': 'not_run_pilot_stop',
                    'C_no_boundary_negatives': 'not_run_pilot_stop',
                    'D_no_periodic_positives': 'not_run_pilot_stop',
                    'E_learned_fixed': 'completed_seed42_pilot',
                    'F_frozen_adaptive': 'completed_seed42_pilot'},
                'limitations': [
                    'Stress-dev is deterministic, not pattern-family stratified.',
                    'Pair labels derive from automatic cues and were not independently annotated.',
                    'Segmentation pilot stopped after seed42 because median region count exceeded 10.',
                    'Region stability is based on real photometric augmentation only; shift/resize were not assessed.',
                    'Frozen-fixed pilot uses current prototype-constrained Stage A graph, not the original first-run graph.',
                    'Local high-frequency gradients replace the proposed true local FFT profile.',
                    'Training omits translation and rot90 positives and structural-edge, synthetic mixed, and contamination negatives.',
                    'Training batch samples 64 grid nodes, not 64 independent image crops; results apply to this simplified pilot only.']}
    write(out / 'A22_segmentation/pilot_analysis.json', analysis)
    required = [out / 'protocol.json', out / 'split_manifest.json',
                out / 'train_normalization.npz', out / 'frozen_check.json',
                out / 'A20_frozen_adaptive/summary.json', out / 'A21_embedding/summary.json',
                out / 'A22_segmentation/seed42/summary.json']
    for seed in (42, 43, 44):
        required += [out / 'A21_embedding' / ('seed%d' % seed) / 'adapter.pt',
                     out / 'A21_embedding' / ('seed%d' % seed) / 'audit.json',
                     out / 'A21_embedding' / ('seed%d' % seed) / 'pair_sampling_manifest.json']
    completion = {'result_complete_for_stopped_pilot': all(p.exists() for p in required),
                  'required_missing': [str(p.relative_to(out)) for p in required if not p.exists()],
                  'split_disjoint': len(set(split['train'] + split['dev'] + split['stress_dev'])) == 96,
                  'dino_unchanged': frozen['dino_changed_tensor_count'] == 0,
                  'e26_unchanged': not frozen['e26_changed'], 'e5_unchanged': not frozen['e5_changed'],
                  'train_seed_count': len(embedding['seeds']),
                  'segmentation_seed_count': 1,
                  'stop_rule_applied': pilot_stop and not pilot['gate_pass'],
                  'gate_pass': False}
    write(out / 'completion_check.json', completion)
    write(out / 'artifact_manifest.json', {'protocol': 'protocol.json',
          'split': 'split_manifest.json', 'frozen_baseline': 'A20_frozen_adaptive/summary.json',
          'embedding_gate': 'A21_embedding/summary.json',
          'segmentation_pilot': 'A22_segmentation/seed42/summary.json',
          'paired_bootstrap_and_errors': 'A22_segmentation/pilot_analysis.json',
          'decision': 'decision_summary.json', 'completion': 'completion_check.json'})
    print(json.dumps({'decision': decision, 'completion': completion,
                      'paired': analysis['frozen_fixed_to_learned_adaptive']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
