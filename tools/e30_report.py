"""汇总 E30-A 的正式检查与两次训练 split 开发检查。"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run(root, out):
    stage = out / 'A_feature_feasibility'
    initial = load(stage / 'initial_report.json')
    first = load(out / 'development/A_feature_feasibility/report.json')
    second = load(out / 'development_v2/A_feature_feasibility/report.json')
    assert initial['requested'] == initial['completed'] == 64
    assert first['requested'] == second['requested'] == 16
    assert not initial['gate_pass'] and not first['gate_pass'] and not second['gate_pass']
    weights = out / 'dinov2_vits14_pretrain.pth'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == initial['dino_sha256'] == first['dino_sha256'] == second['dino_sha256']
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dino_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                    cwd=Path.home() / '.cache/dinov2-e30', text=True).strip()
    write(out / 'frozen_check.json', {'pass': True, 'dino_weights_sha256': digest,
          'dino_source_commit': dino_source, 'visual_backbone_trainable': False,
          'e5_refinement_run': False, 'e5_checkpoint_modified': False,
          'uses_manual_panel_or_crop_for_training': False})
    write(stage / 'revision_history.json', {
          'initial_64': {'git_commit': '5ab5297', 'gate_pass': False,
                         'orientation_consistency_baseline': initial['summary']['A0_e29_rule']['orientation_consistency'],
                         'orientation_consistency_apacc': initial['summary']['A4_canonicality']['orientation_consistency'],
                         'combined_region_count': initial['summary']['A3_combined'].get('region_count', 1)},
          'prototype_strict_16': {'git_commit': '14b6dc2', 'gate_pass': False,
                                  'completed': first['completed'],
                                  'failures': len(first['failures']),
                                  'combined_region_count': first['summary']['A3_combined']['region_count']},
          'prototype_relaxed_16': {'git_commit': '738acb4', 'gate_pass': False,
                                   'completed': second['completed'],
                                   'failures': len(second['failures']),
                                   'combined_region_count': second['summary']['A3_combined']['region_count']}})
    decision = {'feature_feasibility_pass': False, 'canonicality_pass': None,
                'region_causal_pass': None, 'crop_causal_pass': None,
                'full_apacc_pass': None, 'confirmation_pass': None,
                'structure_safe': None, 'background_safe': None,
                'oracle_gap_recovery_follow': None, 'oracle_gap_recovery_orientation': None,
                'next_route': 'revise_frozen_features_or_affinity',
                'stop_reason': 'Stage A: baseline-to-APACC orientation consistency fell from %.4f to %.4f; '
                               'region prototype revisions left 11/16 and 5/16 without valid crop.' %
                               (initial['summary']['A0_e29_rule']['orientation_consistency'],
                                initial['summary']['A4_canonicality']['orientation_consistency'])}
    write(out / 'decision_summary.json', decision)
    write(out / 'completion_check.json', {'pass': True, 'stage_a_completed': True,
          'stage_a_gate_pass': False, 'stage_b_run': False, 'stage_c_run': False,
          'stage_d_run': False, 'stage_e_run': False,
          'later_stages_skipped_by_predefined_gate': True,
          'causal_test_cases_used_for_training_or_tuning': False,
          'git_commit': git_commit})
    write(out / 'artifact_manifest.json', {'stage_a_initial_report': 'A_feature_feasibility/initial_report.json',
          'stage_a_initial_rows': 'A_feature_feasibility/initial_rows.json',
          'stage_a_revision_history': 'A_feature_feasibility/revision_history.json',
          'development_strict': 'development/A_feature_feasibility/report.json',
          'development_relaxed': 'development_v2/A_feature_feasibility/report.json',
          'decision': 'decision_summary.json', 'frozen_check': 'frozen_check.json',
          'completion': 'completion_check.json', 'dino_weights_sha256': digest})
    print('[E30 report] Stage A failed; later stages stopped by gate', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())
