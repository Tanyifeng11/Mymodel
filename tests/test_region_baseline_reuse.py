import pytest

from tools import region_fusion_experiment as experiment


@pytest.fixture
def reuse_environment(tmp_path, monkeypatch):
    root, history, parent, source = [tmp_path / name for name in ('new', 'history', 'parent', 'source')]
    for name, value in [('ROOT', root), ('HISTORY', history), ('PARENT', parent)]:
        monkeypatch.setattr(experiment, name, value)
    monkeypatch.setattr(experiment.core, 'SOURCE', source)
    e1 = tmp_path / 'e1.pt'
    e1.write_bytes(b'frozen E1')
    monkeypatch.setattr(experiment.core, 'E1', e1)
    initial = root / 'initialization/bf_fusion.pt'
    initial.parent.mkdir(parents=True)
    initial.write_bytes(b'frozen BF initialization')
    checkpoint = history / 'training/BF-SD-T3/step28365.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'trained historical model')
    hashes = {}
    for name in ('models/bf_original_fusion.py', 'models/bf_sd_skip.py',
                 'train_GAM_texture_joint.py', 'adapter/attention_processor.py'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('unchanged source', encoding='utf-8')
        hashes[name] = experiment.sha(path)
    experiment.write(history / 'training/BF-SD-T3/complete.json', dict(updates=28365,
        world=2, effective_batch=8, checkpoint_sha256=experiment.sha(checkpoint), source_sha256=hashes))
    (checkpoint.parent / 'training_log.csv').write_text('step,lr\n28365,0\n')
    experiment.write(history / 'protocol.json', dict(e1_sha256=experiment.sha(e1),
        fusion_sha256=experiment.sha(initial), objectives={
            'T3': 'base + .1 reference RGB body + .03816244801608925 GT L1 rest'}))
    for base in (root, history):
        experiment.write(base / 'training_manifest.json', [dict(cloth='train-only.png')])
    experiment.write(root / 'protocol.json', dict(arms=experiment.ARMS, gpus_per_arm=2))
    rows = []
    for index in range(192):
        row = dict(id=str(index), condition='original', sketch_sha256='sketch',
            reference_sha256='reference', gt_sha256='gt', caption='a garment', seed=42)
        rows.append(row)
        image = parent / 'outputs/BF-SD-T3/original' / (str(index) + '.png')
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'fixed generated pixels')
        experiment.write(image.with_suffix('.json'), dict(output_sha256=experiment.sha(image),
            checkpoint_sha256=experiment.sha(checkpoint), input_sketch_sha256='sketch',
            input_reference_sha256='reference', seed=42, GT_used_in_inference=False))
    experiment.write(root / 'evaluation.json', rows)
    experiment.write(parent / 'evaluation.json', rows)
    duplicate = root / 'training/bf'
    duplicate.mkdir(parents=True)
    (duplicate / 'training_log.csv').write_text('cancelled partial training\n')
    return root, history, source


def test_reuses_all_outputs_and_preserves_cancelled_work(reuse_environment):
    root, history, _ = reuse_environment
    experiment.reuse()
    complete = experiment.read(root / 'training/bf/complete.json')
    assert complete['training_reused'] and complete['outputs_reused']
    assert complete['checkpoint_source'] == str(history / 'training/BF-SD-T3/step28365.pt')
    assert len(list((root / 'training/bf/outputs').rglob('*.png'))) == 192
    assert (root / 'training/bf_cancelled_122614/training_log.csv').read_text() == 'cancelled partial training\n'
    protocol = experiment.read(root / 'protocol.json')
    assert protocol['new_training_arms'] == ['concat', 'region']
    metadata = experiment.read(root / 'training/bf/outputs/original/0.json')
    assert metadata['caption_source'] == 'frozen parent evaluation manifest'
    assert metadata['reused_from'].endswith('0.png')
    experiment.reuse()
    assert len(list((root / 'training/bf/outputs').rglob('*.png'))) == 192


def test_rejects_changed_historical_source_before_replacing_partial_work(reuse_environment):
    root, _, source = reuse_environment
    (source / 'models/bf_original_fusion.py').write_text('different fusion', encoding='utf-8')
    with pytest.raises(AssertionError):
        experiment.reuse()
    assert not (root / 'training/bf/complete.json').exists()
    assert not (root / 'training/bf_cancelled_122614').exists()


def test_rejects_unmatched_input_manifest(reuse_environment):
    root, _, _ = reuse_environment
    rows = experiment.read(root / 'evaluation.json')
    rows[0]['caption'] = 'changed text'
    experiment.write(root / 'evaluation.json', rows)
    with pytest.raises(AssertionError):
        experiment.reuse()
    assert not (root / 'audit/reused_baseline.json').exists()
