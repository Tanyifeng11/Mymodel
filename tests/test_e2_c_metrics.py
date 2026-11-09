import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.e2_c_evaluation import region_metrics, checkpoint_record


def fixture():
    mask = np.zeros((64, 48), bool); mask[8:56, 8:40] = 1
    gray = np.full((64, 48, 3), 255, np.uint8)
    gray[8:56, 8] = 0; gray[8:56, 39] = 0
    gray[8, 8:40] = 0; gray[55, 8:40] = 0
    sketch = Image.fromarray(gray)
    image = np.full_like(gray, 255); image[mask] = [180, 20, 20]
    red = Image.new('RGB', (48, 64), (180, 20, 20)); blue = Image.new('RGB', (48, 64), (20, 20, 180))
    return Image.fromarray(image), sketch, red, blue, mask


def test_region_partition_and_reference_color_direction():
    image, sketch, red, blue, mask = fixture()
    out = region_metrics(image, sketch, image, red, blue, mask)
    assert out['body_pixels'] + out['band_pixels'] + out['background_pixels'] == mask.size
    assert out['background_mae'] == 0 and out['boundary_f1_r2'] == 1.
    assert out['body_color_old'] < 1e-4 and out['body_color_new'] > 20
    assert out['band_color_old'] < 1e-4 and not out['failed_generation']


def test_failed_output_is_scored_as_failure_not_silently_excluded():
    image, sketch, red, blue, mask = fixture()
    bad = Image.new('RGB', image.size, 'white')
    out = region_metrics(bad, sketch, image, red, blue, mask)
    assert out['failed_generation'] and out['boundary_f1_r2'] == 0
    assert out['band_edge_f1'] == 0 and out['band_generated_edge_pixels'] == 0
    assert out['band_gradient_spread_pixels'] is None


def test_fixed_probes_reuse_c0_checkpoint_without_fake_training(tmp_path):
    folder = tmp_path / 'training/C0'; folder.mkdir(parents=True)
    value = dict(status='completed', updates=2000, checkpoint_sha256='audited_baseline')
    (folder / 'complete.json').write_text(json.dumps(value))
    assert checkpoint_record(tmp_path, 'C1') == checkpoint_record(tmp_path, 'C2') == value
    assert not (tmp_path / 'training/C1').exists()
