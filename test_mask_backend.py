#!/usr/bin/env python
"""验证 mask 形态学后端的影响, 并防止 cv2 fallback 再次静默发生。

背景: 既有全部评测报告都跑在 cv2 缺失的环境里(走 Pillow fallback)。两条路径的
形态学运算不等价, 导致 auto 策略下 sketch_flood_fill 通过率在 70/500 与 364/500
之间跳, 全部 mask 派生指标不可比。

本测试做三件事:
  1. mask_backend_info() 如实报告当前后端;
  2. 在真实数据上**实测**两条后端产出的 mask 确实不同(证明风险是真的, 不是理论);
  3. cv2 缺失时必须发出 RuntimeWarning(不再静默)。

运行:
    python -m pytest test_mask_backend.py -v
"""
import importlib
import os
import subprocess
import sys
import warnings

import numpy as np

try:
    import pytest
except ImportError:  # 本项目的 IMAGGarment 环境未装 pytest, 允许直接 python 运行
    pytest = None

    class _Skipped(Exception):
        pass

    class _PytestShim(object):
        Skipped = _Skipped

        class mark(object):
            @staticmethod
            def skipif(condition, reason=""):
                def deco(fn):
                    fn.__skip_if__ = (bool(condition), reason)
                    return fn

                return deco

        @staticmethod
        def skip(reason=""):
            raise _Skipped(reason)

    pytest = _PytestShim()

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get(
    "BF_TRAIN_ROOT", "/mnt/f/fuxian/dataset/datasets/BF/training"
)
SAMPLE_CSV = os.path.join(REPO, "eval_outputs/report_e7a/metrics_per_sample.csv")


def _sample_ids(limit=40):
    """从既有 per-sample CSV 取一批真实样本 id。"""
    import csv

    if not os.path.isfile(SAMPLE_CSV):
        return []
    ids = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(os.path.basename(row.get("sketch_path") or ""))[0]
            if stem:
                ids.append(stem)
            if len(ids) >= limit:
                break
    return ids


def _have_data():
    return os.path.isdir(os.path.join(DATA_ROOT, "sketch"))


# --------------------------------------------------------------------------
# 1. 后端自报
# --------------------------------------------------------------------------
def test_backend_info_shape():
    import garment_mask_utils as gmu

    info = gmu.mask_backend_info()
    assert set(info) == {"mask_backend", "cv2_version", "cv2_import_error"}
    assert info["mask_backend"] in ("opencv", "pillow_fallback")
    if info["mask_backend"] == "opencv":
        assert info["cv2_version"], "opencv 后端必须报出版本号"
        assert info["cv2_import_error"] is None
    else:
        assert info["cv2_import_error"], "fallback 必须保留真实的 import 失败原因"


def test_backend_recorded_by_benchmark():
    """run_fixed_benchmark 必须把后端写进 diagnostics, 否则跨 run 无法判断可比性。"""
    src = open(os.path.join(REPO, "tools/run_fixed_benchmark.py"), encoding="utf-8").read()
    assert "mask_backend_info()" in src
    assert "mask_geometry_counts" in src


# --------------------------------------------------------------------------
# 2. 实测两条后端不等价 —— 这是本文件最重要的一条
# --------------------------------------------------------------------------
_PROBE = r'''
import json, os, sys
sys.path.insert(0, %(repo)r)
if %(block)s:
    sys.modules["cv2"] = None
import warnings
warnings.simplefilter("ignore")
import garment_mask_utils as gmu
from eval.eval_utils import prepare_evaluation_masks

out = {"backend": gmu.mask_backend_info()["mask_backend"], "pixels": {}, "source": {}}
for sid in %(ids)r:
    sketch = os.path.join(%(root)r, "sketch", sid + ".jpg")
    target = os.path.join(%(root)r, "cloth", sid + ".jpg")
    if not (os.path.isfile(sketch) and os.path.isfile(target)):
        continue
    b = prepare_evaluation_masks(
        (384, 512), sketch_path=sketch, target_path=target,
        gen_path=target, mask_policy="auto",
    )
    m = b.get("garment")
    out["pixels"][sid] = int(m.sum()) if m is not None else -1
    out["source"][sid] = b.get("stats", {}).get("mask_source")
print("@@@" + json.dumps(out))
'''


def _run_probe(block_cv2, ids):
    code = _PROBE % {
        "repo": REPO,
        "block": "True" if block_cv2 else "False",
        "ids": ids,
        "root": DATA_ROOT,
    }
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@@")]
    assert line, "probe 无输出:\n" + proc.stdout[-2000:]
    import json

    return json.loads(line[-1][3:])


@pytest.mark.skipif(not _have_data(), reason="本地无 BF 数据集")
def test_two_backends_produce_different_masks():
    """两条后端在真实草图上产出的 mask 必须被观测到不同。

    如果这条测试有一天变成"两者相同", 说明 fallback 实现被改成等价的了 ——
    那是好事, 但必须显式确认, 而不是默认。
    """
    ids = _sample_ids(limit=40)
    if not ids:
        pytest.skip("找不到样本 id")

    with_cv2 = _run_probe(False, ids)
    without = _run_probe(True, ids)

    assert with_cv2["backend"] == "opencv", "本机应装有 opencv; 实际: %s" % with_cv2["backend"]
    assert without["backend"] == "pillow_fallback"

    common = sorted(set(with_cv2["pixels"]) & set(without["pixels"]))
    assert len(common) >= 10, "可比样本太少: %d" % len(common)

    a = np.array([with_cv2["pixels"][k] for k in common], dtype=float)
    b = np.array([without["pixels"][k] for k in common], dtype=float)
    differing = int((a != b).sum())

    src_a = [with_cv2["source"][k] for k in common]
    src_b = [without["source"][k] for k in common]
    source_flips = sum(1 for x, y in zip(src_a, src_b) if x != y)

    print(
        "\n[实测] n=%d  像素数不同=%d  mask_source 翻转=%d"
        % (len(common), differing, source_flips)
    )
    print("  opencv  sketch_flood_fill=%d" % src_a.count("sketch_flood_fill"))
    print("  pillow  sketch_flood_fill=%d" % src_b.count("sketch_flood_fill"))

    assert differing > 0, (
        "两条后端产出了完全相同的 mask。若这是有意为之请更新本测试, "
        "否则说明 probe 没有真正屏蔽 cv2。"
    )
    assert source_flips > 0, "mask_source 未发生翻转, 与历史观测(70/500 vs 364/500)不符"


# --------------------------------------------------------------------------
# 3. fallback 不得静默
# --------------------------------------------------------------------------
def test_fallback_warns():
    code = (
        "import sys, warnings\n"
        "sys.path.insert(0, %r)\n"
        "sys.modules['cv2'] = None\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import garment_mask_utils\n"
        "    msgs = [str(x.message) for x in w if x.category is RuntimeWarning]\n"
        "assert msgs, 'cv2 缺失时没有发出 RuntimeWarning'\n"
        "assert 'Pillow' in msgs[0], msgs[0]\n"
        "assert garment_mask_utils.mask_backend_info()['cv2_import_error']\n"
        "print('WARNED_OK')\n" % REPO
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "WARNED_OK" in proc.stdout


def _standalone_main():
    """无 pytest 时的极简 runner。"""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed, skipped = [], []
    for name, fn in tests:
        skip_if = getattr(fn, "__skip_if__", None)
        if skip_if and skip_if[0]:
            print("SKIP %-46s %s" % (name, skip_if[1]))
            skipped.append(name)
            continue
        try:
            fn()
        except getattr(pytest, "Skipped", ()) as exc:
            print("SKIP %-46s %s" % (name, exc))
            skipped.append(name)
        except AssertionError as exc:
            print("FAIL %-46s %s" % (name, exc))
            failed.append(name)
        except Exception as exc:  # noqa: BLE001
            print("ERROR %-45s %s: %s" % (name, type(exc).__name__, exc))
            failed.append(name)
        else:
            print("PASS %s" % name)
    print(
        "\n%d passed, %d failed, %d skipped"
        % (len(tests) - len(failed) - len(skipped), len(failed), len(skipped))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(pytest, "main"):
        sys.exit(pytest.main([__file__, "-v", "-s"]))
    sys.exit(_standalone_main())
