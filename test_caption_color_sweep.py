#!/usr/bin/env python
"""验证 caption 颜色响应实验的实验设计不变量。

这些条件一旦被破坏, 实验会**静默作废** —— 生成图照样出得来, 但结论无效:

  1. 同一 base 样本的所有颜色变体必须共用同一个 seed。
     否则颜色效应与噪声效应混在一起。run_fixed_benchmark 的默认策略是
     seed = base + sample_id, 若把变体当成独立样本就会踩这个坑。
  2. 同一 base 的 sketch / texture / target 必须完全不变。
     颜色词是唯一变量。
  3. 各变体的 prompt 必须两两不同。
     若替换失败而静默返回原串, 就变成了 12 次重复实验。
  4. 每个 base 恰好有一档是原始颜色(作为对照)。
  5. 只接受"caption 里恰好一个颜色词"的样本, 否则"改了哪一个"成为混淆变量。

运行:
    python test_caption_color_sweep.py
    python -m pytest test_caption_color_sweep.py -v    # 若装了 pytest
"""
import collections
import json
import os
import sys
from unittest import mock

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import pytest
except ImportError:
    pytest = None

    class _Skipped(Exception):
        pass

    class _PytestShim(object):
        Skipped = _Skipped

        @staticmethod
        def skip(reason=""):
            raise _Skipped(reason)

    pytest = _PytestShim()

from tools.sweep_caption_color import (
    DEFAULT_COLORS,
    _find_generated,
    _prepare_generated_panel,
    find_color_words,
    replace_color_word,
)

PLAN = os.path.join(REPO, "output_eval/caption_color_sweep/e5/plan.json")


# ---------------------------------------------------------------------------
# 颜色词定位与替换的单元测试
# ---------------------------------------------------------------------------
def test_find_single_color_word():
    hits = find_color_words("a black sleeveless dress with a high collar")
    assert len(hits) == 1
    assert hits[0][2].lower() == "black"


def test_find_multiple_color_words():
    hits = find_color_words("a red and blue striped shirt")
    assert len(hits) == 2
    assert {h[2].lower() for h in hits} == {"red", "blue"}


def test_no_substring_false_positive():
    """颜色词必须按整词匹配, 不能命中单词内部。"""
    assert find_color_words("a blackened redwood pattern") == []
    assert find_color_words("greyhound print tee") == []


def test_replace_preserves_rest_of_caption():
    cap = "a black high-waisted leather legging with a shiny finish"
    out = replace_color_word(cap, "yellow")
    assert out == "a yellow high-waisted leather legging with a shiny finish"
    assert out != cap


def test_replace_case_insensitive():
    out = replace_color_word("A Black Dress", "green")
    assert out == "A green Dress"


def test_replace_rejects_multi_color():
    """多颜色词必须拒绝, 而不是替换第一个 —— 否则混淆变量。"""
    assert replace_color_word("a red and blue shirt", "green") is None


def test_replace_rejects_no_color():
    assert replace_color_word("a floral summer dress", "green") is None


def test_replace_to_every_default_color_is_distinct():
    cap = "a black cotton tee"
    outs = {replace_color_word(cap, c) for c in DEFAULT_COLORS}
    assert None not in outs
    assert len(outs) == len(DEFAULT_COLORS), "各颜色档必须产出不同的 prompt"


def test_score_uses_generated_panel_not_mask():
    """评分必须排除 inference 产出的 *_mask.png，并裁出右侧生成面板。"""
    with mock.patch("tools.sweep_caption_color.os.path.isdir", return_value=True), \
         mock.patch("tools.sweep_caption_color.os.listdir",
                    return_value=["sample_mask.png", "sample.jpg"]):
        assert _find_generated("/out").endswith("sample.jpg")

    with mock.patch("tools.sweep_caption_color._find_generated",
                    return_value="/out/sample.jpg"), \
         mock.patch("tools.sweep_caption_color.os.path.isfile", return_value=False), \
         mock.patch("tools.sweep_caption_color.shutil.copy2") as copy, \
         mock.patch("tools.sweep_caption_color.extract_generated_panel") as extract:
        panel = _prepare_generated_panel("/out")
        assert panel.endswith("generated_panel.png")
        copy.assert_called_once_with("/out/sample.jpg", panel)
        extract.assert_called_once_with(panel)


# ---------------------------------------------------------------------------
# plan 的实验设计不变量
# ---------------------------------------------------------------------------
def _load_plan():
    if not os.path.isfile(PLAN):
        pytest.skip("未生成 plan.json, 先跑 --stage plan")
    with open(PLAN, encoding="utf-8") as f:
        return json.load(f)


def _by_base(plan):
    groups = collections.defaultdict(list)
    for item in plan["items"]:
        groups[item["base_id"]].append(item)
    return groups


def test_plan_seed_constant_within_base():
    """最关键的一条: 同一 base 的所有变体共用一个 seed。"""
    groups = _by_base(_load_plan())
    bad = {b: sorted({x["seed"] for x in g}) for b, g in groups.items()
           if len({x["seed"] for x in g}) != 1}
    assert not bad, "seed 在 base 内不唯一, 颜色效应会与噪声混淆: %s" % dict(list(bad.items())[:3])


def test_plan_seed_differs_across_base():
    """不同 base 之间 seed 应当不同, 避免所有样本共用一个噪声。"""
    groups = _by_base(_load_plan())
    seeds = {g[0]["seed"] for g in groups.values()}
    assert len(seeds) == len(groups), "不同 base 之间 seed 重复了"


def test_plan_conditions_fixed_within_base():
    groups = _by_base(_load_plan())
    for base, g in groups.items():
        for key in ("sketch_path", "texture_path", "target_path"):
            assert len({x[key] for x in g}) == 1, "base %s 的 %s 变了" % (base, key)


def test_plan_prompts_distinct_within_base():
    groups = _by_base(_load_plan())
    bad = [b for b, g in groups.items() if len({x["prompt"] for x in g}) != len(g)]
    assert not bad, "以下 base 的 prompt 有重复(替换静默失败): %s" % bad[:5]


def test_plan_exactly_one_original_per_base():
    groups = _by_base(_load_plan())
    bad = {b: sum(x["is_original"] for x in g) for b, g in groups.items()
           if sum(x["is_original"] for x in g) != 1}
    assert not bad, "原始颜色档不唯一: %s" % dict(list(bad.items())[:3])


def test_plan_every_base_has_all_colors():
    plan = _load_plan()
    groups = _by_base(plan)
    n = len(plan["colors"])
    bad = {b: len(g) for b, g in groups.items() if len(g) != n}
    assert not bad, "变体数不足: %s" % dict(list(bad.items())[:3])


def test_plan_output_dirs_unique():
    """输出目录必须两两不同, 否则变体互相覆盖。"""
    plan = _load_plan()
    dirs = [x["out_dir"] for x in plan["items"]]
    assert len(set(dirs)) == len(dirs), "out_dir 有碰撞, 变体会互相覆盖"


def test_plan_records_mask_backend():
    plan = _load_plan()
    assert plan.get("mask_backend", {}).get("mask_backend") in ("opencv", "pillow_fallback")


def test_plan_all_source_files_exist():
    plan = _load_plan()
    missing = [
        (x["base_id"], key)
        for x in plan["items"][:60]
        for key in ("sketch_path", "texture_path", "target_path")
        if not os.path.isfile(x[key])
    ]
    assert not missing, "源文件缺失: %s" % missing[:5]


# ---------------------------------------------------------------------------
# 与 run_fixed_benchmark 的生成口径一致性 —— 防漂移
# ---------------------------------------------------------------------------
def _benchmark_passed_flags():
    """从 run_fixed_benchmark.py 源码里抽出它传给 inference 的 flag 名。"""
    import re

    src = open(os.path.join(REPO, "tools/run_fixed_benchmark.py"), encoding="utf-8").read()
    start = src.index('"inference_IMAGGarment-1.py"')
    end = src.index("subprocess.run(cmd, check=True)", start)
    return set(re.findall(r'"(--[a-zA-Z0-9_]+)"', src[start:end]))


def _sweep_passed_flags(override=False):
    from tools.sweep_caption_color import _inference_cmd, build_parser
    from tools.run_fixed_benchmark import experiment_to_flags, mode_to_flags

    argv = ["--gam_ckpt", "/x/g.pt", "--texture_ckpt", "/x/t.bin"]
    if override:
        argv.append("--override_generation_params")
    args = build_parser().parse_args(argv)
    item = {
        "prompt": "p", "seed": 1, "out_dir": "/tmp/o",
        "sketch_path": "/x/s.jpg", "texture_path": "/x/t.jpg",
    }
    cmd = _inference_cmd(
        args, item, mode_to_flags(args.mode),
        experiment_to_flags(args.run_name, args),
    )
    return {c for c in cmd if isinstance(c, str) and c.startswith("--")}


def test_sweep_flags_match_benchmark():
    """扫描必须与 benchmark 传完全同一套 flag。

    既有 E5/E7a 报告全部跑在 inference 的默认生成参数上(benchmark 不传
    width/height/guidance_scale/ipa_scale)。扫描若多传或少传, 就不在同一
    生成条件下, 判定无法迁移到那些报告上。
    """
    bench = _benchmark_passed_flags()
    sweep = _sweep_passed_flags()
    # benchmark 里的 gate_trace 是条件分支, 扫描不需要
    bench -= {"--balanced_gate_trace_path", "--balanced_gate_trace_sample_id"}
    assert sweep == bench, (
        "flag 集合不一致\n  扫描多传: %s\n  扫描少传: %s"
        % (sorted(sweep - bench), sorted(bench - sweep))
    )


def test_generation_params_not_passed_by_default():
    """默认必须不传这四个, 否则与报告不可比。"""
    sweep = _sweep_passed_flags()
    for flag in ("--width", "--height", "--guidance_scale", "--ipa_scale"):
        assert flag not in sweep, "%s 默认不该传给 inference" % flag


def test_override_flag_adds_generation_params():
    sweep = _sweep_passed_flags(override=True)
    for flag in ("--width", "--height", "--guidance_scale", "--ipa_scale"):
        assert flag in sweep


def test_all_sweep_flags_exist_in_inference():
    """扫描传的每个 flag 都必须在 inference 的 argparse 里, 否则集群上直接报错。"""
    import re

    src = open(os.path.join(REPO, "inference_IMAGGarment-1.py"), encoding="utf-8").read()
    declared = set(re.findall(r"""add_argument\(\s*['"](--[a-zA-Z0-9_]+)['"]""", src))
    for flag in sorted(_sweep_passed_flags(override=True)):
        assert flag in declared, "inference 没有 %s" % flag


# ---------------------------------------------------------------------------
def _standalone_main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed, skipped = [], []
    for name, fn in tests:
        try:
            fn()
        except getattr(pytest, "Skipped", ()) as exc:
            print("SKIP %-52s %s" % (name, exc))
            skipped.append(name)
        except AssertionError as exc:
            print("FAIL %-52s %s" % (name, exc))
            failed.append(name)
        except Exception as exc:  # noqa: BLE001
            print("ERROR %-51s %s: %s" % (name, type(exc).__name__, exc))
            failed.append(name)
        else:
            print("PASS %s" % name)
    print("\n%d passed, %d failed, %d skipped"
          % (len(tests) - len(failed) - len(skipped), len(failed), len(skipped)))
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(pytest, "main"):
        sys.exit(pytest.main([__file__, "-v"]))
    sys.exit(_standalone_main())
