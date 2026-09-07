"""检查 E8c 仅新增文本模块，E5 的所有已有权重保持逐值不变。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.e8_checkpoint_audit import (
    COMPONENTS, compare_checkpoints, load_checkpoint, summarize_differences,
)


REQUIRED_COMPONENTS = ("unet", "ref_unet", "texture_adapter", "bf_texture_conditioner", "tcpm_lite")
CSV_FIELDS = (
    "comparison", "component", "key", "status", "expected_trainable", "unexpected",
    "numel", "reference_dtype", "candidate_dtype", "changed_numel", "max_abs_delta",
    "rms_delta", "relative_l2", "precision_cast_only", "reference_nonfinite", "candidate_nonfinite",
    "reference_shape", "candidate_shape",
)


def audit_e8c(reference, candidate):
    errors, compatible_differences = [], []
    original_reference = reference
    reference_meta, candidate_meta = reference.get("meta", {}), candidate.get("meta", {})
    for label, checkpoint in (("e5", reference), ("text", candidate)):
        for component in REQUIRED_COMPONENTS:
            value = checkpoint.get(component)
            if not isinstance(value, dict) or not any(isinstance(item, torch.Tensor) for item in value.values()):
                errors.append({"checkpoint": label, "component": component, "error": "required_component_missing_or_empty"})
    reference_bf = reference.get("bf_texture_conditioner", {})
    if reference_meta.get("resampler_training", "off") != "off":
        errors.append({"checkpoint": "e5", "error": "initialization_training_mode_must_be_off",
                       "actual": reference_meta.get("resampler_training")})
    if (isinstance(reference_bf, dict) and any(key.startswith("text_guidance.") for key in reference_bf)) or reference_meta.get("text_guidance_dim", 0):
        errors.append({"checkpoint": "e5", "error": "initialization_already_contains_text_guidance"})
    if candidate_meta.get("resampler_training") != "text_only":
        errors.append({"checkpoint": "text", "error": "resampler_training_must_be_text_only",
                       "actual": candidate_meta.get("resampler_training", "<missing>")})

    # 已核实的旧格式差异：禁用的 AA-TCR 以前不保存，新格式保存空字典。
    if ("aa_tcr_fuser" not in reference and isinstance(candidate.get("aa_tcr_fuser"), dict)
            and not candidate["aa_tcr_fuser"] and not reference_meta.get("use_aa_tcr_fuse", 0)
            and not candidate_meta.get("use_aa_tcr_fuse", 0)):
        reference = dict(reference)
        reference["aa_tcr_fuser"] = {}
        compatible_differences.append("disabled_aa_tcr_missing_to_empty")

    rows = compare_checkpoints(reference, candidate, "E5_vs_E8c", candidate_has_text=True)
    for row in rows:
        is_text = row["component"] == "bf_texture_conditioner" and row["key"].startswith("text_guidance.")
        # 必须覆盖 E8a/E8b 的旧白名单：原查询和视觉重采样器在 E8c 中也冻结。
        row["expected_trainable"] = is_text and row["status"] == "allowed_added_text"
        if is_text:
            row["unexpected"] = not row["expected_trainable"]
        else:
            if row["status"] == "changed_trainable":
                row["status"] = "unexpected_change"
            if (row["status"] == "precision_cast_only" and row["changed_numel"] == 0
                    and not row["reference_nonfinite"] and not row["candidate_nonfinite"]):
                row["status"] = "value_preserving_dtype_change"
            row["unexpected"] = row["status"] not in ("unchanged", "value_preserving_dtype_change")

    violations = [row for row in rows if row["unexpected"]]
    components = sorted(set(COMPONENTS) | {row["component"] for row in rows})
    passed = not errors and not violations
    report = {
        "status": "passed" if passed else "failed", "check_completed": True,
        "frozen_passed": passed, "execution_failed": False,
        "scope": "仅允许新增完整的 text_guidance；所有 E5 已有参数逐值不变，量化造成的数值变化也不放行。",
        "errors": errors, "compatible_format_differences": compatible_differences,
        "component_presence": {name: {"e5": name in original_reference, "text": name in candidate} for name in components},
        "allowed_text_tensors": sum(row["expected_trainable"] for row in rows),
        "frozen_tensors_checked": sum(not row["expected_trainable"] and row["reference_dtype"] is not None for row in rows),
        "violation_count": len(violations), "violations": violations,
        "component_summary": summarize_differences(rows),
    }
    return report, rows


def write_outputs(output, report, rows):
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "frozen_differences.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if row["status"] != "unchanged":
                writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5-ckpt", required=True)
    parser.add_argument("--text-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    rows = []
    report = {"status": "failed", "check_completed": False, "frozen_passed": False,
              "execution_failed": True, "errors": []}
    try:
        torch.set_num_threads(4)
        print("[E8c] 加载 E5 与文本模块权重，执行 CPU 只读冻结检查。", flush=True)
        reference, candidate = load_checkpoint(args.e5_ckpt), load_checkpoint(args.text_ckpt)
        report, rows = audit_e8c(reference, candidate)
    except Exception as exc:
        report["errors"].append({"error": type(exc).__name__, "message": str(exc)})
    report["checkpoints"] = {"e5": str(Path(args.e5_ckpt).resolve()), "text": str(Path(args.text_ckpt).resolve())}
    try:
        write_outputs(output, report, rows)
    except Exception as exc:
        report.update(status="failed", frozen_passed=False, execution_failed=True)
        report["errors"].append({"error": "output_write_failed", "message": str(exc)})
        try:
            (output / "frozen_check.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except Exception:
            pass  # 目录本身不可写时只能通过退出码和标准错误报告。
        print("[E8c] failed：检查结果无法写入 %s：%s" % (output, exc), file=sys.stderr, flush=True)
        return 1
    print("[E8c] %s：%s" % (report["status"], output / "frozen_check.json"), flush=True)
    return 0 if report["frozen_passed"] and report["check_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
