"""检查 E9 只新增局部旁路：E5 已有权重逐值不变，且 A/B 训练配置可比。"""

import argparse
import csv
import json
import re
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
# 同一个模块同时出现在 unet 和 texture_adapter(处理器 ModuleList)两处导出。
BRANCH_COMPONENTS = ("unet", "texture_adapter")
ADAPTER_MARKER = ".local_detail_adapter."
ADAPTER_TENSORS = ("alpha", "context_norm.bias", "context_norm.weight", "query_norm.bias",
                   "query_norm.weight", "to_k.weight", "to_out.weight", "to_q.weight", "to_v.weight")
VARIANT_SOURCE = {"a": "resampled", "b": "local", "c": "local"}
TEXTURE_ADAPTER_PREFIX = re.compile(r"^(\d+)\.local_detail_adapter\.")
# 与 train_GAM_texture_joint.validate_local_detail_base_config 同一组字段。
BASE_CONFIG_FIELDS = ("texture_num_tokens", "texture_mode", "texture_condition_mode",
                      "texture_preprocess_mode", "clip_hidden_layer", "width", "height",
                      "layer_group_enabled", "use_texture_gate", "use_tcpm_lite",
                      "tcpm_mask_inner_only", "region_kernel_size")
# A/B 之间唯一允许不同的是 local_detail_source。
PARITY_FIELDS = ("local_detail_grid", "local_detail_layer", "local_detail_dim", "local_detail_heads",
                 "local_detail_lr", "train_batch_size", "gradient_accumulation_steps",
                 "max_train_steps", "num_warmup_steps", "max_grad_norm", "mixed_precision",
                 "dataset_json_path", "data_root_path", "training_data_sha256", "seed")
CSV_FIELDS = (
    "comparison", "component", "key", "status", "expected_trainable", "unexpected",
    "numel", "reference_dtype", "candidate_dtype", "changed_numel", "max_abs_delta",
    "rms_delta", "relative_l2", "precision_cast_only", "reference_nonfinite", "candidate_nonfinite",
    "reference_shape", "candidate_shape",
)


def branch_keys(checkpoint):
    """按组件收集新增旁路权重的键，供前缀与完整性检查复用。"""
    return {name: sorted(key for key in checkpoint.get(name, {}) if ADAPTER_MARKER in key)
            for name in BRANCH_COMPONENTS}


def check_reference(reference, errors):
    """E9 A/B 都必须从原 E5 新训，起点自身不能带任何新增模块。"""
    meta = reference.get("meta", {})
    if meta.get("resampler_training", "off") != "off":
        errors.append({"checkpoint": "e5", "error": "initialization_training_mode_must_be_off",
                       "actual": meta.get("resampler_training")})
    if meta.get("text_guidance_dim", 0):
        errors.append({"checkpoint": "e5", "error": "initialization_already_contains_text_guidance"})
    if meta.get("local_detail_source", "off") != "off":
        errors.append({"checkpoint": "e5", "error": "initialization_already_contains_local_detail",
                       "actual": meta.get("local_detail_source")})
    for component in ("unet", "texture_adapter", "bf_texture_conditioner"):
        for key in reference.get(component, {}):
            if ADAPTER_MARKER in key or "text_guidance." in key:
                errors.append({"checkpoint": "e5", "component": component,
                               "error": "initialization_contains_new_module_weights", "key": key})
                break


def check_branch_weights(candidate, meta, errors):
    """新增权重必须完整、只落在声明的那一层，且两处导出逐值一致。"""
    layer = meta.get("local_detail_layer", "")
    if not layer:
        errors.append({"checkpoint": "e9", "error": "local_detail_layer_missing_in_meta"})
        return
    prefixes = {"unet": layer + ADAPTER_MARKER}
    found = branch_keys(candidate)
    indices = {match.group(1) for key in found["texture_adapter"]
               if (match := TEXTURE_ADAPTER_PREFIX.match(key))}
    if len(indices) != 1:
        errors.append({"checkpoint": "e9", "error": "texture_adapter_local_detail_index_not_unique",
                       "actual": sorted(indices)})
        return
    prefixes["texture_adapter"] = indices.pop() + ADAPTER_MARKER
    for component, prefix in prefixes.items():
        keys = found[component]
        if not keys:
            errors.append({"checkpoint": "e9", "component": component,
                           "error": "local_detail_weights_missing"})
            continue
        wrong_layer = [key for key in keys if not key.startswith(prefix)]
        if wrong_layer:
            errors.append({"checkpoint": "e9", "component": component,
                           "error": "local_detail_weights_on_unexpected_layer", "keys": wrong_layer[:8]})
        names = {key[len(prefix):] for key in keys if key.startswith(prefix)}
        if names != set(ADAPTER_TENSORS):
            errors.append({"checkpoint": "e9", "component": component,
                           "error": "local_detail_weights_incomplete",
                           "missing": sorted(set(ADAPTER_TENSORS) - names),
                           "unexpected": sorted(names - set(ADAPTER_TENSORS))})
    if all(found[component] for component in BRANCH_COMPONENTS):
        for name in ADAPTER_TENSORS:
            tensors = [candidate[component].get(prefixes[component] + name)
                       for component in BRANCH_COMPONENTS]
            if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
                continue
            first, second = tensors
            if first.shape != second.shape or not bool(torch.equal(first.float(), second.float())):
                errors.append({"checkpoint": "e9", "error": "local_detail_exports_disagree",
                               "key": name})


def check_candidate_meta(reference_meta, meta, expected_source, errors):
    source = meta.get("local_detail_source", "off")
    if expected_source and source != expected_source:
        errors.append({"checkpoint": "e9", "error": "local_detail_source_mismatch",
                       "expected": expected_source, "actual": source})
    elif source not in ("resampled", "local"):
        errors.append({"checkpoint": "e9", "error": "local_detail_source_must_be_enabled",
                       "actual": source})
    if meta.get("resampler_training", "off") != "off" or meta.get("text_guidance_dim", 0):
        errors.append({"checkpoint": "e9", "error": "local_detail_run_must_not_train_text_modules"})
    if int(meta.get("local_detail_grid", 0)) < 1:
        errors.append({"checkpoint": "e9", "error": "local_detail_grid_must_be_positive",
                       "actual": meta.get("local_detail_grid")})
    # 冻结了权重却改变 E5 的条件语义，A/B 结论同样不可比。
    for field in BASE_CONFIG_FIELDS:
        if field in reference_meta and field in meta and reference_meta[field] != meta[field]:
            errors.append({"checkpoint": "e9", "error": "base_config_differs_from_e5", "field": field,
                           "e5": reference_meta[field], "e9": meta[field]})


def audit_variant(reference, candidate, label, expected_source):
    """逐张量比较：只放行完整的新增旁路，其余一律要求逐值不变。"""
    errors, compatible_differences = [], []
    original_reference = reference
    reference_meta, candidate_meta = reference.get("meta", {}), candidate.get("meta", {})
    for checkpoint_label, checkpoint in (("e5", reference), (label, candidate)):
        for component in REQUIRED_COMPONENTS:
            value = checkpoint.get(component)
            if not isinstance(value, dict) or not any(
                    isinstance(item, torch.Tensor) for item in value.values()):
                errors.append({"checkpoint": checkpoint_label, "component": component,
                               "error": "required_component_missing_or_empty"})
    check_reference(reference, errors)
    check_candidate_meta(reference_meta, candidate_meta, expected_source, errors)
    if label == "e9_c" and candidate_meta.get("local_detail_output_constraint", "off") != "highpass":
        errors.append({"checkpoint": label, "error": "e9_c_requires_highpass_output_constraint",
                       "actual": candidate_meta.get("local_detail_output_constraint", "off")})
    check_branch_weights(candidate, candidate_meta, errors)

    # 已核实的旧格式差异：禁用的 AA-TCR 以前不保存，新格式保存空字典。
    if ("aa_tcr_fuser" not in reference and isinstance(candidate.get("aa_tcr_fuser"), dict)
            and not candidate["aa_tcr_fuser"] and not reference_meta.get("use_aa_tcr_fuse", 0)
            and not candidate_meta.get("use_aa_tcr_fuse", 0)):
        reference = dict(reference)
        reference["aa_tcr_fuser"] = {}
        compatible_differences.append("disabled_aa_tcr_missing_to_empty")

    rows = compare_checkpoints(reference, candidate, "E5_vs_" + label, candidate_has_text=False)
    for row in rows:
        is_branch = row["component"] in BRANCH_COMPONENTS and ADAPTER_MARKER in row["key"]
        # 必须覆盖默认白名单：原查询和视觉重采样器在 E9 中同样冻结。
        row["expected_trainable"] = is_branch and row["status"] == "unexpected_key"
        if is_branch:
            if row["expected_trainable"]:
                row.update(status="allowed_added_local_detail", unexpected=False)
            else:
                row["unexpected"] = True
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
        "label": label, "status": "passed" if passed else "failed", "check_completed": True,
        "frozen_passed": passed,
        "local_detail_source": candidate_meta.get("local_detail_source", "off"),
        "local_detail_layer": candidate_meta.get("local_detail_layer", ""),
        "local_detail_grid": candidate_meta.get("local_detail_grid"),
        "local_detail_output_constraint": candidate_meta.get("local_detail_output_constraint", "off"),
        "local_detail_highpass_kernel": candidate_meta.get("local_detail_highpass_kernel", 3),
        "train_global_step": candidate_meta.get("train_global_step"),
        "scope": "仅允许新增完整的 local_detail_adapter；所有 E5 已有参数逐值不变，"
                 "量化造成的数值变化也不放行。",
        "errors": errors, "compatible_format_differences": compatible_differences,
        "component_presence": {name: {"e5": name in original_reference, label: name in candidate}
                               for name in components},
        "allowed_local_detail_tensors": sum(row["expected_trainable"] for row in rows),
        "frozen_tensors_checked": sum(not row["expected_trainable"] and row["reference_dtype"] is not None
                                      for row in rows),
        "violation_count": len(violations), "violations": violations,
        "component_summary": summarize_differences(rows),
    }
    return report, rows


def audit_parity(metas):
    """A/B 只允许 local_detail_source 不同，否则收益无法归因到表示来源。"""
    labels = sorted(metas)
    report = {"compared": labels, "status": "skipped", "differences": [], "missing_fields": []}
    if len(labels) < 2:
        return report
    first, second = (metas[label] for label in labels)
    for field in PARITY_FIELDS:
        if field not in first or field not in second:
            report["missing_fields"].append(field)
            continue
        if first[field] != second[field]:
            report["differences"].append({"field": field, labels[0]: first[field], labels[1]: second[field]})
    sources = {label: metas[label].get("local_detail_source", "off") for label in labels}
    if len(set(sources.values())) != len(labels):
        report["differences"].append({"field": "local_detail_source", **sources})
    report["local_detail_source"] = sources
    report["status"] = "passed" if not report["differences"] else "failed"
    return report


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
    parser.add_argument("--e9-a-ckpt", help="A 组：局部旁路读取原 16 个纹理 token")
    parser.add_argument("--e9-b-ckpt", help="B 组：局部旁路读取压缩前的局部 token")
    parser.add_argument("--e9-c-ckpt", help="C 组：B 路局部 token，加高通残差输出约束")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    rows = []
    report = {"status": "failed", "check_completed": False, "frozen_passed": False,
              "execution_failed": True, "errors": [], "variants": {}, "parity": {}}
    candidates = {label: path for label, path in (("e9_a", args.e9_a_ckpt), ("e9_b", args.e9_b_ckpt),
                                                    ("e9_c", args.e9_c_ckpt))
                  if path}
    try:
        if not candidates:
            raise ValueError("至少需要 --e9-a-ckpt、--e9-b-ckpt 或 --e9-c-ckpt 之一")
        torch.set_num_threads(4)
        print("[E9] 加载 E5 与 %s，执行 CPU 只读冻结检查。" % "、".join(sorted(candidates)), flush=True)
        reference = load_checkpoint(args.e5_ckpt)
        metas = {}
        for label in sorted(candidates):
            candidate = load_checkpoint(candidates[label])
            metas[label] = candidate.get("meta", {})
            variant_report, variant_rows = audit_variant(
                reference, candidate, label, VARIANT_SOURCE[label.rsplit("_", 1)[-1]])
            report["variants"][label] = variant_report
            rows.extend(variant_rows)
            print("[E9] %s: %s，新增张量 %s，冻结张量 %s，违规 %s" % (
                label, variant_report["status"], variant_report["allowed_local_detail_tensors"],
                variant_report["frozen_tensors_checked"], variant_report["violation_count"]), flush=True)
        report["parity"] = audit_parity(metas)
        if report["parity"]["differences"]:
            print("[E9] A/B 配置不可比：%s" % json.dumps(
                report["parity"]["differences"], ensure_ascii=False), flush=True)
        report["check_completed"] = True
        report["execution_failed"] = False
        report["frozen_passed"] = (
            all(item["frozen_passed"] for item in report["variants"].values())
            and report["parity"]["status"] in ("passed", "skipped"))
        report["status"] = "passed" if report["frozen_passed"] else "failed"
    except Exception as exc:
        report["errors"].append({"error": type(exc).__name__, "message": str(exc)})
    report["checkpoints"] = {"e5": str(Path(args.e5_ckpt).resolve()),
                             **{label: str(Path(path).resolve()) for label, path in candidates.items()}}
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
        print("[E9] failed：检查结果无法写入 %s：%s" % (output, exc), file=sys.stderr, flush=True)
        return 1
    print("[E9] %s：%s" % (report["status"], output / "frozen_check.json"), flush=True)
    return 0 if report["frozen_passed"] and report["check_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
