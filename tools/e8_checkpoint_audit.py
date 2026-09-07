"""E8 的只读权重审计；不实例化扩散模型，不修改 checkpoint。"""

import math
from collections import defaultdict

import torch


COMPONENTS = (
    "unet", "ref_unet", "texture_adapter", "bf_texture_conditioner",
    "spatial_texture_encoder", "spatial_injection", "palette_token_mlp",
    "tcpm_lite", "aa_tcr_fuser",
)
CHUNK_SIZE = 262144
INFERENCE_META = set("""
pretrained_model_name_or_path pretrained_vae_model_path texture_num_tokens
texture_mode texture_condition_mode texture_preprocess_mode layer_group_enabled
use_palette_tokens num_palette_tokens palette_branch_scale_init use_texture_gate
gate_type gate_init gate_min gate_max use_balanced_fusion_gate
balanced_gate_hidden_dim balanced_gate_scale balanced_gate_min balanced_gate_max
use_conflict_aware_gate use_tcpm_lite tcpm_hidden_ratio tcpm_residual_scale_init
tcpm_mask_inner_only use_aa_tcr_fuse aa_tcr_num_heads aa_tcr_head_dim
aa_tcr_alpha_init aa_tcr_max_alpha aa_tcr_empty_fallback
conflict_texture_suppress_strength conflict_palette_suppress_strength
conflict_deltae_norm conflict_threshold image_encoder_path clip_hidden_layer
alpha bf_base_channels width height
""".split())


def load_checkpoint(path):
    # 与项目训练/推理加载器一致；只用于用户自己的训练产物。
    state = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint 顶层必须为 dict: %s" % path)
    return state


def _text_schema(checkpoint):
    state = checkpoint.get("bf_texture_conditioner", {})
    meta = checkpoint.get("meta", {})
    query = state.get("resampler_queries") if isinstance(state, dict) else None
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        return {}, False
    hidden = query.shape[-1]
    try:
        inner, heads = int(meta["text_guidance_dim"]), int(meta["text_guidance_heads"])
        ratio = float(meta["text_guidance_max_ratio"])
        valid = inner > 0 and heads > 0 and inner % heads == 0 and math.isfinite(ratio) and ratio > 0
    except (KeyError, TypeError, ValueError):
        inner, valid = 0, False
    schema = {"gate": (), "to_out.weight": (hidden, inner)}
    schema.update({name + ".weight": (inner, hidden) for name in ("to_q", "to_k", "to_v")})
    schema.update({name + "." + suffix: (hidden,)
                   for name in ("norm_query", "norm_text") for suffix in ("weight", "bias")})
    schema = {"text_guidance." + key: shape for key, shape in schema.items()}
    actual = {key for key in state if key.startswith("text_guidance.")}
    valid = valid and actual == set(schema) and all(
        isinstance(state.get(key), torch.Tensor) and state[key].is_floating_point()
        and tuple(state[key].shape) == shape
        for key, shape in schema.items())
    return schema, valid


def _chunks(tensor):
    for start in range(0, tensor.numel(), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, tensor.numel())
        if tensor.is_contiguous():
            yield tensor.view(-1)[start:end]
        else:
            # 非连续权重按逻辑位置取块，避免 reshape 复制整个张量。
            index = torch.arange(start, end)
            coordinates = []
            for size in reversed(tensor.shape):
                coordinates.append(index.remainder(size))
                index = index.div(size, rounding_mode="floor")
            yield tensor[tuple(reversed(coordinates))]


def _tensor_metrics(reference, candidate):
    same = torch.equal(reference, candidate)
    changed, max_delta, delta_sq, ref_sq = 0, 0.0, 0.0, 0.0
    finite_a, finite_b = True, True
    casts = [candidate.dtype, torch.float16, torch.bfloat16] if (
        reference.is_floating_point() and candidate.is_floating_point()
        and (not same or reference.dtype != candidate.dtype)) else []
    for x, y in zip(_chunks(reference), _chunks(candidate)):
        finite_a = bool(torch.isfinite(x).all()) and finite_a
        finite_b = bool(torch.isfinite(y).all()) and finite_b
        if same:
            continue
        changed += int(torch.ne(x, y).sum())
        xd, yd = x.double(), y.double()
        delta = yd - xd
        if not x.is_floating_point() and not y.is_floating_point():
            # 先做整数差，保留大 int64 的低位，再修正有符号减法溢出。
            xi, yi = x.long(), y.long()
            integer_delta = yi - xi
            delta = integer_delta.double()
            delta += ((yi >= 0) & (xi < 0) & (integer_delta < 0)).double() * float(2 ** 64)
            delta -= ((yi < 0) & (xi >= 0) & (integer_delta >= 0)).double() * float(2 ** 64)
        max_delta = max(max_delta, float(delta.abs().max()))
        delta_sq += float(delta.square().sum())
        ref_sq += float(xd.square().sum())
        casts = [dtype for dtype in casts if torch.equal(x.to(dtype).to(y.dtype), y)]
    finite = finite_a and finite_b
    return {
        "changed_numel": changed, "max_abs_delta": max_delta if finite else None,
        "rms_delta": math.sqrt(delta_sq / max(reference.numel(), 1)) if finite else None,
        "relative_l2": (math.sqrt(delta_sq / ref_sq) if ref_sq else (0.0 if not delta_sq else None)) if finite else None,
        "precision_cast_only": bool(casts) and finite,
        "reference_nonfinite": not finite_a, "candidate_nonfinite": not finite_b,
    }


def _row(comparison, component, key, reference=None, candidate=None):
    return dict(comparison=comparison, component=component, key=key, status="unchanged",
                expected_trainable=False, numel=(candidate.numel() if isinstance(candidate, torch.Tensor)
                    else reference.numel() if isinstance(reference, torch.Tensor) else 0),
                reference_dtype=str(reference.dtype) if isinstance(reference, torch.Tensor) else None,
                candidate_dtype=str(candidate.dtype) if isinstance(candidate, torch.Tensor) else None,
                changed_numel=0, max_abs_delta=None, rms_delta=None, relative_l2=None,
                precision_cast_only=False, reference_nonfinite=False, candidate_nonfinite=False,
                unexpected=False)


def compare_checkpoints(reference, candidate, comparison, candidate_has_text=False):
    rows = []
    discovered = {key for checkpoint in (reference, candidate) for key, value in checkpoint.items()
                  if isinstance(value, torch.Tensor) or (isinstance(value, dict)
                      and any(isinstance(item, torch.Tensor) for item in value.values()))}
    schema, valid_text = _text_schema(candidate)
    for component in sorted(set(COMPONENTS) | discovered):
        present_a, present_b = component in reference, component in candidate
        if not present_a and not present_b:
            continue  # 两边都未保存的可选组件不构成变动。
        first, second = reference.get(component), candidate.get(component)
        if present_a != present_b:
            row = _row(comparison, component, "<component>")
            row.update(status="missing_component" if present_a else "extra_component", unexpected=True)
            rows.append(row)
        if isinstance(first, torch.Tensor):
            first = {"<tensor>": first}
        if isinstance(second, torch.Tensor):
            second = {"<tensor>": second}
        if (present_a and not isinstance(first, dict)) or (present_b and not isinstance(second, dict)):
            row = _row(comparison, component, "<component>")
            row.update(status="invalid_component_type", unexpected=True)
            rows.append(row)
            continue
        first, second = first or {}, second or {}
        keys = set(first) | set(second)
        if component == "bf_texture_conditioner" and candidate_has_text:
            keys |= set(schema)
        for key in sorted(keys):
            a, b = first.get(key), second.get(key)
            row = _row(comparison, component, key, a, b)
            is_text = component == "bf_texture_conditioner" and key.startswith("text_guidance.")
            row["expected_trainable"] = component == "bf_texture_conditioner" and (
                key == "resampler_queries" or key.startswith("resampler.") or
                (is_text and candidate_has_text and valid_text and key in schema))
            if key not in second:
                row.update(status="missing_key", unexpected=True)
            elif key not in first:
                allowed = is_text and candidate_has_text and valid_text and key in schema and present_a
                row.update(status="allowed_added_text" if allowed else "unexpected_key", unexpected=not allowed)
            elif not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                row.update(status="non_tensor_entry", unexpected=True)
            elif a.shape != b.shape:
                row.update(status="shape_mismatch", unexpected=True,
                           reference_shape=list(a.shape), candidate_shape=list(b.shape))
            elif a.is_complex() or b.is_complex() or str(a.dtype) == "torch.uint64" or str(b.dtype) == "torch.uint64":
                row.update(status="unsupported_tensor_dtype", unexpected=True)
            elif a.is_floating_point() != b.is_floating_point():
                row.update(status="dtype_family_changed", unexpected=True)
            else:
                row.update(_tensor_metrics(a, b))
                if row["precision_cast_only"]:
                    row.update(status="precision_cast_only", unexpected=not row["expected_trainable"])
                elif row["changed_numel"]:
                    row.update(status="changed_trainable" if row["expected_trainable"] else "unexpected_change",
                               unexpected=not row["expected_trainable"])
                elif a.dtype != b.dtype:
                    row.update(status="dtype_changed_equal", unexpected=True)
            # 新增、缺失与 shape 不匹配的张量也检查非有限值。
            if row["max_abs_delta"] is None:
                for label, tensor in (("reference", a), ("candidate", b)):
                    if isinstance(tensor, torch.Tensor):
                        row[label + "_nonfinite"] = any(not bool(torch.isfinite(chunk).all())
                            for chunk in _chunks(tensor))
            if row["reference_nonfinite"] or row["candidate_nonfinite"]:
                row.update(status=row["status"] + ";nonfinite", unexpected=True)
            if is_text and (not candidate_has_text or not valid_text):
                row.update(status=row["status"] + ";invalid_text_schema", unexpected=True)
            rows.append(row)
    return rows


def summarize_differences(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["comparison"], row["component"])].append(row)
    result = []
    for (comparison, component), items in sorted(grouped.items()):
        values = [row["max_abs_delta"] for row in items if row["max_abs_delta"] is not None]
        result.append(dict(comparison=comparison, component=component, tensor_count=sum(
            row["reference_dtype"] is not None or row["candidate_dtype"] is not None for row in items),
            changed_tensors=sum(bool(row["changed_numel"]) for row in items),
            unexpected_tensors=sum(row["unexpected"] for row in items),
            precision_only_tensors=sum(row["precision_cast_only"] for row in items),
            nonfinite_tensors=sum(row["reference_nonfinite"] or row["candidate_nonfinite"] for row in items),
            changed_numel=sum(row["changed_numel"] for row in items), max_abs_delta=max(values) if values else None,
            status_counts={status: sum(row["status"] == status for row in items)
                           for status in sorted({row["status"] for row in items})}))
    return result


def metadata_comparison(reference, candidate):
    first, second = reference.get("meta", {}), candidate.get("meta", {})
    groups = {name: [] for name in ("inference", "training", "text_guidance", "other")}
    training_exact = {"resampler_training", "seed", "style_loss_type", "region_kernel_size",
                      "reload_texture_adapter_after_gam_init", "freeze_for_tcpm_lite", "train_detail_texture_gate"}
    for key in sorted(set(first) | set(second) | INFERENCE_META):
        a, b = first.get(key, "<missing>"), second.get(key, "<missing>")
        if key in first and key in second and a == b:
            continue
        group = "inference" if key in INFERENCE_META else (
            "training" if key in training_exact or key.endswith("_lr") or key.startswith(("lambda_", "ctd_", "joint_", "train_"))
            or key.endswith("_reg_weight") else "text_guidance" if key.startswith("text_guidance_") else "other")
        status = "changed" if key in first and key in second else (
            "missing_both" if key not in first and key not in second else
            "missing_reference" if key not in first else "missing_candidate")
        groups[group].append(dict(key=key, reference=a, candidate=b, status=status))
    groups["checkpoint_format"] = dict(reference=reference.get("checkpoint_format", "<missing>"),
                                       candidate=candidate.get("checkpoint_format", "<missing>"))
    return groups
