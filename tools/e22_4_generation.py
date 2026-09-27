"""E22.4：固定原 O4 生成协议，补测真实前驱；所有权重冻结。"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from models.localized_spatial import LocalizedAdapter
from tools import e22_o4_generation as gen
from tools.e15_common import write_json
from tools.e15_d5_generation import build_inference_args, load_inference_module
from tools.e19_handcrafted import bf_tokens, text_tokens
from tools.e20_utilization import case_stat, frozen_digest, load_pipeline
from tools.e22_4_genealogy import SOURCES, file_hash
from tools.e22_o4_metrics import axial_distance


def native_pipeline(root, name):
    checkpoint = str(root / SOURCES[name])
    args = argparse.Namespace(checkpoint=checkpoint, texture_ckpt=checkpoint,
                              base_model_path=str(root / "models/stable-diffusion-v1-5"),
                              vae_model_path=str(root / "models/stable-diffusion-v1-5/vae"),
                              clip_model=str(root / "models/clip"), device="cuda:0", seed=42, steps=50)
    ns = build_inference_args(args)
    ns.texture_num_tokens, ns.force_texture_num_tokens_override = 16, False
    pipe, _ = load_inference_module().prepare(ns)
    modules = {k: getattr(pipe, k) for k in ("unet", "reference_unet", "bf_texture_conditioner",
                                            "tcpm_lite", "vae", "text_encoder", "image_encoder")}
    for module in modules.values():
        module.eval().requires_grad_(False)
    return pipe, modules, ns.width, ns.height


def module_hashes(modules):
    return {k: frozen_digest(m) for k, m in modules.items()}


def paired(rows):
    index = {(r["reference"], r["sketch"], r["seed"], r["variant"]): r for r in rows}
    result = []
    for key, a in index.items():
        if key[-1] != "original":
            continue
        b = index[key[:3] + ("rot90",)]
        delta = axial_distance(a["direction"]["theta"], b["direction"]["theta"])
        valid = a["direction"]["valid"] and b["direction"]["valid"]
        result.append({"reference": key[0], "sketch": key[1], "seed": key[2],
                       "flip_correct": float(a["direction"]["correct"] and b["direction"]["correct"] and delta >= 70),
                       "delta_theta": delta, "valid_delta_theta": delta if valid else 0.})
    return result


def summarize(records, order, out):
    groups, pairs, contrasts = {}, {}, {}
    for name in order:
        rows = records[name]
        assert len(rows) == 64
        pairs[name] = paired(rows)
        groups[name] = {k: case_stat([(r["reference"], r[k]) for r in rows])
                        for k in ("sketch_iou", "edge_f1", "leakage")}
        groups[name]["direction_accuracy"] = case_stat([(r["reference"], float(r["direction"]["correct"])) for r in rows])
        groups[name].update({k: case_stat([(r["reference"], r[k]) for r in pairs[name]])
                             for k in ("flip_correct", "delta_theta", "valid_delta_theta")})
    first = None
    for parent, child in zip(order, order[1:]):
        a, b = records[child], records[parent]
        assert [(r["reference"], r["sketch"], r["seed"], r["variant"]) for r in a] == [
            (r["reference"], r["sketch"], r["seed"], r["variant"]) for r in b]
        result = {k: case_stat([(x["reference"], x[k] - y[k]) for x, y in zip(a, b)])
                  for k in ("sketch_iou", "edge_f1", "leakage")}
        result.update({k: case_stat([(x["reference"], x[k] - y[k]) for x, y in zip(pairs[child], pairs[parent])])
                       for k in ("flip_correct", "valid_delta_theta")})
        result["clear_deterioration"] = bool(result["leakage"]["ci95"][0] > .02 or
                                             result["sketch_iou"]["ci95"][1] < -.02 or
                                             result["edge_f1"]["ci95"][1] < -.02)
        edge = parent + " -> " + child
        if first is None and result["clear_deterioration"]:
            first = edge
        contrasts[edge] = result
    for seed in gen.SEEDS:
        assert len({r["initial_latent_sha256"] for rows in records.values() for r in rows if r["seed"] == seed}) == 1
    result = {"order": order, "groups": groups, "adjacent_contrasts": contrasts,
              "first_clear_deterioration": first, "training_steps": 0, "same_initial_noise": True,
              "deterioration_rule": "adjacent reference-cluster 95% CI entirely beyond +.02 leakage or -.02 IoU/EdgeF1",
              "scope": "existing checkpoint / composition transitions; not the first training step; controlled stripes only"}
    write_json(out / "report.json", result)
    write_json(out / "pairs.json", pairs)
    print("[root-cause]", json.dumps(result), flush=True)


def previews(records, cases, order, out):
    folder = out / "previews"
    folder.mkdir(exist_ok=True)
    for sk in (0, 1):
        for seed in gen.SEEDS:
            sheet = Image.new("RGB", ((2 + len(order)*2)*112, 8*168), "white")
            draw = ImageDraw.Draw(sheet)
            for ref in cases["references"]:
                row = ref["id"]
                sources = [(out/v["path"], v["variant"]) for v in ref["variants"]]
                for arm in order:
                    sources += [(out/r["path"], arm + " " + r["variant"]) for r in records[arm]
                                if r["reference"] == row and r["sketch"] == sk and r["seed"] == seed]
                for col, (path, label) in enumerate(sources):
                    image = Image.open(path).convert("RGB")
                    image.thumbnail((112, 145))
                    sheet.paste(image, (col*112, row*168))
                    draw.text((col*112+1, row*168+146), label, fill="black")
            sheet.save(folder / ("sketch%d_seed%d.jpg" % (sk, seed)), quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    old, out = root / "e22_o4_generation", root / "e22_4"
    genealogy = json.loads((out / "genealogy.json").read_text())
    # 元数据声明与逐张量证据同时要求成立，不能只按实验名称推断。
    manifest = genealogy["manifests"]["e17/direct_train_select/experiment_manifest.json"]["content"]
    assert Path(manifest["film_base_checkpoint"]).resolve() == (root/SOURCES["E5"]).resolve()
    b2 = genealogy["manifests"]["e18_2/b2/train_report.json"]["content"]
    assert Path(b2["start_checkpoint"]).resolve() == (root/SOURCES["E17_direct"]).resolve()
    for comparison in genealogy["comparisons"].values():
        assert all(not k.startswith(("unet.", "ref_unet.", "texture_adapter."))
                   for k in comparison["changed_tensors"]), "Unexpected generator ancestor: inspect before composing chain"
    assert all(genealogy["component_inheritance"].values())
    shutil.copytree(old / "inputs", out / "inputs", dirs_exist_ok=True)
    cases = json.loads((old / "cases.json").read_text())
    write_json(out / "cases.json", cases)
    old_protocol = json.loads((old / "protocol.json").read_text())
    protocol = {"source_protocol": old_protocol, "source_cases_sha256": file_hash(old/"cases.json"),
                "genealogy_sha256": file_hash(out/"genealogy.json"), "training_steps": 0,
                "checkpoint_chain": ["E5", "E17_direct", "E18_B2"],
                "assembly_chain": ["E18_B2", "S0_BF16_if_different", "S0", "O4"],
                "reuse": "E5/S0/O4 require pixel-exact same-hardware original+rot90 reproduction before reuse",
                "pattern_branch": "A3 geometry and mapper verified against E19.1/E19-A; A3 identity retrained from original frequency CNN, not a continuation of A2 identity"}
    write_json(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    records, audits = {}, {}

    def check_pipe(pipe, w, h):
        assert (w, h) == (old_protocol["width"], old_protocol["height"])
        assert dict(pipe.scheduler.config) == old_protocol["scheduler_config"]

    def reproduce(pipe, name, w, h, bank=None, injection=None):
        short = {**cases, "references": cases["references"][:1], "sketches": cases["sketches"][:1]}
        seeds = gen.SEEDS
        gen.SEEDS = (42,)
        try:
            new = gen.generate_arm(pipe, name + "_anchor", short, out, w, h, bank, injection,
                                   num_tokens=16 if name == "E5" else 24)
        finally:
            gen.SEEDS = seeds
        comparison = []
        for row in new:
            a, b = out/row["path"], old/name/Path(row["path"]).name
            comparison.append({"file": b.name, "pixel_exact": bool(np.array_equal(np.array(Image.open(a)), np.array(Image.open(b)))),
                               "old_png_sha256": file_hash(b), "new_png_sha256": file_hash(a)})
        audits[name] = {"anchors": comparison}
        write_json(out / "audit_partial.json", audits)
        assert all(r["pixel_exact"] for r in comparison), "Anchor mismatch: existing outputs must not be silently reused"
        rows = json.loads((old/name/"records.json").read_text())
        for row in rows:
            row["path"] = "../e22_o4_generation/" + row["path"]
        records[name] = rows

    native_bank = {}
    for name in ("E5", "E17_direct", "E18_B2"):
        pipe, modules, w, h = native_pipeline(root, name)
        check_pipe(pipe, w, h)
        before = module_hashes(modules)
        if name == "E5":
            reproduce(pipe, name, w, h)
        else:
            records[name] = gen.generate_arm(pipe, name, cases, out, w, h, num_tokens=16)
        if name == "E18_B2":
            with torch.inference_mode():
                text = text_tokens(pipe, ["a cloth"])
                neg = text_tokens(pipe, [" worst quality, low quality"])
                for ref in cases["references"]:
                    for v in ref["variants"]:
                        native_bank[ref["id"], v["variant"]] = tuple(t.cpu() for t in pipe.get_image_embeds(
                            pil_image=Image.open(out/v["path"]).convert("RGB"), width=w, height=h,
                            texture_mode="patch_resampled", text_embeds=text, negative_text_embeds=neg))
        assert module_hashes(modules) == before
        audits.setdefault(name, {}).update(frozen=True, loaded_module_hashes=before)
        write_json(out / "audit_partial.json", audits)
        del pipe, modules
        gc.collect()
        torch.cuda.empty_cache()

    pipe, bf, pattern, modules, w, h = load_pipeline(root, "cuda:0")
    check_pipe(pipe, w, h)
    before = module_hashes(modules)
    bank, bf_bank, differences = {}, {}, []
    with torch.inference_mode():
        text = text_tokens(pipe, ["a cloth"])
        null16, null24 = gen.current_null(pipe, bf, w, h, 0), gen.current_null(pipe, bf, w, h)
        for ref in cases["references"]:
            for v in ref["variants"]:
                key = ref["id"], v["variant"]
                image = Image.open(out/v["path"]).convert("RGB")
                bf_bank[key] = (pipe.tcpm_lite(bf_tokens(pipe, bf, image, text, w, h), text), null16)
                differences.append(max(float((x.cpu()-y).abs().max()) for x, y in zip(bf_bank[key], native_bank[key])))
                bank[key] = gen.current_tokens(pipe, bf, pattern, image, w, h), null24
    interface_equal = max(differences) == 0
    audits["S0_BF16_interface"] = {"exactly_equal_to_native_B2_tokens": interface_equal,
                                  "per_reference_max_abs_diff": differences}
    order = ["E5", "E17_direct", "E18_B2"]
    if not interface_equal:
        records["S0_BF16"] = gen.generate_arm(pipe, "S0_BF16", cases, out, w, h, bf_bank, num_tokens=16)
        order.append("S0_BF16")
    reproduce(pipe, "S0", w, h, bank)
    safe = torch.load(root/SOURCES["O4"], map_location="cpu", weights_only=False)
    adapter = LocalizedAdapter(channels=640, input_channels=3, policy=safe["policy"]).to(pipe.device)
    adapter.load_state_dict(safe["adapter"])
    adapter.eval().requires_grad_(False)
    adapter_hash = frozen_digest(adapter)
    injection = gen.ConditionalO4(pipe.unet, adapter, safe["site"])
    reproduce(pipe, "O4", w, h, bank, injection)
    injection.close()
    assert frozen_digest(adapter) == adapter_hash and module_hashes(modules) == before
    audits["S0_O4_frozen"] = {"modules": before, "adapter": adapter_hash, "unchanged": True}
    order += ["S0", "O4"]
    write_json(out / "freeze_and_reproduction_audit.json", audits)
    write_json(out / "records.json", records)
    summarize(records, order, out)
    previews(records, cases, order, out)


if __name__ == "__main__":
    main()
