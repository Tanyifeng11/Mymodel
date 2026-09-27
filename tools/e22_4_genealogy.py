"""E22.4：从已有文件与张量哈希恢复组件继承关系；不训练、不修改 checkpoint。"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch


SOURCES = {
    "E5": "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt",
    "E17_direct": "e17/direct_train_select/checkpoint-final/joint_model.pt",
    "E18_B2": "e18_2/b2/joint_model.pt",
    "E19_mapper": "e19/a/pattern_branch.pt",
    "E19_frequency": "e19_1/fft_bias/encoder_42.pt",
    "E19_A2": "e19_2_a2/branches_42.pt",
    "E19_A3": "e19_2_a3/fft_rotation_42.pt",
    "E22_orientation": "e22/S2_orientation/adapter.pt",
    "O4": "e22_1/safe_orientation.pt",
}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_state(value, name="", tensors=None):
    if tensors is None:
        tensors = {}
    if torch.is_tensor(value):
        t = value.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        tensors[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                         "shape": list(t.shape), "dtype": str(t.dtype)}
        return {"tensor": name}
    if isinstance(value, dict):
        return {str(k): inspect_state(v, name + "." + str(k) if name else str(k), tensors)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [inspect_state(v, name + "." + str(i), tensors) for i, v in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def prefix_hash(tensors, prefix):
    selected = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest() if selected else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "e22_4"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    records = {}
    for name, relative in SOURCES.items():
        path = root / relative
        state = torch.load(path, map_location="cpu", weights_only=False)
        tensors = {}
        metadata = inspect_state(state, tensors=tensors)
        components = {key: prefix_hash(tensors, key + ".") for key in state if isinstance(state[key], dict)}
        records[name] = {"path": str(path), "resolved_path": str(path.resolve()),
                         "file_sha256": file_hash(path), "components": components,
                         "metadata": metadata, "tensors": tensors}
        print("[genealogy]", name, list(state), "file", records[name]["file_sha256"], flush=True)
        del state
        gc.collect()
        (out / "genealogy_partial.json").write_text(json.dumps(records, indent=2))
    manifests = {}
    for relative in ("e17/direct_train_select/experiment_manifest.json", "e18_2/b2/train_report.json",
                     "e19_2_a2/protocol.json", "e19_2_a3/protocol.json", "e22_1/report.json",
                     "e22_o4_generation/protocol.json", "e22_o4_generation/freeze_audit.json"):
        path = root / relative
        if path.exists():
            manifests[relative] = {"sha256": file_hash(path), "content": json.loads(path.read_text())}
    comparisons = {}
    for parent, child in (("E5", "E17_direct"), ("E17_direct", "E18_B2")):
        a, b = records[parent]["tensors"], records[child]["tensors"]
        changed = [k for k in a.keys() & b.keys() if a[k] != b[k]]
        comparisons[parent + " -> " + child] = {
            "changed_tensors": sorted(changed), "added_tensors": sorted(b.keys() - a.keys()),
            "removed_tensors": sorted(a.keys() - b.keys()),
            "unchanged_components": [k for k, v in records[parent]["components"].items()
                                     if v and records[child]["components"].get(k) == v]}
    def equal(a, ap, b, bp):
        return prefix_hash(records[a]["tensors"], ap) == prefix_hash(records[b]["tensors"], bp)
    inheritance = {
        "A3_geometry_equals_frequency": equal("E19_A3", "model.geometry.", "E19_frequency", "encoder."),
        "A3_mapper_equals_handcrafted": equal("E19_A3", "model.mapper.", "E19_mapper", "mapper."),
        "A3_geometry_equals_A2": equal("E19_A3", "model.geometry.", "E19_A2", "model.geometry."),
        "O4_adapter_equals_E22_orientation": equal("O4", "adapter.", "E22_orientation", "adapter."),
    }
    result = {"checkpoints": records, "manifests": manifests, "comparisons": comparisons,
              "component_inheritance": inheritance, "training_steps": 0}
    (out / "genealogy.json").write_text(json.dumps(result, indent=2))
    compact = {"files": {k: {p: v[p] for p in ("path", "resolved_path", "file_sha256", "components")}
                         for k, v in records.items()}, "comparisons": comparisons,
               "component_inheritance": inheritance}
    (out / "genealogy_summary.json").write_text(json.dumps(compact, indent=2))
    print("[inheritance]", inheritance, flush=True)
    for k, v in comparisons.items():
        print("[comparison]", k, "changed", len(v["changed_tensors"]),
              "unchanged", v["unchanged_components"], flush=True)


if __name__ == "__main__":
    main()
