"""Collect existing B0/B1 and new B2 geometry without retraining the controls."""

import argparse
import json
import shutil
from pathlib import Path

from tools.e15_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    controls, output = Path(args.controls), Path(args.output)
    result = {}
    for arm in ("b0", "b1", "b2"):
        report_path = output / ("probe_" + arm) / "report.json"
        if arm != "b2":
            report_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(controls / ("probe_" + arm) / "report.json", report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result[arm] = {
            "checkpoint": report["checkpoint"], "data": report["data"],
            "train": report["orientation_geometry_train"],
            "validation": report["orientation_geometry_validation"],
            "auxiliary_probes": report["scores"]}
        print("[e18.2-compare]", arm, result[arm]["validation"]["mean"], flush=True)
    if len({item["data"] for item in result.values()}) != 1:
        raise ValueError("Controls and B2 did not use the same dataset")
    write_json(output / "comparison.json", {
        "primary": ["held-out fused cosine gap", "held-out fused SupCon", "per-frequency/phase consistency"],
        "secondary": "passive final geometry with B2 downstream weights unchanged",
        "controls_reused_from": str(controls), "arms": result,
        "caveat": "B2 removes final/relation and preservation losses as well as freezing readout; this isolates a fused-only recipe, not one causal factor"})


if __name__ == "__main__":
    main()
