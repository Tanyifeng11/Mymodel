#!/usr/bin/env python3
import argparse
import csv
import os


METRICS = [
    ("tcf_lab_delta_mean", "lower"),
    ("tcf_hsv_l1_mean", "lower"),
    ("prompt_color_delta_e_mean", "lower"),
    ("tpf_patch_sim_mean", "higher"),
    ("leak_colored_frac_mean", "lower"),
    ("leak_edge_density_mean", "lower"),
    ("edge_f1_mean", "higher"),
    ("sketch_iou_mean", "higher"),
]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare E4b and E4d-lite conflict bucket metrics.")
    parser.add_argument("--baseline", required=True, help="E4b conflict_bucket_metrics.csv")
    parser.add_argument("--candidate", required=True, help="E4d-lite conflict_bucket_metrics.csv")
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    baseline = {
        (row.get("mode"), row.get("conflict_bucket")): row
        for row in read_rows(args.baseline)
    }
    candidate = {
        (row.get("mode"), row.get("conflict_bucket")): row
        for row in read_rows(args.candidate)
    }

    out_rows = []
    for key in sorted(set(baseline) & set(candidate)):
        mode, bucket = key
        base_row = baseline[key]
        cand_row = candidate[key]
        out = {
            "mode": mode,
            "conflict_bucket": bucket,
            "baseline_num_samples": base_row.get("num_samples"),
            "candidate_num_samples": cand_row.get("num_samples"),
        }
        for metric, direction in METRICS:
            base_value = to_float(base_row.get(metric))
            cand_value = to_float(cand_row.get(metric))
            out[f"baseline_{metric}"] = base_value
            out[f"candidate_{metric}"] = cand_value
            out[f"delta_{metric}"] = (
                None if base_value is None or cand_value is None else cand_value - base_value
            )
            if base_value is None or cand_value is None:
                out[f"{metric}_better"] = ""
            elif direction == "lower":
                out[f"{metric}_better"] = cand_value < base_value
            else:
                out[f"{metric}_better"] = cand_value > base_value
        out_rows.append(out)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        keys = sorted({key for row in out_rows for key in row})
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(out_rows)

    keys = ["mode", "conflict_bucket", "baseline_num_samples", "candidate_num_samples"]
    keys += [f"delta_{metric}" for metric, _ in METRICS]
    print(",".join(keys))
    for row in out_rows:
        print(",".join("" if row.get(key) is None else str(row.get(key)) for key in keys))


if __name__ == "__main__":
    main()
