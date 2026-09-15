"""逐像素检查 E12 OFF 与 E5；任何不一致都令服务器作业失败。"""

import argparse
import json
from pathlib import Path

from PIL import Image


def check_pixels(e5_dir, off_dir, expected_count):
    def load_rows(directory):
        rows = json.loads((Path(directory) / "metrics_per_sample.json").read_text(encoding="utf-8"))
        mapping = {str(row["sample_id"]): row for row in rows}
        if len(rows) != expected_count or len(mapping) != expected_count:
            raise ValueError(f"{directory} 样本数或唯一 ID 数与预期不一致")
        return mapping

    baseline, candidate = load_rows(e5_dir), load_rows(off_dir)
    if baseline.keys() != candidate.keys():
        raise ValueError("E5 与 OFF 样本 ID 不一致")
    mismatches = []
    for sample_id, row in baseline.items():
        # 使用原始生成图，不能比较缩放后的指标输入图。
        first = row.get("source_gen_path") or row["gen_path"]
        other = candidate[sample_id]
        second = other.get("source_gen_path") or other["gen_path"]
        with Image.open(first) as a, Image.open(second) as b:
            a, b = a.convert("RGB"), b.convert("RGB")
            if a.size != b.size or a.tobytes() != b.tobytes():
                mismatches.append(sample_id)
    if mismatches:
        raise ValueError(f"E12 OFF 与 E5 像素不一致，样本：{mismatches[:20]}")
    print(f"[E12 OFF] {expected_count} 张图全部与 E5 逐像素一致")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e5_dir", required=True)
    parser.add_argument("--off_dir", required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    args = parser.parse_args()
    check_pixels(args.e5_dir, args.off_dir, args.expected_count)


if __name__ == "__main__":
    main()
