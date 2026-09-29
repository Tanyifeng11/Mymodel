"""在 HPC 网页终端用 ANSI 真彩色预览 E25 生成图，不复制图像文件。"""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    images = [Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BOX)
              for path in args.images]
    print(" | ".join(str(path) for path in args.images), flush=True)
    for y in range(0, 32, 2):
        blocks = []
        for image in images:
            pixels = image.load()
            line = []
            for x in range(32):
                top = ";".join(map(str, pixels[x, y]))
                bottom = ";".join(map(str, pixels[x, y + 1]))
                line.append(f"\x1b[38;2;{top}m\x1b[48;2;{bottom}m▀")
            blocks.append("".join(line) + "\x1b[0m")
        print("  ".join(blocks), flush=True)


if __name__ == "__main__":
    main()
