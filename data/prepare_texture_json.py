import json
import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CAPTION_EXTS = {".txt"}

def index_dir(folder: Path, allowed_exts):
    index = {}
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in allowed_exts:
            index[p.stem] = p
    return index

def read_caption(txt_path: Path):
    return txt_path.read_text(encoding="utf-8").strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="e.g. /mnt/d/tyf/fuxian/datasets/MMDGarment")
    parser.add_argument("--output_json", type=str, required=True,
                        help="e.g. /mnt/d/tyf/fuxian/Mymodel/data/train_MMD_texture.json")
    parser.add_argument("--require_sketch", action="store_true",
                        help="skip samples if sketch is missing")
    args = parser.parse_args()

    root = Path(args.dataset_root)

    # 固定目录，不递归乱找
    caption_dir = root / "text"
    cloth_dir = root / "cloth"
    texture_dir = root / "texture"
    sketch_dir = root / "sketch"

    print(f"dataset_root = {root}")
    print(f"caption_dir  = {caption_dir}")
    print(f"cloth_dir    = {cloth_dir}")
    print(f"texture_dir  = {texture_dir}")
    print(f"sketch_dir   = {sketch_dir}")

    caption_index = index_dir(caption_dir, CAPTION_EXTS)
    cloth_index = index_dir(cloth_dir, IMAGE_EXTS)
    texture_index = index_dir(texture_dir, IMAGE_EXTS)
    sketch_index = index_dir(sketch_dir, IMAGE_EXTS) if sketch_dir.exists() else {}

    print(f"caption files: {len(caption_index)}")
    print(f"cloth files:   {len(cloth_index)}")
    print(f"texture files: {len(texture_index)}")
    print(f"sketch files:  {len(sketch_index)}")

    samples = []
    skipped = []

    cloth_stems = sorted(cloth_index.keys())

    for i, stem in enumerate(cloth_stems, 1):
        cloth_file = cloth_index.get(stem)
        texture_file = texture_index.get(stem)
        caption_file = caption_index.get(stem)
        sketch_file = sketch_index.get(stem)

        if caption_file is None:
            skipped.append((stem, "missing caption"))
            continue
        if texture_file is None:
            skipped.append((stem, "missing texture"))
            continue
        if args.require_sketch and sketch_file is None:
            skipped.append((stem, "missing sketch"))
            continue

        try:
            caption = read_caption(caption_file)
        except Exception as e:
            skipped.append((stem, f"bad caption: {e}"))
            continue

        item = {
            "caption": caption,
            "texture": f"texture/{texture_file.name}",
            "cloth": f"cloth/{cloth_file.name}",
        }

        if sketch_file is not None:
            item["sketch"] = f"sketch/{sketch_file.name}"

        samples.append(item)

        if i % 1000 == 0:
            print(f"processed {i}/{len(cloth_stems)} | valid {len(samples)} | skipped {len(skipped)}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== DONE ====")
    print(f"saved {len(samples)} samples to {out_path}")
    print(f"skipped {len(skipped)} samples")

    if skipped:
        print("\nfirst 20 skipped:")
        for stem, reason in skipped[:20]:
            print(f"  {stem}: {reason}")

if __name__ == "__main__":
    main()