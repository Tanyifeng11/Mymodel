import argparse
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT = (
    "Describe only the garment in this image in one concise English sentence. "
    "Mention garment type, main color, sleeve length, neckline/collar, closure, "
    "and visible logos, prints, pockets, stripes, or texture. "
    "Do not describe the person, pose, background, image quality, or camera view. "
    "Use the style of a fashion dataset caption, for example: "
    "'a black t-shirt with a white logo on the front.'"
)


def clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^assistant\s*[:：]\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].lower() + text[1:]
    return text


def caption_one(model, processor, image_path: Path, device: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_caption(output_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/mnt/d/tyf/fuxian/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--cloth_dir",
        default="/mnt/d/tyf/fuxian/datasets/vitonhd/test/cloth",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/d/tyf/fuxian/datasets/vitonhd/test/text",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    cloth_dir = Path(args.cloth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in cloth_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA is not available, falling back to CPU.")
        args.device = "cpu"

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    done = 0
    skipped = 0
    for idx, image_path in enumerate(image_paths, start=1):
        out_path = output_dir / f"{image_path.stem}.txt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        caption = caption_one(
            model,
            processor,
            image_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        out_path.write_text(caption, encoding="utf-8")
        done += 1
        print(f"[{idx}/{len(image_paths)}] {image_path.name} -> {caption}")

    print(f"[done] generated={done} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
