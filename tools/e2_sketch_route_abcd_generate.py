"""在 E1-BFM-T3 epoch20 上，独立切换 BFM 和 GAM 接收的草图。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT = Path("/share/home/u2515283058/Mymodel")
SOURCE = PROJECT / "output_eval/e1_cfg_validate_20261123/source"
DEPENDENCIES = PROJECT / "output_eval/sd_bfm_fresh_20261113/dependencies"
SIZE = (384, 512)
PERIOD = 16
ROUTES = {
    "A": ("original", "original"),
    "B": ("original", "clean"),
    "C": ("clean", "original"),
    "D": ("clean", "clean"),
}
DIRECTIONS = ("horizontal", "vertical")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_text(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.tmp")
    image.save(tmp, format="PNG")
    tmp.replace(path)


def make_references(run):
    width, height = SIZE
    y, x = np.indices((height, width))
    paths = {}
    for direction, axis in zip(DIRECTIONS, (y, x)):
        bright = axis % PERIOD >= PERIOD // 2
        assert int(bright.sum()) * 2 == width * height
        # 两幅参考均由等量 32 灰和 224 灰组成，周期为 8 暗 + 8 亮。
        rgb = Image.fromarray(np.where(bright, 224, 32).astype("uint8")).convert("RGB")
        path = run / "references" / (direction + ".png")
        if path.exists():
            with Image.open(path) as old:
                if old.size != SIZE or old.convert("RGB").tobytes() != rgb.tobytes():
                    raise ValueError("参考像素与协议不符：" + str(path))
        else:
            save_image(path, rgb)
        paths[direction] = path
    return paths


def case_inputs(run, case):
    name = str(case["case_id"])
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("无效身份：" + name)
    base = run / "inputs" / name
    paths = {k: base / (k + ".png") for k in ("original", "clean", "mask")}
    if (base / "structure_mask.png").is_file():
        paths["structure_mask"] = base / "structure_mask.png"
    return paths


def render(runtime, pipe, network, torch, bfm, gam, reference, latent, prompt, seed):
    with torch.inference_mode(), runtime.BranchAudit(network, match_visual=False) as audit:
        try:
            network.adapter.condition(bfm, reference)
            result = pipe(
                ref_image=gam, prompt=prompt, texture_clip_image=None, texture_embeds=None,
                null_prompt="", negative_prompt=runtime.PROFILES[runtime.ARM]["negative_prompt"],
                width=SIZE[0], height=SIZE[1], num_images_per_prompt=1, guidance_scale=7.,
                sketch_scale=.6, ipa_scale=0., generator=torch.Generator(device="cuda").manual_seed(seed),
                latents=latent.clone(), num_inference_steps=50,
            )[0]
            return result.convert("RGB"), audit.inference_summary()
        finally:
            network.adapter.clear()


def generate(run):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path[:0] = [str(DEPENDENCIES), str(SOURCE), str(SOURCE / "tools")]
    import torch
    from tools import e1_cfg_validate as runtime

    if Path(runtime.__file__).resolve().parent.parent != SOURCE.resolve():
        raise RuntimeError("未加载冻结的第 20 轮推理源码")
    if sha(runtime.WEIGHT) != runtime.WEIGHT_SHA:
        raise ValueError("第 20 轮底座权重与归档不符")
    cases_path = run / "inputs/cases.json"
    cases = read(cases_path)
    ids = [str(case["case_id"]) for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("身份清单为空或包含重复项")
    references = make_references(run)
    inputs = {name: case_inputs(run, case) for name, case in zip(ids, cases)}
    inputs_sha = {name: {key: sha(path) for key, path in paths.items()} for name, paths in inputs.items()}
    ref_sha = {direction: sha(path) for direction, path in references.items()}
    protocol = dict(
        script_sha256=sha(Path(__file__)), source_sha256=sha(Path(runtime.__file__)),
        checkpoint_sha256=runtime.WEIGHT_SHA, cases_sha256=sha(cases_path),
        inputs_sha256=inputs_sha, reference_sha256=ref_sha, routes=ROUTES,
        period=PERIOD, size=SIZE, colors=[32, 224], cfg=7., sketch_scale=.6,
        steps=50, training_updates=0, torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(0),
    )
    protocol_text = json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    protocol_file = run / "generation_protocol.json"
    if protocol_file.exists():
        if read(protocol_file) != json.loads(protocol_text):
            raise ValueError("断点的输入、权重或配置已改变，请使用新的运行目录")
    else:
        atomic_text(protocol_file, protocol_text)
    protocol_sha = sha(protocol_file)
    pipe, network = runtime.shared.load_network("cuda", runtime.WEIGHT)
    network.eval().requires_grad_(False)
    pipe.unet.half()
    pipe.vae.half()
    network.adapter.half()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tensors = {key: runtime.core.tensor(path).cuda().half() for key, path in references.items()}
    records = []
    checked_controls = []
    for case, name in zip(cases, ids):
        paths = inputs[name]
        sketch = {key: runtime.core.tensor(paths[key]).cuda().half() for key in ("original", "clean")}
        identical = torch.equal(sketch["original"], sketch["clean"])
        kind = str(case["sample_type"])
        if ("control" in kind.lower() or "对照" in kind) and not identical:
            raise ValueError("对照身份两张草图在模型输入处不同：" + name)
        seed = int(case.get("seed", 42))
        latent = torch.randn((1, 4, 64, 48), device="cuda", dtype=torch.float16,
                             generator=torch.Generator(device="cuda").manual_seed(seed))
        latent_sha = runtime.digest(latent)
        case_records = []
        for direction in DIRECTIONS:
            output_pixels = []
            for arm, (bfm_key, gam_key) in ROUTES.items():
                image_file = run / "outputs" / name / (direction + "_" + arm + ".png")
                audit_path = image_file.with_suffix(".json")
                record = dict(
                    case_id=name, seed=seed, arm=arm, sample_type=kind, prompt=case["prompt"],
                    texture_direction=direction, period=PERIOD, image=image_file.relative_to(run).as_posix(),
                    mask=paths["mask"].relative_to(run).as_posix(), sketch=paths["clean"].relative_to(run).as_posix(),
                    bfm_sketch=paths[bfm_key].relative_to(run).as_posix(),
                    gam_sketch=paths[gam_key].relative_to(run).as_posix(),
                    reference=references[direction].relative_to(run).as_posix(),
                    reference_sha256=ref_sha[direction], initial_latent_sha256=latent_sha,
                    bfm_sketch_sha256=inputs_sha[name][bfm_key], gam_sketch_sha256=inputs_sha[name][gam_key],
                    generation_protocol_sha256=protocol_sha, audit=audit_path.relative_to(run).as_posix(),
                )
                if "structure_mask" in paths:
                    record["structure_mask"] = paths["structure_mask"].relative_to(run).as_posix()
                if image_file.exists() and audit_path.exists():
                    stored = read(audit_path)
                    if (stored["record"] != record or stored["image_sha256"] != sha(image_file)
                            or not stored["branch_audit"]["all_steps_verified"]):
                        raise ValueError("断点核验失败：" + str(audit_path))
                else:
                    image, branch = render(runtime, pipe, network, torch, sketch[bfm_key], sketch[gam_key],
                                           tensors[direction], latent, case["prompt"], seed)
                    save_image(image_file, image)
                    stored = dict(record=record, branch_audit=branch, image_sha256=sha(image_file))
                    atomic_text(audit_path, json.dumps(stored, ensure_ascii=False, indent=2) + "\n")
                case_records.append(dict(record, image_sha256=stored["image_sha256"]))
                if identical:
                    with Image.open(image_file) as output:
                        output_pixels.append(hashlib.sha256(output.convert("RGB").tobytes()).hexdigest())
            if identical and len(set(output_pixels)) != 1:
                raise RuntimeError("相同草图的四臂实际结果不一致：" + name + "/" + direction)
        if identical:
            checked_controls.append(name)
        records.extend(case_records)
        atomic_text(run / "manifest.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        print("完成 {}/{}，{}，累计 {} 张".format(len(records) // 8, len(cases), name, len(records)), flush=True)
    atomic_text(run / "generation_complete.json", json.dumps(dict(
        cases=len(cases), outputs=len(records), training_updates=0, checked_identical_cases=checked_controls,
        generation_protocol_sha256=protocol_sha, manifest_sha256=sha(run / "manifest.jsonl"),
    ), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    generate(args.run.resolve())
