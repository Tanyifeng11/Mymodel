import os
import torch


def load_checkpoint_file(path: str):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".safetensors":
        from safetensors import safe_open

        out = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    return torch.load(path, map_location="cpu")


def detect_gam_checkpoint_format(state_dict: dict) -> str:
    if isinstance(state_dict, dict) and state_dict.get("checkpoint_format") == "gam_texture_joint_v2":
        return "gam_texture_joint_v2"
    if isinstance(state_dict, dict) and state_dict.get("checkpoint_format") == "gam_texture_joint_v1":
        return "gam_texture_joint_v1"
    if all(k in state_dict for k in ["unet", "ref_unet", "texture_adapter"]):
        return "gam_texture_joint_v1"
    if "module" in state_dict:
        return "legacy_module"
    return "unknown"


def extract_texture_metadata(state_dict: dict):
    if not isinstance(state_dict, dict):
        return {}
    return state_dict.get("meta", {}) or state_dict.get("metadata", {}) or {}


def infer_texture_num_tokens(state_dict: dict, default: int = 16) -> int:
    meta = extract_texture_metadata(state_dict)
    if "texture_num_tokens" in meta:
        return int(meta["texture_num_tokens"])

    bf_sd = state_dict.get("bf_texture_conditioner", {}) if isinstance(state_dict, dict) else {}
    if isinstance(bf_sd, dict) and "resampler_queries" in bf_sd:
        return int(bf_sd["resampler_queries"].shape[1])
    return default


def infer_clip_embed_dim(state_dict: dict, fallback: int) -> int:
    meta = extract_texture_metadata(state_dict)
    if "clip_embeddings_dim" in meta:
        return int(meta["clip_embeddings_dim"])

    bf_sd = state_dict.get("bf_texture_conditioner", {}) if isinstance(state_dict, dict) else {}
    if isinstance(bf_sd, dict):
        if "token_source_proj.0.0.weight" in bf_sd:
            return int(bf_sd["token_source_proj.0.0.weight"].shape[0])
        if "token_source_proj.0.1.weight" in bf_sd:
            return int(bf_sd["token_source_proj.0.1.weight"].shape[1])
    return int(fallback)
