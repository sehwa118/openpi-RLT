#!/usr/bin/env python3
"""Offline converter: Robbyant/lingbot-touch PyTorch checkpoint -> immutable Flax msgpack bundle."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.nn import functional as fn

from openpi.models.lingbot_touch_flax import MODALITIES
from openpi.models.lingbot_touch_flax import LingBotTouchConfig
from openpi.models.lingbot_touch_flax import LingBotTouchEncoder
from openpi.models.lingbot_touch_flax import convert_torch_state_dict_to_flax
from openpi.trlt.cache import write_json
from openpi.trlt.config import root_path


def _torch_functional_forward(state_dict, config: LingBotTouchConfig, signal, pos_xyz, coverage_mask, *, modality):
    """Run exact TargetEncoder math functionally on a loaded PyTorch state_dict without mutating files."""
    batch, frames, fingers, taxels, channels = signal.shape
    keep = coverage_mask.reshape(-1).nonzero().flatten()
    spatial = keep.numel()
    compact_signal = signal.reshape(batch, frames, fingers * taxels, channels).index_select(2, keep)
    compact_pos = pos_xyz.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch, fingers * taxels, 3).index_select(
        1, keep
    )
    finger = (
        torch.arange(fingers, dtype=torch.long)
        .view(fingers, 1)
        .expand(fingers, taxels)
        .reshape(-1)
        .index_select(0, keep)
    )
    times = frames // config.tubelet_size
    tubelets = (
        compact_signal.reshape(batch, times, config.tubelet_size, spatial, channels)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, times, spatial, config.tubelet_size * channels)
    )
    h = fn.linear(
        tubelets,
        state_dict[f"tokenizer.tokenizers.{modality}.signal_mlp.0.weight"],
        state_dict[f"tokenizer.tokenizers.{modality}.signal_mlp.0.bias"],
    )
    x = fn.linear(
        fn.gelu(h),
        state_dict[f"tokenizer.tokenizers.{modality}.signal_mlp.2.weight"],
        state_dict[f"tokenizer.tokenizers.{modality}.signal_mlp.2.bias"],
    ).reshape(batch, times * spatial, config.embed_dim)

    if config.use_shared_projector:
        sp = fn.layer_norm(
            x,
            (config.embed_dim,),
            state_dict["shared_projector.norm.weight"],
            state_dict["shared_projector.norm.bias"],
            1e-6,
        )
        sp = fn.gelu(
            fn.linear(sp, state_dict["shared_projector.fc1.weight"], state_dict["shared_projector.fc1.bias"])
        )
        x = x + fn.linear(sp, state_dict["shared_projector.fc2.weight"], state_dict["shared_projector.fc2.bias"])

    time_id = torch.arange(times, dtype=torch.long).repeat_interleave(spatial).unsqueeze(0).expand(batch, -1)
    finger_id = finger.repeat(times).unsqueeze(0).expand(batch, -1)
    pos_b = compact_pos[:, None, :, :].expand(batch, times, spatial, 3).reshape(batch, times * spatial, 3)

    head_dim = config.embed_dim // config.num_heads
    axis_dim = 2 * ((head_dim // 2) // 5)
    scale_fp32 = float(torch.tensor(config.spatial_coordinate_scale, dtype=torch.float32).item())
    positions = (time_id, finger_id, pos_b[..., 0], pos_b[..., 1], pos_b[..., 2])
    thetas = (config.time_theta, config.finger_theta, config.spatial_theta, config.spatial_theta, config.spatial_theta)
    scales = (1.0, 1.0, scale_fp32, scale_fp32, scale_fp32)

    def apply_rope(tensor):
        chunks = []
        start = 0
        for theta, pos, scale in zip(thetas, positions, scales, strict=True):
            end = start + axis_dim
            inv_freq = torch.pow(
                torch.tensor(theta, dtype=torch.float32),
                -torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim,
            )
            phase = pos.float()[:, None, :, None] * float(scale) * inv_freq[None, None, None, :]
            cos, sin = phase.cos(), phase.sin()
            vals = tensor[..., start:end].float()
            even, odd = vals[..., 0::2], vals[..., 1::2]
            chunks.append(torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2))
            start = end
        if start < head_dim:
            chunks.append(tensor[..., start:].float())
        return torch.cat(chunks, dim=-1).to(tensor.dtype)

    outputs = []
    out_set = set(config.output_layers)
    out_idx = 0
    dim = config.embed_dim
    for i in range(config.depth):
        p = f"transformer.blocks.{i}"
        n1 = fn.layer_norm(x, (dim,), state_dict[f"{p}.norm1.weight"], state_dict[f"{p}.norm1.bias"], 1e-6)
        qkv = fn.linear(n1, state_dict[f"{p}.attn.qkv.weight"], state_dict[f"{p}.attn.qkv.bias"])
        q, k, v = qkv.reshape(batch, times * spatial, 3, config.num_heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        attn = fn.scaled_dot_product_attention(apply_rope(q), apply_rope(k), v, dropout_p=0.0, is_causal=False)
        attn_out = fn.linear(
            attn.transpose(1, 2).reshape(batch, times * spatial, dim),
            state_dict[f"{p}.attn.proj.weight"],
            state_dict[f"{p}.attn.proj.bias"],
        )
        x = x + attn_out
        n2 = fn.layer_norm(x, (dim,), state_dict[f"{p}.norm2.weight"], state_dict[f"{p}.norm2.bias"], 1e-6)
        mlp = fn.gelu(fn.linear(n2, state_dict[f"{p}.mlp.0.weight"], state_dict[f"{p}.mlp.0.bias"]))
        x = x + fn.linear(mlp, state_dict[f"{p}.mlp.2.weight"], state_dict[f"{p}.mlp.2.bias"])
        if i in out_set:
            outputs.append(
                fn.layer_norm(
                    x,
                    (dim,),
                    state_dict[f"transformer.norms_block.{out_idx}.weight"],
                    state_dict[f"transformer.norms_block.{out_idx}.bias"],
                    1e-6,
                )
            )
            out_idx += 1
    return torch.cat(outputs, dim=-1)


def convert_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    lingbot_repo: Path | None = None,
    verify: bool = True,
    atol: float = 1e-4,
) -> dict:
    """Convert config.json + encoder.pt in bundle_dir into verified Flax msgpack weights."""
    config_path = bundle_dir / "config.json"
    weights_path = bundle_dir / "encoder.pt"
    raw_config = json.loads(config_path.read_text())
    config = LingBotTouchConfig.from_dict(raw_config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    flax_params = convert_torch_state_dict_to_flax(state_dict, config)

    max_diff = None
    if verify:
        rng = np.random.default_rng(0)
        signal_np = rng.uniform(0.0, 1.0, size=(1, config.tubelet_size * 2, 5, 6, 3)).astype(np.float32)
        pos_np = rng.normal(0.0, 0.5, size=(5, 6, 3)).astype(np.float32)
        cov_np = np.ones((5, 6), dtype=bool)
        cov_np[1, -2:] = False

        encoder = LingBotTouchEncoder(config)
        max_diff = 0.0
        torch_encoder = None
        if lingbot_repo is not None:
            sys.path.insert(0, str(lingbot_repo))
            from tactile_jepa_inference import load_encoder  # noqa: PLC0415

            torch_encoder = load_encoder(bundle_dir, device="cpu")

        for modality in MODALITIES:
            sig = signal_np.copy()
            if modality != "magnetic":
                sig[..., :2] = 0.0
            with torch.no_grad():
                if torch_encoder is not None:
                    ref = torch_encoder.encode(
                        torch.from_numpy(sig), torch.from_numpy(pos_np), torch.from_numpy(cov_np), modality=modality
                    ).latent.numpy()
                else:
                    ref = _torch_functional_forward(
                        state_dict,
                        config,
                        torch.from_numpy(sig),
                        torch.from_numpy(pos_np),
                        torch.from_numpy(cov_np),
                        modality=modality,
                    ).numpy()
            out = np.asarray(
                encoder.apply(
                    {"params": flax_params},
                    jnp.asarray(sig),
                    jnp.asarray(pos_np),
                    cov_np,
                    modality=modality,
                ).latent
            )
            diff = float(np.max(np.abs(out - ref)))
            max_diff = max(max_diff, diff)
            if not math.isfinite(diff) or diff > atol:
                raise AssertionError(f"PyTorch vs Flax parity failed for {modality}: max_abs_diff={diff} > {atol}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = serialization.to_bytes(flax_params)
    (output_dir / "params.msgpack").write_bytes(raw_bytes)
    write_json(output_dir / "config.json", config.to_dict())

    source_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    params_sha = hashlib.sha256(raw_bytes).hexdigest()
    param_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(flax_params))
    manifest = {
        "source_bundle": str(bundle_dir),
        "source_sha256": source_sha,
        "params_sha256": params_sha,
        "parameter_count": param_count,
        "verified": verify,
        "max_abs_diff": max_diff,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--bundle-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--lingbot-repo", type=Path)
    root.add_argument("--no-verify", action="store_true")
    root.add_argument("--atol", type=float, default=1e-4)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = convert_bundle(
        root_path(args.bundle_dir),
        root_path(args.output_dir),
        lingbot_repo=root_path(args.lingbot_repo) if args.lingbot_repo else None,
        verify=not args.no_verify,
        atol=args.atol,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
