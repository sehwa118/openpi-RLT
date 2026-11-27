#!/usr/bin/env python3
"""Batch-1 inference latency of one tactile MLP/GRU Stage B condition (docs/trlt/assembly_tactile_gru_plan.md §10).

Measures, in one process and on one device, with dropout off and synchronized timing:
  * touch_encoder: the trainable raw-token encoders of every touch cross-attention layer (MLP or GRU+proj)
    applied to one real [1, 4, 15, 3T] window, i.e. the per-query work G5 actually does (no streaming state);
  * full_actor: the complete Stage B actor (including the touch encoders) on the same real query.
Compilation is timed separately from warm iterations. Use one process per condition so memory statistics
are not mixed across conditions. Run with --device cpu for a CPU-only dry check.
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_attention_actor_latency import _benchmark_jit_callable
from scripts.benchmark_attention_actor_latency import _configure_jax_platform


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path, help="Probe checkpoint directory (…/best)")
    p.add_argument("--latents", required=True, type=Path, help="Frozen latent split providing the query row")
    p.add_argument("--touch", type=Path, help="Touch cache of the same split (touch conditions only)")
    p.add_argument("--row", type=int, default=0, help="Latent row used as the benchmark input")
    p.add_argument("--device", default="0", help="Single GPU index or 'cpu'")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _configure_jax_platform(args.device)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new output path")
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi.models.tactile_readout import _raw_gru_tokens
    from openpi.models.trlt_actor import build_action_probe
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.runner import PROBE_FIELDS
    from openpi.trlt.runner import LatentDataset
    from openpi.trlt.runner import apply_probe
    from openpi.trlt.runner import collate
    from openpi.trlt.runner import load_checkpoint
    from openpi.trlt.runner import probe_fields

    params, metadata = load_checkpoint(args.checkpoint)
    config = ProbeConfig(**metadata["train_config"])
    touch_arch = config.architecture == "touch_part_attention"
    if touch_arch != (args.touch is not None):
        raise ValueError("--touch is required exactly for touch_part_attention checkpoints")
    dataset = LatentDataset(args.latents)
    if touch_arch:
        from openpi.trlt.touch_tokens import TouchLatentDataset

        dataset = TouchLatentDataset(dataset, args.touch)
        if dataset.provenance["touch_identity"] != metadata["provenance"].get("touch_identity"):
            raise ValueError("touch_identity mismatch between cache and checkpoint")
    batch = collate([dataset[args.row]], probe_fields(config) if touch_arch else PROBE_FIELDS)
    batch = {k: np.asarray(v) for k, v in batch.items() if k not in ("action_target", "action_valid_mask")}
    model = build_action_probe(config)
    device = jax.devices()[0]
    params = jax.device_put(params, device)

    full = jax.jit(lambda b: apply_probe(model, config, params, b, train=False))
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_parameter_hash": metadata["parameter_hash"],
        "train_config": metadata["train_config"],
        "batch_size": 1,
        "dtype": "float32",
        "dropout": "off (train=False)",
        "warmup": args.warmup,
        "iters": args.iters,
        "device": str(device),
        "device_kind": getattr(device, "device_kind", None),
        "software": {"python": platform.python_version(), "jax": jax.__version__, "host": platform.node()},
        "env": {
            k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE")
        },
        "parameter_count": int(sum(x.size for x in jax.tree.leaves(params))),
    }
    if touch_arch:
        mode = config.touch_kv_mode
        layers = sorted(k for k in params if k.startswith("touch_ca_"))
        width = config.touch_raw_hidden_dim or config.hidden_dim

        class Encoder(nn.Module):  # same submodules/names as PerFingerTouchCrossAttention's raw encoders
            @nn.compact
            def __call__(self, touch):
                touch = jax.lax.stop_gradient(touch)
                if mode == "raw_gru":
                    return _raw_gru_tokens(touch, config.touch_gru_hidden_dim, config.hidden_dim)
                if mode == "raw_mlp":
                    hidden = nn.gelu(nn.Dense(width, name="touch_raw_fc1")(touch), approximate=False)
                    return nn.Dense(config.hidden_dim, name="touch_proj")(hidden)
                raise ValueError(f"encoder benchmark supports raw kv modes only, got {mode}")

        keys = ("touch_gru", "touch_proj") if mode == "raw_gru" else ("touch_raw_fc1", "touch_proj")
        enc_params = [{k: params[layer][k] for k in keys} for layer in layers]
        enc_model = Encoder()
        enc = jax.jit(lambda t: [enc_model.apply({"params": p}, t) for p in enc_params])
        report["touch_kv_mode"] = mode
        report["touch_encoder_layers"] = len(layers)
        report["touch_encoder_parameters"] = int(sum(x.size for x in jax.tree.leaves(enc_params)))
        # Equivalence: the standalone encoder reproduces the actor's encoder submodules exactly.
        report["touch_encoder"] = _benchmark_jit_callable(
            jax.jit(lambda t: jnp.stack(enc(t))), (batch["touch_tokens"],), device,
            batch_size=1, warmup=args.warmup, iters=args.iters,
        )  # fmt: skip
    report["full_actor"] = _benchmark_jit_callable(
        jax.jit(lambda b: full(b)), (batch,), device, batch_size=1, warmup=args.warmup, iters=args.iters
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("device", "parameter_count")}, indent=2))
    for part in ("touch_encoder", "full_actor"):
        if part in report:
            r = report[part]["device_resident"]
            print(f"{part}: compile={report[part]['compile_ms']:.1f}ms p50={r['p50_ms']:.4f}ms p95={r['p95_ms']:.4f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
