#!/usr/bin/env python3
"""Benchmark Stage B Attention Actor inference latency on GPU or CPU (and optional Stage A encode)."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import time

DATASET_PRESETS: dict[str, tuple[str, ...]] = {
    "3dtacdex_assembly": (
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_mlp.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_resmlp_d160.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_resmlp.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_hierarchical_d64.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_kinematic_d64.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_hierarchical.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_kinematic.json",
        "configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_full.json",
    ),
    "trex_squeeze": (
        "configs/trlt/attention_actor/probe_mlp.json",
        "configs/trlt/attention_actor/probe_resmlp.json",
        "configs/trlt/attention_actor/probe_part_attn_full.json",
        "configs/trlt/attention_actor/probe_part_attn_kinematic.json",
        "configs/trlt/attention_actor/probe_part_attn_hierarchical.json",
    ),
}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--dataset", choices=tuple(DATASET_PRESETS), default="3dtacdex_assembly")
    root.add_argument("--configs", nargs="+", type=Path, help="Explicit probe config JSON paths")
    root.add_argument("--device", default="0", help="Single GPU index (e.g. '0') or 'cpu'")
    root.add_argument("--batch-sizes", nargs="+", type=int, default=[1], help="Batch sizes to benchmark")
    root.add_argument("--warmup", type=int, default=50, help="Synchronized warmup iterations per measurement")
    root.add_argument("--iters", type=int, default=500, help="Timed iterations per measurement")
    root.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    root.add_argument("--seed", type=int, default=0)
    root.add_argument(
        "--include-stage-a",
        action="store_true",
        help="Also benchmark Stage A representation encode() from the first config and report E2E totals",
    )
    root.add_argument("--output", type=Path, help="Optional JSON report path")
    root.add_argument("--markdown-output", type=Path, help="Optional Markdown summary table path")
    root.add_argument("--dry-run", action="store_true", help="Print resolved benchmark plan without running JAX")
    return root


def _configure_jax_platform(device_arg: str) -> None:
    if not device_arg or "," in device_arg:
        raise ValueError(f"--device must be 'cpu' or a single GPU index (got {device_arg!r})")
    if device_arg == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = device_arg
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"


def _stats_ms(samples_ms: list[float], batch_size: int) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(samples_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    return {
        "iters": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_ms": float(np.min(arr)),
        "p50_ms": p50,
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "hz_p50": float(1000.0 / p50) if p50 > 0 else 0.0,
        "samples_per_sec_p50": float(batch_size * 1000.0 / p50) if p50 > 0 else 0.0,
    }


def _device_memory_stats(device) -> dict[str, int] | None:
    try:
        stats = device.memory_stats()
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    filtered = {k: int(stats[k]) for k in keys if k in stats}
    return filtered or None


def _benchmark_jit_callable(fn, host_args, device, *, batch_size: int, warmup: int, iters: int) -> dict:
    import jax
    import numpy as np

    device_args = jax.tree.map(lambda x: jax.device_put(x, device), host_args)
    jax.block_until_ready(device_args)

    t0 = time.perf_counter()
    compiled = fn.lower(*device_args).compile()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        jax.block_until_ready(compiled(*device_args))

    dev_samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        out = jax.block_until_ready(compiled(*device_args))
        dev_samples.append((time.perf_counter() - start) * 1000.0)

    out_np = np.asarray(out)
    if not np.all(np.isfinite(out_np)):
        raise FloatingPointError("Non-finite output during latency benchmark")

    host_warmup = min(warmup, 20)
    for _ in range(host_warmup):
        transferred = jax.tree.map(lambda x: jax.device_put(x, device), host_args)
        _ = np.asarray(compiled(*transferred))

    host_samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        transferred = jax.tree.map(lambda x: jax.device_put(x, device), host_args)
        _ = np.asarray(compiled(*transferred))
        host_samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "compile_ms": float(compile_ms),
        "output_shape": list(out_np.shape),
        "device_resident": _stats_ms(dev_samples, batch_size),
        "host_roundtrip": _stats_ms(host_samples, batch_size),
        "memory_stats": _device_memory_stats(device),
    }


def _stage_a_synthetic_batch(model_cfg, *, batch_size: int, np_dtype, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    nv = model_cfg.num_cameras * 256
    tactile_shape = (
        (
            batch_size,
            model_cfg.history_frames,
            model_cfg.num_sensors,
            model_cfg.num_taxels,
            1,
            model_cfg.tactile_channels,
        )
        if model_cfg.tactile_format == "taxels"
        else (
            batch_size,
            model_cfg.history_frames,
            model_cfg.num_sensors,
            model_cfg.tactile_resolution,
            model_cfg.tactile_resolution,
            model_cfg.tactile_channels,
        )
    )
    batch = {
        "visual_features": rng.normal(0.0, 0.2, size=(batch_size, nv, model_cfg.input_dim)).astype(np_dtype),
        "visual_valid_mask": np.ones((batch_size, nv), dtype=bool),
        "tactile": rng.uniform(-1.0, 1.0, size=tactile_shape).astype(np_dtype),
        "tactile_valid_mask": np.ones((batch_size, model_cfg.history_frames, model_cfg.num_sensors), dtype=bool),
        "tactile_timestamps": np.broadcast_to(
            np.arange(-model_cfg.history_frames + 1, 1, dtype=np.float32)[:, None] * 0.2,
            (batch_size, model_cfg.history_frames, model_cfg.num_sensors),
        ).copy(),
        "proprioception": rng.normal(0.0, 0.2, size=(batch_size, model_cfg.state_dim)).astype(np_dtype),
        "query_timestamp": np.full((batch_size,), 1.0, dtype=np.float32),
        "visual_timestamp": np.full((batch_size,), 0.8, dtype=np.float32),
    }
    if model_cfg.geometry_mode != "none":
        batch["tactile_positions"] = rng.normal(
            0.0, 0.05, size=(batch_size, model_cfg.history_frames, model_cfg.num_sensors, 3)
        ).astype(np.float32)
        batch["geometry_valid_mask"] = batch["tactile_valid_mask"].copy()
    return batch


def _format_markdown(report: dict) -> str:
    include_stage_a = bool(report.get("stage_a"))
    header = (
        f"# Attention Actor Latency Benchmark (`{report['dataset']}`)\n\n"
        f"- **Device:** `{report['device_kind']}` (`{report['platform']}`, precision=`{report['dtype']}`)\n"
        f"- **Warmup / Iters:** `{report['warmup']}` / `{report['iters']}`\n\n"
    )
    cols = [
        "Variant",
        "Params",
        "Batch",
        "Compile (ms)",
        "Dev p50 (ms)",
        "Dev p95 (ms)",
        "Dev p99 (ms)",
        "Dev Hz",
        "Host↔Dev p50 (ms)",
    ]
    align = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    if include_stage_a:
        cols.extend(["E2E Dev p50 (ms)", "Actor Share (%)"])
        align.extend(["---:", "---:"])
    lines = [header, "| " + " | ".join(cols) + " |", "| " + " | ".join(align) + " |"]

    if include_stage_a:
        for b_str, st in sorted(report["stage_a"]["by_batch"].items(), key=lambda item: int(item[0])):
            row = [
                f"`stage_a:{report['stage_a']['mode']}`",
                f"{report['stage_a']['parameter_count']:,}",
                b_str,
                f"{st['compile_ms']:.1f}",
                f"{st['device_resident']['p50_ms']:.3f}",
                f"{st['device_resident']['p95_ms']:.3f}",
                f"{st['device_resident']['p99_ms']:.3f}",
                f"{st['device_resident']['hz_p50']:.1f}",
                f"{st['host_roundtrip']['p50_ms']:.3f}",
                "—",
                "—",
            ]
            lines.append("| " + " | ".join(row) + " |")

    for entry in report["variants"]:
        for b_str, r in sorted(entry["by_batch"].items(), key=lambda item: int(item[0])):
            row = [
                f"`{entry['variant']}`",
                f"{entry['parameter_count']:,}",
                b_str,
                f"{r['compile_ms']:.1f}",
                f"{r['device_resident']['p50_ms']:.3f}",
                f"{r['device_resident']['p95_ms']:.3f}",
                f"{r['device_resident']['p99_ms']:.3f}",
                f"{r['device_resident']['hz_p50']:.1f}",
                f"{r['host_roundtrip']['p50_ms']:.3f}",
            ]
            if include_stage_a:
                row.extend([f"{r['e2e_device_p50_ms']:.3f}", f"{r['actor_share_device_percent']:.1f}%"])
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.warmup < 0 or args.iters <= 0 or any(b <= 0 for b in args.batch_sizes):
        raise ValueError("warmup must be >= 0 and iters/batch-sizes must be positive")
    _configure_jax_platform(args.device)

    from openpi.trlt.config import ModelConfig
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.config import probe_variant_name
    from openpi.trlt.config import root_path

    config_paths = [root_path(p) for p in (args.configs or DATASET_PRESETS[args.dataset])]
    for cfg_path in config_paths:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

    if args.dry_run:
        plan = {
            "status": "plan-only",
            "dataset": args.dataset,
            "device": args.device,
            "configs": [str(p) for p in config_paths],
            "batch_sizes": list(args.batch_sizes),
            "warmup": args.warmup,
            "iters": args.iters,
            "dtype": args.dtype,
            "include_stage_a": bool(args.include_stage_a),
        }
        print(json.dumps(plan, indent=2))
        return 0

    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi.models.trlt import build_model
    from openpi.models.trlt_actor import build_action_probe

    devices = jax.devices()
    device = devices[0]
    if args.device != "cpu" and device.platform != "gpu":
        raise RuntimeError(f"Expected GPU device for --device {args.device}, got {device.platform}")

    jnp_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    np_dtype = np.float32

    stage_a_report = None
    first_raw = json.loads(config_paths[0].read_text())
    if args.include_stage_a:
        model_cfg = ModelConfig(**first_raw.get("model", {}))
        stage_a_model = build_model(model_cfg)
        init_batch = _stage_a_synthetic_batch(model_cfg, batch_size=1, np_dtype=np_dtype, seed=args.seed)
        init_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x, dtype=jnp_dtype if x.dtype == np.float32 else x.dtype), device),
            init_batch,
        )
        # Keep timestamps float32 even in bfloat16 mode for numerical stability.
        for ts_key in ("tactile_timestamps", "query_timestamp", "visual_timestamp"):
            init_jax[ts_key] = jnp.asarray(init_batch[ts_key], dtype=jnp.float32)
        stage_a_vars = stage_a_model.init(
            jax.random.PRNGKey(args.seed), init_jax, method=stage_a_model.encode, train=False
        )
        stage_a_params = jax.tree.map(lambda x: x.astype(jnp_dtype), stage_a_vars["params"])
        jax.block_until_ready(stage_a_params)
        encode_fn = jax.jit(
            lambda b, _m=stage_a_model, _p=stage_a_params: _m.apply({"params": _p}, b, method=_m.encode, train=False)
        )
        by_batch = {}
        for batch_size in args.batch_sizes:
            host_batch = _stage_a_synthetic_batch(
                model_cfg, batch_size=batch_size, np_dtype=np_dtype, seed=args.seed + batch_size
            )
            by_batch[str(batch_size)] = _benchmark_jit_callable(
                encode_fn,
                (host_batch,),
                device,
                batch_size=batch_size,
                warmup=args.warmup,
                iters=args.iters,
            )
        stage_a_report = {
            "mode": model_cfg.mode,
            "parameter_count": int(sum(x.size for x in jax.tree.leaves(stage_a_params))),
            "by_batch": by_batch,
        }
        del stage_a_vars, stage_a_params, stage_a_model
        jax.clear_caches()

    variants_report = []
    for idx, cfg_path in enumerate(config_paths):
        raw = json.loads(cfg_path.read_text())
        model_cfg = ModelConfig(**raw.get("model", {}))
        probe_cfg = ProbeConfig(**raw.get("probe", {}))
        variant = probe_variant_name(probe_cfg)
        probe_model = build_action_probe(probe_cfg)
        rng_np = np.random.default_rng(args.seed + idx)

        z0 = jnp.zeros((1, 1, model_cfg.embed_dim), dtype=jnp_dtype)
        s0 = jnp.zeros((1, probe_cfg.action_dim), dtype=jnp_dtype)
        a0 = jnp.zeros((1,), dtype=jnp.float32)
        variables = probe_model.init(jax.random.PRNGKey(args.seed + idx), z0, s0, a0, train=False)
        params = jax.tree.map(lambda x: jax.device_put(x.astype(jnp_dtype), device), variables["params"])
        jax.block_until_ready(params)
        param_count = int(sum(x.size for x in jax.tree.leaves(params)))

        forward_fn = jax.jit(lambda z, s, a, _m=probe_model, _p=params: _m.apply({"params": _p}, z, s, a, train=False))
        by_batch = {}
        for batch_size in args.batch_sizes:
            z_host = rng_np.normal(0.0, 0.5, size=(batch_size, 1, model_cfg.embed_dim)).astype(np_dtype)
            s_host = rng_np.normal(0.0, 0.5, size=(batch_size, probe_cfg.action_dim)).astype(np_dtype)
            a_host = rng_np.uniform(0.0, 0.5, size=(batch_size,)).astype(np.float32)
            # In bfloat16 mode convert once inside JIT or host cast via jnp.asarray during device_put.
            if jnp_dtype != jnp.float32:
                host_args = (
                    jnp.asarray(z_host, dtype=jnp_dtype),
                    jnp.asarray(s_host, dtype=jnp_dtype),
                    jnp.asarray(a_host, dtype=jnp.float32),
                )
            else:
                host_args = (z_host, s_host, a_host)

            timing = _benchmark_jit_callable(
                forward_fn,
                host_args,
                device,
                batch_size=batch_size,
                warmup=args.warmup,
                iters=args.iters,
            )
            if stage_a_report is not None:
                sa_dev = stage_a_report["by_batch"][str(batch_size)]["device_resident"]["p50_ms"]
                sa_host = stage_a_report["by_batch"][str(batch_size)]["host_roundtrip"]["p50_ms"]
                act_dev = timing["device_resident"]["p50_ms"]
                act_host = timing["host_roundtrip"]["p50_ms"]
                e2e_dev = sa_dev + act_dev
                timing["e2e_device_p50_ms"] = float(e2e_dev)
                timing["e2e_host_p50_ms"] = float(sa_host + act_host)
                timing["actor_share_device_percent"] = float(100.0 * act_dev / e2e_dev) if e2e_dev > 0 else 0.0
            by_batch[str(batch_size)] = timing

        variants_report.append(
            {
                "variant": variant,
                "config": str(cfg_path),
                "architecture": probe_cfg.architecture,
                "attention_mask_mode": probe_cfg.attention_mask_mode,
                "hidden_dim": probe_cfg.hidden_dim,
                "depth": probe_cfg.depth,
                "action_dim": probe_cfg.action_dim,
                "horizon": probe_cfg.horizon,
                "parameter_count": param_count,
                "by_batch": by_batch,
            }
        )
        del variables, params, probe_model
        jax.clear_caches()

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "platform": device.platform,
        "device_kind": getattr(device, "device_kind", str(device)),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "batch_sizes": list(args.batch_sizes),
        "stage_a": stage_a_report,
        "variants": variants_report,
    }
    markdown = _format_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
