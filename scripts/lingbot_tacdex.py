#!/usr/bin/env python3
"""Test frozen LingBot-Touch on 3DTacDex PaXini tactile data (CPU-friendly; no training).

Subcommands (explicit; none launches another):
  inspect  Adapter-only check: geometry, clipping statistics, token counts. No model.
  extract  Encode causal tactile windows with a converted Flax bundle (or a random-init control)
           and save per-finger mean-pooled features under datasets/3dtacdex/lingbot_features/.
  probe    Ridge-regression diagnostics on saved features vs raw-tactile baselines, grouped by
           original trajectory, written under experiments/3dtacdex_analysis/lingbot/.

Example (CPU only):
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/lingbot_tacdex.py extract \
      --zarr datasets/3dtacdex/openbox/openbox.zarr.zip \
      --bundle checkpoints/lingbot_touch/tiny_flax_<rev> --tag tiny_tip_t2
"""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from openpi.trlt.cache import write_json
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_lingbot import TACDEX_FINGERS
from openpi.trlt.tacdex_lingbot import PaxiniLingBotConfig
from openpi.trlt.tacdex_lingbot import canonical_geometry
from openpi.trlt.tacdex_lingbot import causal_windows
from openpi.trlt.tacdex_lingbot import grouped_split
from openpi.trlt.tacdex_lingbot import load_oxt_task
from openpi.trlt.tacdex_lingbot import prepare_signal

FEATURE_ROOT = Path("datasets/3dtacdex/lingbot_features")
REPORT_ROOT = Path("experiments/3dtacdex_analysis/lingbot")


def _adapter_config(args) -> PaxiniLingBotConfig:
    return PaxiniLingBotConfig(
        pads=args.pads,
        frames=args.frames,
        scales_xyz=tuple(args.scales_xyz),
        axis_signs=tuple(args.axis_signs),
        radius_mm=args.radius_mm,
        origin_mm=tuple(args.origin_mm),
        pulp_gap_mm=args.pulp_gap_mm,
    )


def _subset_segments(task, max_segments: int | None):
    if not max_segments or max_segments >= len(task.episode_ends):
        return task
    end = int(task.episode_ends[max_segments - 1])
    for name in ("tactile", "hand_joints", "arm_joints", "timestamps", "trajectory_id"):
        setattr(task, name, getattr(task, name)[:end])
    if task.hand_command is not None:
        task.hand_command = task.hand_command[:end]
        task.hand_command_present = task.hand_command_present[:end]
    task.episode_ends = task.episode_ends[:max_segments]
    return task


def cmd_inspect(args):
    config = _adapter_config(args)
    task = _subset_segments(load_oxt_task(root_path(args.zarr), args.group), args.max_segments)
    pos, coverage = canonical_geometry(config)
    signal, stats = prepare_signal(task.tactile, config)
    anchors, _, padded, _ = causal_windows(task.episode_ends, config.frames)
    tubelet = 2  # Tiny; Small uses 4. Checked again at extraction against the bundle config.
    per_pad_active = {
        f"{finger}_{pad}": float(np.mean(np.abs(task.tactile[:, 2 * i + j]).sum(axis=(1, 2)) > 0))
        for i, finger in enumerate(TACDEX_FINGERS)
        for j, pad in enumerate(("tip", "pulp"))
    }
    report = {
        "zarr": str(task.path),
        "group": task.group,
        "frames": task.frames,
        "segments": int(len(task.episode_ends)),
        "trajectories": int(len(set(task.trajectory_id.tolist()))),
        "adapter": config.to_dict(),
        "signal_shape_per_frame": list(signal.shape[1:]),
        "coverage_valid_per_finger": coverage.sum(axis=1).tolist(),
        "tokens_per_window_tiny": (config.frames // tubelet) * int(coverage.sum())
        if config.frames % tubelet == 0
        else None,
        "anchors": int(anchors.size),
        "anchors_with_replicated_history": int(np.sum(padded > 0)),
        "signal_stats_valid_slots": stats.to_dict(),
        "frac_frames_pad_nonzero": per_pad_active,
        "pos_xyz_valid_min": pos[coverage].min(axis=0).round(4).tolist(),
        "pos_xyz_valid_max": pos[coverage].max(axis=0).round(4).tolist(),
        "normalized_valid_percentiles(50,90,99,99.9,100)": {
            axis: np.percentile(np.abs(signal[:, :4, :, k]), [50, 90, 99, 99.9, 100]).round(4).tolist()
            for k, axis in enumerate("xyz")
        },
        "has_hand_command": task.hand_command is not None,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(root_path(args.output), report)


def _load_model(args):
    import jax

    from openpi.models.lingbot_touch_flax import LingBotTouchConfig
    from openpi.models.lingbot_touch_flax import init_random_params
    from openpi.models.lingbot_touch_flax import load_flax_bundle

    if (args.bundle is None) == (args.random_init_seed is None):
        raise SystemExit("Pass exactly one of --bundle or --random-init-seed")
    if args.bundle is not None:
        config, params, manifest = load_flax_bundle(root_path(args.bundle))
        source = {"kind": "pretrained", "bundle": str(root_path(args.bundle)), "manifest": manifest}
    else:
        config = LingBotTouchConfig.tiny()
        params = init_random_params(config, seed=args.random_init_seed)
        source = {"kind": "random_init_control", "seed": args.random_init_seed, "config": config.to_dict()}
    source["devices"] = [str(d) for d in jax.devices()]
    return config, params, source


def cmd_extract(args):
    import jax
    import jax.numpy as jnp

    from openpi.models.lingbot_touch_flax import encode_single_hand

    if any(d.platform != "cpu" for d in jax.devices()) and not args.allow_accelerator:
        raise SystemExit("Non-CPU JAX device visible; set JAX_PLATFORMS=cpu or pass --allow-accelerator")
    adapter = _adapter_config(args)
    config, params, source = _load_model(args)
    if adapter.frames % config.tubelet_size:
        raise SystemExit(f"--frames must be divisible by tubelet_size={config.tubelet_size}")
    task = _subset_segments(load_oxt_task(root_path(args.zarr), args.group), args.max_segments)
    pos, coverage = canonical_geometry(adapter)
    signal, stats = prepare_signal(task.tactile, adapter)
    anchors, windows, padded, segments = causal_windows(task.episode_ends, adapter.frames, stride=args.stride)

    out_dir = root_path(args.output_dir or FEATURE_ROOT / Path(task.group).stem / args.tag)
    if (out_dir / "features.npz").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir}/features.npz exists; pass --overwrite or choose another --tag")
    m = adapter.taxels_per_finger
    times = adapter.frames // config.tubelet_size
    c = config.target_dim
    pos_j = jnp.asarray(pos)

    def encode(batch_signal):
        latent, time_id, finger_id, _ = encode_single_hand(
            params, config, jnp.asarray(batch_signal), pos_j, coverage, modality=adapter.modality
        )
        return np.asarray(latent), np.asarray(time_id), np.asarray(finger_id)

    zero_latent, time_id, finger_id = encode(np.zeros((1, adapter.frames, 5, m, 3), np.float32))
    expected_tokens = times * int(coverage.sum())
    if zero_latent.shape != (1, expected_tokens, c):
        raise AssertionError(f"Unexpected latent shape {zero_latent.shape}")
    # Tokens are time-major, then finger-major compact spatial slots (pinky slot has no tokens).
    expected_finger = np.tile(np.repeat(np.arange(4), m), times)
    if not np.array_equal(finger_id, expected_finger):
        raise AssertionError("Token finger ordering differs from the expected time/finger/taxel layout")

    def pool(latent):
        return latent.reshape(latent.shape[0], times, 4, m, c).mean(axis=(1, 3))

    finger_mean = np.zeros((anchors.size, 4, c), np.float32)
    last_tokens = np.zeros((anchors.size, 4, m, c), np.float16) if args.save_last_tokens else None
    start_time = time.perf_counter()
    for begin in range(0, anchors.size, args.batch_size):
        idx = slice(begin, min(begin + args.batch_size, anchors.size))
        latent, _, _ = encode(signal[windows[idx]])
        if not np.isfinite(latent).all():
            raise FloatingPointError(f"Nonfinite LingBot output in batch starting at {begin}")
        finger_mean[idx] = pool(latent)
        if last_tokens is not None:
            last_tokens[idx] = latent.reshape(-1, times, 4, m, c)[:, -1].astype(np.float16)
        if args.progress and (begin // args.batch_size) % 10 == 0:
            print(f"encoded {idx.stop}/{anchors.size} anchors", flush=True)
    elapsed = time.perf_counter() - start_time

    arrays = {
        "anchor": anchors,
        "window": windows,
        "replicated_history": padded,
        "segment": segments,
        "finger_mean": finger_mean,
        "zero_input_finger_mean": pool(zero_latent)[0],
    }
    if last_tokens is not None:
        arrays["last_tubelet_tokens"] = last_tokens
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "features.npz", **arrays)
    meta = {
        "created_unix": time.time(),
        "zarr": task.path,
        "group": task.group,
        "max_segments": args.max_segments,
        "frames_total": task.frames,
        "anchors": int(anchors.size),
        "stride": args.stride,
        "adapter": adapter.to_dict(),
        "adapter_identity": adapter.identity,
        "model": source,
        "encoder_config": config.to_dict(),
        "tokens_per_window": expected_tokens,
        "feature": "finger_mean [N, 4 fingers (thumb,index,middle,ring), C] = mean over tubelets and taxels",
        "signal_stats_valid_slots": stats.to_dict(),
        "encode_seconds": elapsed,
        "anchors_per_second": anchors.size / max(elapsed, 1e-9),
        "python": platform.python_version(),
    }
    write_json(out_dir / "meta.json", meta)
    print(
        json.dumps({k: meta[k] for k in ("anchors", "tokens_per_window", "encode_seconds")} | {"output": str(out_dir)})
    )


# ---------------------------------------------------------------------------------------------
# Ridge diagnostics


def _ridge_fit_predict(x_tr, y_tr, x_ev_list, alphas):
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    ym = y_tr.mean(0)
    u, s, vt = np.linalg.svd((x_tr - mu) / sd, full_matrices=False)
    uty = u.T @ (y_tr - ym)
    preds = []
    for alpha in alphas:
        w = vt.T @ ((s / (s**2 + alpha))[:, None] * uty)
        preds.append([((x - mu) / sd) @ w + ym for x in x_ev_list])
    return preds


def _r2(y, p):
    var = y.var(0)
    keep = var > 1e-12
    if not keep.any():
        return float("nan"), float("nan")
    per_dim = 1.0 - ((y - p) ** 2).mean(0)[keep] / var[keep]
    pooled = 1.0 - ((y - p) ** 2).mean(0)[keep].sum() / var[keep].sum()
    return float(per_dim.mean()), float(pooled)


def _tip_normal_force(task, anchors):
    """Per-finger summed raw normal (axis 2) reading of the four tip pads, [N, 4]."""
    return task.tactile[anchors][:, [0, 2, 4, 6]][..., 2].sum(axis=2)


def _targets(task, anchors, segments):
    hj = task.hand_joints
    same_next = np.zeros(anchors.size, bool)
    nxt = anchors + 1
    inside = nxt < task.frames
    same_next[inside] = task.segment_ids()[nxt[inside]] == segments[inside]
    targets = {
        "hand_joint_delta_next": (hj[np.minimum(nxt, task.frames - 1)] - hj[anchors], same_next),
        "hand_joint_state": (hj[anchors], np.ones(anchors.size, bool)),
        "tip_normal_force_sum": (_tip_normal_force(task, anchors), np.ones(anchors.size, bool)),
    }
    if task.hand_command is not None:
        targets["hand_command_minus_state"] = (
            task.hand_command[anchors] - hj[anchors],
            task.hand_command_present[anchors] & np.isfinite(task.hand_command[anchors]).all(1),
        )
    return targets


def cmd_probe(args):
    feature_dirs = [root_path(p) for p in args.features]
    metas = [json.loads((d / "meta.json").read_text()) for d in feature_dirs]
    feats = [np.load(d / "features.npz") for d in feature_dirs]
    base = feats[0]
    for f, m in zip(feats[1:], metas[1:], strict=True):
        if not np.array_equal(f["anchor"], base["anchor"]) or m["zarr"] != metas[0]["zarr"]:
            raise SystemExit("All --features must come from the same archive and anchors")
    adapter = PaxiniLingBotConfig.from_dict(metas[0]["adapter"])
    task = _subset_segments(load_oxt_task(metas[0]["zarr"], metas[0]["group"]), metas[0]["max_segments"])
    anchors, segments, windows = base["anchor"], base["segment"], base["window"]
    signal, _ = prepare_signal(task.tactile, adapter)
    raw_window = signal[windows][:, :, :4].reshape(anchors.size, -1)
    raw_last = signal[anchors][:, :4].reshape(anchors.size, -1)

    feature_sets = {"raw_last_frame": raw_last, "raw_window": raw_window}
    for d, f, m in zip(feature_dirs, feats, metas, strict=True):
        label = d.name if m["model"]["kind"] == "pretrained" else f"{d.name}(random)"
        feature_sets[f"lingbot:{label}"] = f["finger_mean"].reshape(anchors.size, -1).astype(np.float64)

    split = grouped_split(task.trajectory_id[anchors], seed=args.split_seed)
    alphas = np.logspace(-2, 5, 15)
    results = {}
    for tname, (y_all, valid) in _targets(task, anchors, segments).items():
        tr, va, te = (split[k] & valid for k in ("train", "validation", "test"))
        if min(tr.sum(), va.sum(), te.sum()) < 10:
            continue
        results[tname] = {"n": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())}}
        for fname, x in feature_sets.items():
            preds = _ridge_fit_predict(x[tr], y_all[tr], [x[va], x[te]], alphas)
            val_scores = [_r2(y_all[va], p[0])[1] for p in preds]
            best = int(np.nanargmax(val_scores))
            mean_r2, pooled_r2 = _r2(y_all[te], preds[best][1])
            results[tname][fname] = {
                "alpha": float(alphas[best]),
                "val_pooled_r2": float(val_scores[best]),
                "test_mean_r2": mean_r2,
                "test_pooled_r2": pooled_r2,
            }

    # Contact sensitivity: distance of each embedding from the all-zero-input embedding, relative to the
    # median feature norm (cosine is undefined for a random-init control whose zero-input output is 0).
    sensitivity = {}
    force = _tip_normal_force(task, anchors).sum(axis=1)
    contact = force > 0
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)  # noqa: E731
    for d, f in zip(feature_dirs, feats, strict=True):
        a = f["finger_mean"].reshape(anchors.size, -1).astype(np.float64)
        z = f["zero_input_finger_mean"].reshape(1, -1).astype(np.float64)
        dist = np.linalg.norm(a - z, axis=1) / max(float(np.median(np.linalg.norm(a, axis=1))), 1e-12)
        sensitivity[d.name] = {
            "spearman_reldist_vs_tip_force": float(np.corrcoef(rank(dist), rank(force))[0, 1]),
            "mean_reldist_no_tip_contact": float(dist[~contact].mean()) if (~contact).any() else None,
            "mean_reldist_tip_contact": float(dist[contact].mean()) if contact.any() else None,
            "zero_input_embedding_norm": float(np.linalg.norm(z)),
        }

    report = {
        "status": "completed (CPU diagnostic; not a policy/benchmark result)",
        "zarr": metas[0]["zarr"],
        "anchors": int(anchors.size),
        "split": "grouped by data/source_episode_id, fractions 0.6/0.2/0.2",
        "split_seed": args.split_seed,
        "split_counts_anchors": {k: int(v.sum()) for k, v in split.items()},
        "adapter": metas[0]["adapter"],
        "feature_dirs": [str(d) for d in feature_dirs],
        "models": [m["model"]["kind"] for m in metas],
        "ridge": "standardized features (train stats), alpha in logspace(-2,5,15) selected by validation pooled R2",
        "results": results,
        "contact_sensitivity": sensitivity,
    }
    out_dir = root_path(args.output_dir or REPORT_ROOT / f"{Path(task.group).stem}_{feature_dirs[0].name}")
    write_json(out_dir / "probe.json", report)
    lines = [
        f"# LingBot-Touch probe on {Path(task.group).stem}",
        "",
        f"- Status: {report['status']}",
        f"- Features: {', '.join(report['feature_dirs'])}",
        f"- Split: {report['split']} (seed {args.split_seed}); anchors {report['split_counts_anchors']}",
        "- Metric: test pooled R2 (1 - SSE/SST over target dims); higher is better.",
        "",
        "| target | " + " | ".join(feature_sets) + " |",
        "|---|" + "---:|" * len(feature_sets),
    ]
    for tname, row in results.items():
        cells = [f"{row[f]['test_pooled_r2']:.3f}" if f in row else "-" for f in feature_sets]
        lines.append(f"| {tname} (n_test={row['n']['test']}) | " + " | ".join(cells) + " |")
    lines += ["", "Contact sensitivity (L2 distance to zero-input embedding / median feature norm):", ""]
    lines += [f"- {k}: {json.dumps(v)}" for k, v in sensitivity.items()]
    (out_dir / "probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    def adapter_args(p):
        p.add_argument("--zarr", required=True, help="3DTacDex <task>.zarr.zip")
        p.add_argument("--group", help="Zarr group inside the ZIP (default: the single <task>.zarr)")
        p.add_argument("--max-segments", type=int, help="Use only the first N segments (smoke tests)")
        p.add_argument("--pads", choices=("tip", "tip_pulp"), default="tip")
        p.add_argument("--frames", type=int, default=2, help="Window length T (multiple of tubelet size)")
        p.add_argument("--scales-xyz", type=float, nargs=3, default=(30.0, 30.0, 30.0))
        p.add_argument("--axis-signs", type=float, nargs=3, default=(1.0, 1.0, 1.0))
        p.add_argument("--radius-mm", type=float, default=14.0)
        p.add_argument("--origin-mm", type=float, nargs=3, default=(0.0, 13.25, 3.0))
        p.add_argument("--pulp-gap-mm", type=float, default=10.0)

    p = sub.add_parser("inspect", help="Adapter-only statistics (no model)")
    adapter_args(p)
    p.add_argument("--output", help="Optional JSON output path")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("extract", help="Encode windows with frozen LingBot-Touch and save pooled features")
    adapter_args(p)
    p.add_argument("--bundle", help="Converted Flax bundle dir (scripts/convert_lingbot_touch.py output)")
    p.add_argument("--random-init-seed", type=int, help="Random-init architecture control instead of --bundle")
    p.add_argument("--tag", required=True, help="Output subdirectory name")
    p.add_argument("--output-dir", help=f"Override output dir (default {FEATURE_ROOT}/<task>/<tag>)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--save-last-tokens", action="store_true", help="Also save last-tubelet tokens (float16)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--allow-accelerator", action="store_true", help="Permit a non-CPU JAX device")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("probe", help="Ridge diagnostics on extracted features")
    p.add_argument("--features", nargs="+", required=True, help="One or more extract output dirs (same archive)")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--output-dir", help=f"Override report dir (default {REPORT_ROOT}/<task>_<first tag>)")
    p.set_defaults(func=cmd_probe)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
