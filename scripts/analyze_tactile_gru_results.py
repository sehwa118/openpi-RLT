#!/usr/bin/env python3
"""Aggregate the Assembly tactile MLP-GRU test evaluations (docs/trlt/assembly_tactile_gru_plan.md §9).

Reads `<logs>/seed<S>/eval_<variant>/predictions.jsonl.gz` written by `trex_offline.py evaluate` and reports,
per variant and seed and as 3-seed mean/std: normalized trajectory-macro MSE (primary), first action, arm/hand/
per-finger, and contact-change queries (definition fixed from train before test access). Paired differences use
the source trajectory (episode_id) as the resampling unit: per trajectory, the seed-mean MSE difference is
bootstrapped; per-seed directions are reported separately. Frames are never treated as independent samples.
CPU only; reads files, trains nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

DEFAULT_VARIANTS = ("b0", "m1", "m5", "m5s", "g5")
PAIRS = (
    ("g5", "m5"),
    ("g5", "m5s"),
    ("m5s", "m5"),
    ("m5", "m1"),
    ("m1", "b0"),
    ("m5", "b0"),
    ("m5s", "b0"),
    ("g5", "b0"),
)
GROUPS = {
    "all": slice(0, 22),
    "arm": slice(0, 6),
    "hand": slice(6, 22),
    "index": slice(6, 10),
    "middle": slice(10, 14),
    "ring": slice(14, 18),
    "thumb": slice(18, 22),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", required=True, type=Path, nargs="+", help="Launcher <out-root>/logs (one or more)")
    p.add_argument("--normalization", required=True, type=Path)
    p.add_argument("--contact-definition", required=True, type=Path)
    p.add_argument("--test-touch", required=True, type=Path, help="raw_t5 test cache (contact-change selection)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    p.add_argument("--pairs", nargs="+", help="A:B differences (default: the original M/G-vs-B0 set)")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--output", required=True, type=Path)
    return p


def variant_name(v: str) -> str:
    return "part_attention_hierarchical_drop01" if v == "b0" else f"touch_part_attention_hierarchical_tgru_{v}"


def eval_dir(roots: list[Path], seed: int, v: str) -> Path:
    found = [r / f"seed{seed}" / f"eval_{variant_name(v)}" for r in roots]
    found = [d for d in found if (d / "metrics.json").exists()]
    if not found:
        raise FileNotFoundError(f"{v} seed{seed}: no evaluation under {roots}")
    # The same checkpoint may be re-evaluated under several roots (e.g. B0): require identical primary metrics.
    primary = {
        json.loads((d / "metrics.json").read_text())["metrics"]["primary_normalized_episode_macro_mse"] for d in found
    }
    if len(primary) != 1:
        raise ValueError(f"{v} seed{seed}: evaluations disagree across roots {found}")
    return found[0]


def contact_keys(test_touch: Path, threshold: float) -> set[tuple[int, int]]:
    tokens = np.load(test_touch / "tokens.npy").astype(np.float32)
    tokens = tokens.reshape(*tokens.shape[:3], -1, 3)
    delta = np.abs(tokens[..., -1, :] - tokens[..., -2, :]).mean(axis=(1, 2, 3))
    with np.load(test_touch / "keys.npz") as keys:
        seg, frame = keys["segment_id"], keys["query_frame_index"]
    return {(int(s), int(f)) for s, f, d in zip(seg, frame, delta, strict=True) if d >= threshold}


def load_errors(path: Path, std: np.ndarray, contact: set[tuple[int, int]]):
    """Normalized squared errors [R, 22], validity [R, 22], and per-row metadata."""
    err, valid, episode, horizon, is_contact = [], [], [], [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            r = json.loads(line)
            mask = np.asarray(r["action_valid_mask"], bool)  # element mask, as ActionMetrics (not row "valid")
            diff = (np.asarray(r["predicted_action"]) - np.asarray(r["gt_action"])) / std
            err.append(np.where(mask, diff**2, 0.0))
            valid.append(mask)
            episode.append(int(r["episode_id"]))
            horizon.append(int(r["horizon_offset"]))
            is_contact.append((int(r["segment_id"]), int(r["query_frame_index"])) in contact)
    return np.asarray(err), np.asarray(valid), np.asarray(episode), np.asarray(horizon), np.asarray(is_contact)


def per_episode(err, valid, episode, rows, cols):
    out = {}
    for e in np.unique(episode[rows]):
        sel = rows & (episode == e)
        n = valid[sel][:, cols].sum()
        if n:
            out[int(e)] = float(err[sel][:, cols].sum() / n)
    return out


def metrics_for(path, std, contact):
    err, valid, episode, horizon, is_contact = load_errors(path, std, contact)
    every = np.ones(len(err), bool)
    result, per_traj = {}, {}
    for name, cols in GROUPS.items():
        per_traj[name] = per_episode(err, valid, episode, every, cols)
    per_traj["first_action"] = per_episode(err, valid, episode, horizon == 0, GROUPS["all"])
    per_traj["contact_change"] = per_episode(err, valid, episode, is_contact, GROUPS["all"])
    for name, values in per_traj.items():
        result[f"{name}_trajectory_macro"] = float(np.mean(list(values.values())))
    result["contact_change_micro"] = float(err[is_contact].sum() / max(valid[is_contact].sum(), 1))
    result["contact_change_rows"] = int(is_contact.sum())
    result["contact_change_trajectories"] = len(per_traj["contact_change"])
    result["trajectories"] = len(per_traj["all"])
    return result, per_traj


def bootstrap(diffs: np.ndarray, samples: int, rng) -> dict:
    boot = diffs[rng.integers(len(diffs), size=(samples, len(diffs)))].mean(axis=1)
    return {
        "mean": float(diffs.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "trajectories": len(diffs),
        "fraction_trajectories_improved": float((diffs < 0).mean()),
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists")
    norm = json.loads(args.normalization.read_text())
    std = np.asarray(norm["action"]["std"], np.float64)
    definition = json.loads(args.contact_definition.read_text())
    contact = contact_keys(args.test_touch, definition["threshold"])
    rng = np.random.default_rng(42)
    runs, traj = {}, {}
    for v in args.variants:
        for s in args.seeds:
            path = eval_dir(args.logs, s, v) / "predictions.jsonl.gz"
            runs[(v, s)], traj[(v, s)] = metrics_for(path, std, contact)
            reference = json.loads((path.parent / "metrics.json").read_text())["metrics"]
            got, want = runs[(v, s)]["all_trajectory_macro"], reference["primary_normalized_episode_macro_mse"]
            if abs(got - want) > 1e-6 * max(1.0, abs(want)):
                raise ValueError(f"{v} seed{s}: recomputed primary {got} != evaluator {want}")
    keys = [k for k in runs[(args.variants[0], args.seeds[0])] if k.endswith(("_macro", "_micro"))]
    summary = {}
    for v in args.variants:
        summary[v] = {}
        for k in keys:
            vals = np.asarray([runs[(v, s)][k] for s in args.seeds])
            summary[v][k] = {
                "per_seed": vals.tolist(),
                "mean": float(vals.mean()),
                "std_ddof1": float(vals.std(ddof=1)),
            }
    pairs = {}
    pair_list = [tuple(x.split(":")) for x in args.pairs] if args.pairs else PAIRS
    for a, b in pair_list:
        if a not in args.variants or b not in args.variants:
            if args.pairs:
                raise ValueError(f"pair {a}:{b} needs both variants in --variants")
            continue
        entry = {}
        for metric in ("all", "first_action", "hand", "arm", "contact_change"):
            common = sorted(set.intersection(*(set(traj[(x, s)][metric]) for x in (a, b) for s in args.seeds)))
            per_seed = [
                float(np.mean([traj[(a, s)][metric][e] - traj[(b, s)][metric][e] for e in common])) for s in args.seeds
            ]
            seed_mean_diff = np.asarray(
                [np.mean([traj[(a, s)][metric][e] - traj[(b, s)][metric][e] for s in args.seeds]) for e in common]
            )
            base = float(np.mean([np.mean([traj[(b, s)][metric][e] for s in args.seeds]) for e in common]))
            entry[metric] = {
                **bootstrap(seed_mean_diff, args.bootstrap_samples, rng),
                "relative_to_reference": float(seed_mean_diff.mean() / base),
                "per_seed_difference": per_seed,
                "seeds_improved": int(sum(d < 0 for d in per_seed)),
            }
        pairs[f"{a}_minus_{b}"] = entry
    report = {
        "lower_is_better": True,
        "units": "MSE of actions normalized by the protocol train action std (normalization.json)",
        "aggregation": "trajectory (episode_id) macro; pairs: seed-mean per-trajectory difference, bootstrap over trajectories",
        "bootstrap_samples": args.bootstrap_samples,
        "seeds": args.seeds,
        "contact_change_definition": definition,
        "per_run": {f"{v}_seed{s}": r for (v, s), r in runs.items()},
        "summary": summary,
        "pairs": pairs,
        "note": "Exploratory: this Assembly test split was already used by earlier follow-ups (B0 evaluated before).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for v in args.variants:
        s = summary[v]
        print(
            f"{v:4s} primary {s['all_trajectory_macro']['mean']:.6f}±{s['all_trajectory_macro']['std_ddof1']:.6f}"
            f"  first {s['first_action_trajectory_macro']['mean']:.6f}  hand {s['hand_trajectory_macro']['mean']:.6f}"
            f"  arm {s['arm_trajectory_macro']['mean']:.6f}  contact {s['contact_change_trajectory_macro']['mean']:.6f}"
        )
    for name, entry in pairs.items():
        e = entry["all"]
        print(
            f"{name:14s} Δprimary {e['mean']:+.6f} CI[{e['ci95'][0]:+.6f},{e['ci95'][1]:+.6f}] seeds↓{e['seeds_improved']}/3"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
