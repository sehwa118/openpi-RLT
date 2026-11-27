#!/usr/bin/env python3
"""Pool the held-out evaluations of `run_tactile_gru_cv.py` (docs/trlt/assembly_tactile_gru_cv_plan.md).

Every train+validation trajectory is held out exactly once, so each variant has one error per trajectory, computed
by a model that never trained or early-stopped on it. Paired differences are bootstrapped over trajectories;
per-fold directions are reported separately. Metric definitions are those of analyze_tactile_gru_results.py
(recomputed primary checked against each fold's evaluator metrics.json). CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_tactile_gru_results import bootstrap
from analyze_tactile_gru_results import contact_keys
from analyze_tactile_gru_results import metrics_for
from analyze_tactile_gru_results import variant_name

METRICS = ("all", "first_action", "arm", "hand", "contact_change")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv-root", required=True, type=Path)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="A:B differences")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--normalization", required=True, type=Path)
    p.add_argument("--contact-definition", required=True, type=Path)
    p.add_argument("--touch-t5-root", required=True, type=Path, help="raw_t5 directory with train/ and validation/")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists")
    std = np.asarray(json.loads(args.normalization.read_text())["action"]["std"], np.float64)
    definition = json.loads(args.contact_definition.read_text())
    train_keys = contact_keys(args.touch_t5_root / "train", definition["threshold"])
    val_keys = contact_keys(args.touch_t5_root / "validation", definition["threshold"])
    contact = train_keys | val_keys
    rng = np.random.default_rng(42)
    traj, per_fold, fold_of = {}, {}, {}
    for v in args.variants:
        traj[v] = {m: {} for m in METRICS}
        per_fold[v] = []
        for k in range(args.folds):
            d = args.cv_root / "logs" / f"fold{k}" / f"eval_{variant_name(v)}"
            result, per = metrics_for(d / "predictions.jsonl.gz", std, contact)
            want = json.loads((d / "metrics.json").read_text())["metrics"]["primary_normalized_episode_macro_mse"]
            if abs(result["all_trajectory_macro"] - want) > 1e-6 * max(1.0, abs(want)):
                raise ValueError(f"{v} fold{k}: recomputed primary {result['all_trajectory_macro']} != {want}")
            per_fold[v].append(result["all_trajectory_macro"])
            for m in METRICS:
                overlap = set(per[m]) & set(traj[v][m])
                if overlap:
                    raise ValueError(f"{v} fold{k}: trajectories {sorted(overlap)} held out twice")
                traj[v][m].update(per[m])
            for e in per["all"]:
                if fold_of.setdefault(e, k) != k:
                    raise ValueError(f"trajectory {e} in different folds across variants")
    n_traj = {v: len(traj[v]["all"]) for v in args.variants}
    if len(set(n_traj.values())) != 1:
        raise ValueError(f"variants cover different trajectories: {n_traj}")
    summary = {
        v: {
            **{f"{m}_trajectory_macro": float(np.mean(list(traj[v][m].values()))) for m in METRICS},
            "per_fold_primary": per_fold[v],
            "trajectories": n_traj[v],
        }
        for v in args.variants
    }
    pairs = {}
    for pair in args.pairs:
        a, b = pair.split(":")
        entry = {}
        for m in METRICS:
            common = sorted(set(traj[a][m]) & set(traj[b][m]))
            diffs = np.asarray([traj[a][m][e] - traj[b][m][e] for e in common])
            folds = [
                float(np.mean([d for d, e in zip(diffs, common, strict=True) if fold_of[e] == k]))
                for k in range(args.folds)
            ]
            base = float(np.mean([traj[b][m][e] for e in common]))
            entry[m] = {
                **bootstrap(diffs, args.bootstrap_samples, rng),
                "relative_to_reference": float(diffs.mean() / base),
                "per_fold_difference": folds,
                "folds_improved": int(sum(d < 0 for d in folds)),
            }
        pairs[f"{a}_minus_{b}"] = entry
    report = {
        "lower_is_better": True,
        "design": "trajectory-grouped K-fold over train+validation; held-out fold evaluated once per variant",
        "folds": args.folds,
        "bootstrap_samples": args.bootstrap_samples,
        "contact_change_definition": definition,
        "summary": summary,
        "pairs": pairs,
        "note": "Stage A (frozen) was trained on the train split, so absolute errors on train trajectories are "
        "optimistic; the bias is shared by all variants. The test split is not read.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for v, s in summary.items():
        folds = " ".join(f"{x:.5f}" for x in s["per_fold_primary"])
        print(f"{v:4s} pooled {s['all_trajectory_macro']:.6f} (n={s['trajectories']})  folds {folds}")
    for name, entry in pairs.items():
        e = entry["all"]
        print(
            f"{name:12s} Δprimary {e['mean']:+.6f} CI[{e['ci95'][0]:+.6f},{e['ci95'][1]:+.6f}]"
            f" {100 * e['relative_to_reference']:+.1f}% folds↓{e['folds_improved']}/{args.folds}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
