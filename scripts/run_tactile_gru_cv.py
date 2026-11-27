#!/usr/bin/env python3
"""Trajectory-grouped K-fold CV of tactile Stage B conditions (docs/trlt/assembly_tactile_gru_cv_plan.md).

Pools the frozen Assembly *train* and *validation* latent (and raw touch) caches, never the test split. For each
outer fold k: train on K-2 folds, early-stop/select on fold k+1 (validation normalized_action_micro_mse, as in
Stage B), then evaluate the best checkpoint once on held-out fold k. Rows are read in place; caches are not copied.

  $GRU_PY scripts/run_tactile_gru_cv.py --latent-root <…/trlt_self_only/seed0> --touch-root <touch_cache> \
      --normalization <protocol/normalization.json> --out-root <…/experiments/cv> --variants b0 m5 m20 g20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs/trlt/3dtacdex_assembly"
FRAMES = {"m1": 1, "m5": 5, "m5s": 5, "g5": 5, "m10": 10, "m10s": 10, "g10": 10, "m20": 20, "m20s": 20, "g20": 20}


def config_path(v: str) -> Path:
    if v == "b0":
        return CONFIGS / "attention_actor/probe_part_attn_hierarchical_drop01.json"
    if v not in FRAMES:
        raise ValueError(f"unknown variant {v}")
    return CONFIGS / f"tactile_gru/probe_tgru_{v}.json"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-root", required=True, type=Path, help="Directory holding train/ and validation/")
    p.add_argument("--touch-root", required=True, type=Path, help="Holds raw_t<T>/{train,validation}")
    p.add_argument("--normalization", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--outer", type=int, nargs="+", help="Outer folds to run (default: all)")
    p.add_argument("--seed", type=int, default=0, help="Stage B training seed")
    p.add_argument("--device", default="0", help="Single GPU index, or 'cpu' for CPU-only checks")
    p.add_argument("--mem-fraction", default="0.05")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true", help="Build folds and print sizes; no training")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    out = args.out_root.resolve()
    user = os.environ.get("USER", "")
    if not user or not str(out).startswith(f"/workspace/users/{user}/"):
        raise ValueError(f"--out-root must be under /workspace/users/{user}/")
    if args.device == "cpu":
        os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        if not args.device.isdigit():
            raise ValueError("--device must be one GPU index or 'cpu'")
        os.environ.update(
            CUDA_VISIBLE_DEVICES=args.device,
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION=args.mem_fraction,
            JAX_DEFAULT_MATMUL_PRECISION="highest",
        )
    sys.path.insert(0, str(REPO / "src"))
    import jax

    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.config import probe_variant_name
    from openpi.trlt.cv_folds import build_cv_datasets
    from openpi.trlt.runner import LatentDataset
    from openpi.trlt.runner import evaluate_action_probe
    from openpi.trlt.runner import train_action_probe
    from openpi.trlt.touch_tokens import TouchLatentDataset

    jax.config.update("jax_default_matmul_precision", "highest")
    if args.device != "cpu":
        from openpi.trlt.runner import assert_gpu

        assert_gpu()
    outers = args.outer if args.outer is not None else list(range(args.folds))
    plan = []
    for v in args.variants:
        probe = ProbeConfig(**json.loads(config_path(v).read_text())["probe"])
        name = probe_variant_name(probe)
        sources = []
        for split in ("train", "validation"):
            latents = LatentDataset(args.latent_root / split)
            if latents.provenance.get("split") != split:
                raise ValueError(f"{args.latent_root / split} is not the {split} split")
            sources.append(
                latents if v == "b0" else TouchLatentDataset(latents, args.touch_root / f"raw_t{FRAMES[v]}" / split)
            )
        for k in outers:
            datasets, spec = build_cv_datasets(sources, k=args.folds, seed=args.fold_seed, outer=k)
            fold_file = out / "folds" / f"fold{k}.json"
            assignment = {key: spec[key] for key in ("k", "seed", "outer_fold", "early_stop_fold", "assignment")}
            if fold_file.exists():
                if json.loads(fold_file.read_text())["assignment_spec"] != assignment:
                    raise ValueError(f"{fold_file}: fold assignment differs from the recorded one")
            elif not args.dry_run:
                fold_file.parent.mkdir(parents=True, exist_ok=True)
                fold_file.write_text(
                    json.dumps({"assignment_spec": assignment, "spec_by_variant": {}}, indent=2) + "\n"
                )
            sizes = {role: len(d) for role, d in datasets.items()}
            trajectories = {
                role: len({int(e) for e, f in spec["assignment"].items() if _role(f, k, args.folds) == role})
                for role in datasets
            }
            plan.append((v, name, k, probe, datasets, spec, sizes, trajectories))
            print(f"{v:4s} fold{k}: rows {sizes} trajectories {trajectories} fold_hash {spec['hash'][:8]}")
    if args.dry_run:
        return 0
    for v, name, k, probe, datasets, spec, _, _ in plan:
        ckpt = out / "checkpoints" / f"fold{k}" / f"probe_{name}"
        log = out / "logs" / f"fold{k}" / f"probe_{name}"
        evaluation = out / "logs" / f"fold{k}" / f"eval_{name}"
        for path in (ckpt, evaluation):
            if path.exists():
                raise FileExistsError(f"{path} exists; inspect it instead of re-running")
        fold_file = out / "folds" / f"fold{k}.json"
        record = json.loads(fold_file.read_text())
        record["spec_by_variant"][v] = {"hash": spec["hash"], "touch_cache_hashes": spec["touch_cache_hashes"]}
        fold_file.write_text(json.dumps(record, indent=2) + "\n")
        print(f"== train {v} fold{k}", flush=True)
        train_action_probe(
            datasets["train"], datasets["early_stop"], probe, str(ckpt),
            seed=args.seed, max_steps=args.max_steps, log_dir=str(log), num_workers=args.num_workers,
        )  # fmt: skip
        print(f"== evaluate {v} fold{k} (held-out)", flush=True)
        report = evaluate_action_probe(
            datasets["heldout"], str(ckpt / "best"), str(args.normalization), str(evaluation)
        )
        print(
            f"{v} fold{k} held-out primary {report['metrics']['primary_normalized_episode_macro_mse']:.6f}", flush=True
        )
    return 0


def _role(fold: int, outer: int, k: int) -> str:
    from openpi.trlt.cv_folds import role_of

    return role_of(fold, outer, k)


if __name__ == "__main__":
    raise SystemExit(main())
