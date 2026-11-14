#!/usr/bin/env python3
"""Explicit stages for T-Rex offline experiments. Training never runs implicitly."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
import os
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("train-representation", "export-latents", "train-probe", "evaluate", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument(
            "--model", choices=("rlt_original", "trlt_self_only", "trlt_cross_only", "trlt_alternating")
        )
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--device", default="0", help="Physical GPU index for CUDA_VISIBLE_DEVICES")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--output-dir")
        if name in ("train-representation", "train-probe", "smoke"):
            command.add_argument(
                "--max-steps", type=int, help="Total optimizer-step limit; overrides representation.max_steps"
            )
        if name in ("train-representation", "train-probe"):
            command.add_argument("--resume", action="store_true")
            command.add_argument("--no-progress", action="store_true")
            command.add_argument(
                "--num-workers", type=int, default=2, help="CPU DataLoader workers; 0 for synchronous loading"
            )
            command.add_argument("--prefetch-factor", type=int, default=2, help="Batches queued per CPU worker")
            command.add_argument("--wandb", choices=("disabled", "online", "offline"), default="disabled")
            command.add_argument("--wandb-project")
            command.add_argument("--wandb-entity")
            command.add_argument("--wandb-name")
            command.add_argument("--wandb-group")
        if name in ("train-representation", "export-latents"):
            command.add_argument("--data-root", default="datasets/trex_squeeze")
            command.add_argument("--split-manifest")
            command.add_argument("--normalization")
            command.add_argument("--visual-cache")
        if name == "export-latents":
            command.add_argument("--representation-checkpoint", required=True)
            command.add_argument("--split", choices=("train", "validation", "test"), default="train")
            command.add_argument("--batch-size", type=int, default=32)
            command.add_argument("--max-samples", type=int)
            command.add_argument("--force-cache-rebuild", action="store_true")
        if name == "train-probe":
            command.add_argument("--train-latents", required=True)
            command.add_argument("--validation-latents", required=True)
        if name == "evaluate":
            command.add_argument("--latents", required=True)
            command.add_argument("--probe-checkpoint", required=True)
            command.add_argument("--normalization", required=True)
            command.add_argument("--batch-size", type=int, default=256)
    summary = sub.add_parser("summarize")
    summary.add_argument("metrics", nargs="+", type=Path)
    summary.add_argument("--output", default="experiments/trlt/summary.json")
    summary.add_argument("--bootstrap-samples", type=int, default=2000)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    from openpi.trlt.config import ModelConfig
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.config import TemporalConfig
    from openpi.trlt.config import TrainConfig
    from openpi.trlt.config import root_path

    if args.command == "summarize":
        from openpi.trlt.metrics import summarize_results

        result = summarize_results(
            [root_path(path) for path in args.metrics], root_path(args.output), bootstrap_samples=args.bootstrap_samples
        )
        print(json.dumps(result, indent=2))
        return
    raw = json.loads(root_path(args.config).read_text()) if args.config else {}
    config = ModelConfig(**raw.get("model", {}))
    if args.model:
        config = replace(config, mode=args.model)
    temporal = TemporalConfig(**raw.get("temporal", {}))
    representation = TrainConfig(**raw.get("representation", {}))
    probe = ProbeConfig(**raw.get("probe", {}))
    if config.history_frames != temporal.tactile_history_frames or probe.horizon != temporal.action_horizon_frames:
        raise ValueError("Model/probe and temporal configuration disagree")
    if config.state_dim != 58 or probe.action_dim != 58:
        raise ValueError("T-Rex probe preserves all 58 state/action dimensions")
    if args.command != "smoke" and (config.num_sensors != 10 or config.tactile_channels != 1):
        raise ValueError("T-Rex requires ten grayscale deformation sensors")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "model": asdict(config),
                    "temporal": asdict(temporal),
                    "representation": asdict(representation),
                    "probe": asdict(probe),
                    "arguments": vars(args),
                    "status": "plan-only; no model initialization or training",
                },
                indent=2,
                default=str,
            )
        )
        return
    if not args.device or args.device == "cpu" or "," in args.device:
        raise ValueError("Select one explicitly allocated GPU index with --device")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    from openpi.trlt.runner import LatentDataset
    from openpi.trlt.runner import assert_gpu
    from openpi.trlt.runner import evaluate_action_probe
    from openpi.trlt.runner import export_frozen_latents
    from openpi.trlt.runner import synthetic_smoke
    from openpi.trlt.runner import train_action_probe
    from openpi.trlt.runner import train_representation

    assert_gpu()
    tracking = {key: getattr(args, "wandb_" + key, None) for key in ("project", "entity", "name", "group")}
    tracking["mode"] = getattr(args, "wandb", "disabled")
    base = f"checkpoints/trlt/{config.mode}/seed{args.seed}"
    logs = f"experiments/trlt/{config.mode}/seed{args.seed}"
    if args.command == "smoke":
        result = synthetic_smoke(
            args.output_dir or base + "/synthetic_smoke",
            mode=config.mode,
            seed=args.seed,
            max_steps=args.max_steps or 2,
        )
    elif args.command in ("train-representation", "export-latents"):
        from openpi.trlt.cache import FeatureCache
        from openpi.trlt.data import TrexDataset

        data_root = root_path(args.data_root)
        split_path = root_path(args.split_manifest) if args.split_manifest else data_root / "splits.json"
        norm_path = root_path(args.normalization) if args.normalization else data_root / "normalization.json"
        cache = FeatureCache(root_path(args.visual_cache) if args.visual_cache else data_root / "visual_cache")

        def dataset(split):
            return TrexDataset(
                data_root,
                split_path,
                norm_path,
                split=split,
                temporal=temporal,
                tactile_resolution=config.tactile_resolution,
                visual_cache=cache,
            )

        if args.command == "train-representation":
            result = train_representation(
                dataset("train"),
                dataset("validation"),
                config,
                representation,
                args.output_dir or base + "/representation",
                seed=args.seed,
                max_steps=args.max_steps,
                resume=args.resume,
                tracking=tracking,
                progress=not args.no_progress,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                log_dir=logs + "/representation",
            )
        else:
            result = export_frozen_latents(
                dataset(args.split),
                args.representation_checkpoint,
                args.output_dir or f"datasets/trex_squeeze/latents/{config.mode}/seed{args.seed}/{args.split}",
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                force_cache_rebuild=args.force_cache_rebuild,
            ).manifest
    elif args.command == "train-probe":
        result = train_action_probe(
            LatentDataset(args.train_latents),
            LatentDataset(args.validation_latents),
            probe,
            args.output_dir or base + "/probe",
            seed=args.seed,
            max_steps=args.max_steps,
            resume=args.resume,
            tracking=tracking,
            progress=not args.no_progress,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            log_dir=logs + "/probe",
        )
    else:
        result = evaluate_action_probe(
            LatentDataset(args.latents),
            args.probe_checkpoint,
            args.normalization,
            args.output_dir or logs + "/evaluation",
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
