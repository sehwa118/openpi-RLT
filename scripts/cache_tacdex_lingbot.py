#!/usr/bin/env python3
"""Explicit frozen tactile extraction for Assembly; never starts training."""

import argparse
import json
from pathlib import Path
import sys

# Shared editable env may point at another worktree: always import this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only_2layer.json")
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 (physical GPU selected externally)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--selection-seal")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not args.device.startswith("cuda:") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("GPU extraction requires --device cuda:0 and explicit CUDA_VISIBLE_DEVICES")
    from openpi.trlt.config import ModelConfig
    from openpi.trlt.config import TemporalConfig
    from openpi.trlt.config import root_path
    from openpi.trlt.lingbot_touch import FrozenPaxiniEncoder
    from openpi.trlt.lingbot_touch import cache_identity
    from openpi.trlt.lingbot_touch import encoder_identity
    from openpi.trlt.lingbot_touch import extract_cache
    from openpi.trlt.tacdex_data import AssemblyDataset

    raw = json.loads(root_path(args.config).read_text())
    cfg = ModelConfig(**raw["model"])
    if raw.get("dataset_kind") != "3dtacdex_assembly" or cfg.tactile_encoder != "lingbot_touch":
        raise ValueError("An Assembly LingBot config is required")
    if args.split == "test":
        if args.max_samples is not None:
            raise ValueError("No partial test extraction")
        from openpi.trlt.tacdex_experiment import validate_selection_seal

        validate_selection_seal(args.selection_seal, protocol_dir=raw["protocol_dir"])
    ds = AssemblyDataset(
        raw["data_root"],
        raw["protocol_dir"],
        split=args.split,
        temporal=TemporalConfig(**raw["temporal"]),
        tactile_preprocessing=raw.get("tactile_preprocessing"),
    )
    opts = raw["lingbot_touch"]
    options = {k: opts[k] for k in ("source", "bundle", "geometry")}
    identity = encoder_identity(**options)
    plan = {
        "split": args.split,
        "queries": len(ds),
        "identity": cache_identity(ds, identity),
        "output": args.output_dir or opts["cache"],
        "device": args.device,
        "full_feature_bytes": len(ds) * 2 * 8 * 768 * 4,
        "status": "plan-only; no model initialization or extraction",
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    encoder = FrozenPaxiniEncoder(**options, device=args.device)
    result = extract_cache(
        ds,
        args.output_dir or opts["cache"],
        encoder,
        batch_size=args.batch_size or opts.get("batch_size", 32),
        shard_size=args.shard_size or opts.get("shard_size", 256),
        max_samples=args.max_samples,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
