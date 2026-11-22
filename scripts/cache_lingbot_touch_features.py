#!/usr/bin/env python3
"""Explicit frozen tactile extraction. Does not start Stage A/B or extract another split."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trlt/lingbot_touch/trlt_self_only.json", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--device", default="cpu", help="cpu or one physical GPU index")
    parser.add_argument("--batch-size", type=int, default=4, help="Queries per encoder call (two hands per query)")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-test", action="store_true", help="Explicit final-stage test extraction after selections"
    )
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("Test features are final-stage only; explicitly select --allow-test after checkpoint selection")
    if args.device != "cpu" and (not args.device.isdigit()):
        parser.error("--device must be cpu or one physical GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from openpi.trlt.config import TemporalConfig
    from openpi.trlt.config import root_path
    from openpi.trlt.data import TrexDataset
    from openpi.trlt.lingbot_touch import extract_trex_tactile_cache

    raw = json.loads(root_path(args.config).read_text())
    if raw["model"].get("tactile_encoder") != "lingbot_touch":
        raise ValueError("Expected explicit LingBot model config")
    temporal = TemporalConfig(**raw["temporal"])
    if temporal.tactile_history_frames != 10:
        raise ValueError("Pinned T-Rex extractor requires10-frame windows")
    dataset = TrexDataset(data_root=raw.get("data_root", "datasets/trex_squeeze"), temporal=temporal, split=args.split)
    result = extract_trex_tactile_cache(
        dataset,
        raw["tactile_features"],
        device="cpu" if args.device == "cpu" else "cuda:0",
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
