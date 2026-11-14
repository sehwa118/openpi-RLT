"""Extract frozen anchor features; defaults to a size-only plan, not a backbone run."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/trlt/trlt_alternating.json")
    parser.add_argument("--data-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--output-dir")
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value; GPU required for extraction")
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--extract", action="store_true", help="Execute extraction; otherwise print plan only")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force-cache-rebuild", action="store_true")
    args = parser.parse_args()
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_PLATFORMS"] = "cuda" if args.extract and not args.plan_only else "cpu"
    from openpi.trlt.config import TemporalConfig
    from openpi.trlt.config import root_path
    from openpi.trlt.data import TrexDataset
    from openpi.trlt.features import extract_visual_cache

    config = json.loads(root_path(args.config).read_text())
    temporal = TemporalConfig(**config["temporal"])
    data_root = root_path(args.data_root or config.get("data_root", "datasets/trex_squeeze"))
    split_manifest = root_path(args.split_manifest) if args.split_manifest else data_root / "splits.json"
    normalization = root_path(args.normalization) if args.normalization else data_root / "normalization.json"
    output_dir = root_path(args.output_dir) if args.output_dir else data_root / "visual_cache"
    backbone = args.backbone_checkpoint or config.get(
        "backbone_checkpoint", "checkpoints/openpi-assets/checkpoints/pi05_base/params"
    )
    dataset = TrexDataset(
        data_root=data_root,
        split_manifest=split_manifest,
        normalization=normalization,
        split=args.split,
        temporal=temporal,
        tactile_resolution=config["model"]["tactile_resolution"],
    )
    # Source provenance must come from the pinned selection, not a caller-supplied mutable revision.
    provenance = dataset.provenance
    result = extract_visual_cache(
        dataset,
        backbone_checkpoint=backbone,
        output_dir=output_dir,
        revision=provenance["revision"],
        episode_manifest_hash=provenance["episode_manifest_hash"],
        temporal=temporal,
        max_anchors=args.max_anchors,
        episode_ids=args.episode_ids,
        force=args.force_cache_rebuild,
        plan_only=not args.extract or args.plan_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
