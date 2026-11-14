"""Explicit host-only preparation; never initiates model training."""

import argparse
import json

from openpi.trlt.config import root_path
from openpi.trlt.data import build_query_manifest
from openpi.trlt.data import build_subset_manifest
from openpi.trlt.data import compute_train_normalization
from openpi.trlt.data import download_subset
from openpi.trlt.data import make_episode_splits
from openpi.trlt.data import plan_download
from openpi.trlt.data import validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "split", "normalize", "validate"))
    parser.add_argument("--data-root", default="datasets/trex_squeeze")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=100.0,
        help="Decimal GB of missing source files; excludes existing files",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--packet-workers", type=int, default=8, help="CPU fork workers for MP4 PTS scan, capped at 8")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-manifest")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-video-decode", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--priority-episode-ids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    root = root_path(args.data_root)
    if args.command == "plan":
        manifest = build_subset_manifest(root, args.dataset_revision)
        result = plan_download(root, manifest)
    elif args.command == "download":
        result = download_subset(
            root,
            max_download_gb=args.max_download_gb,
            workers=args.workers,
            retry_failed=args.retry_failed,
            priority_episode_ids=args.priority_episode_ids,
        )
    elif args.command == "split":
        result = make_episode_splits(
            root / "subset_manifest.json", args.split_seed, args.split_manifest or root / "splits.json"
        )
        build_query_manifest(root / "subset_manifest.json", result, output_root=root / "query_manifests")
    elif args.command == "normalize":
        result = compute_train_normalization(root, args.split_manifest, args.epsilon)
    else:
        result = validate_dataset(
            root,
            split_manifest=args.split_manifest,
            decode_videos=not args.skip_video_decode,
            max_episodes=args.max_episodes,
            workers=args.workers,
            packet_workers=args.packet_workers,
        )
    # Exact file/pointer inventories live on disk; keep terminal output readable.
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "episodes")}, indent=2))
    if args.command == "validate" and result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
