"""Explicit CPU-only Assembly download, command audit and LeRobot conversion."""

import argparse
import json
import os

# Data preparation must not initialize accelerators even on a shared GPU server.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.trlt.config import root_path
from openpi.trlt.tacdex_source import DEFAULT_REPORT
from openpi.trlt.tacdex_source import DEFAULT_ROOT
from openpi.trlt.tacdex_source import audit_assembly
from openpi.trlt.tacdex_source import download_assembly
from openpi.trlt.tacdex_source import download_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "audit", "convert", "validate"))
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--cohort",
        choices=("paper", "all"),
        default="all",
        help="paper excludes source_label=nouse; all preserves every source segment",
    )
    parser.add_argument("--threads", type=int, default=2, help="CPU encoder threads per camera")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rgb-boundaries-only", action="store_true", help="Explicitly limit pixel comparisons")
    args = parser.parse_args()
    # Set before datasets/LeRobot imports; do not redirect HF_HOME or auth.
    os.environ["HF_DATASETS_CACHE"] = str(root_path(args.data_root) / "reader_cache")
    if args.command == "plan":
        result = download_plan(args.data_root)
    elif args.command == "download":
        result = download_assembly(args.data_root)
    elif args.command == "audit":
        result = audit_assembly(args.data_root, args.report_dir)
    else:
        from openpi.trlt.tacdex_lerobot import convert_assembly
        from openpi.trlt.tacdex_lerobot import validate_assembly

        if args.command == "convert":
            manifest = convert_assembly(
                args.data_root, output_dir=args.output_dir, cohort=args.cohort, threads=args.threads, resume=args.resume
            )
            result = {k: v for k, v in manifest.items() if k not in ("segments", "source_attributes", "identity")}
        else:
            result = validate_assembly(
                args.data_root,
                output_dir=args.output_dir,
                report_dir=args.report_dir,
                all_rgb=not args.rgb_boundaries_only,
            )
            result = {k: v for k, v in result.items() if k != "file_sha256"}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
