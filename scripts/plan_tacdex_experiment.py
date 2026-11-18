#!/usr/bin/env python3
"""Prepare ordered Assembly commands or seal validation-selected checkpoints; never launch jobs."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--device", default="0")
    seal = sub.add_parser("seal")
    seal.add_argument("--checkpoint-root", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--protocol-dir", default="datasets/3dtacdex_assembly/protocol")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    from openpi.trlt.tacdex_experiment import seal_selections
    from openpi.trlt.tacdex_experiment import write_plan

    if args.command == "plan":
        result = write_plan(args.run_id, device=args.device)
    else:
        result = seal_selections(args.checkpoint_root, args.output, protocol_dir=args.protocol_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
