#!/usr/bin/env python3
"""Seal both validation-selected LingBot checkpoints before explicit test access."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-dir", default="datasets/3dtacdex_assembly/protocol")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import json

    from openpi.trlt.tacdex_experiment import seal_lingbot_selections

    print(
        json.dumps(
            seal_lingbot_selections(
                args.representation, args.probe, args.output, protocol_dir=args.protocol_dir, seed=args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
