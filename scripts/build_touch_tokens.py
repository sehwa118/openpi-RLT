#!/usr/bin/env python3
"""Build a frozen per-finger touch-token cache for one 3DTacDex Assembly split (CPU only).

Sources: lingbot (pretrained bundle), random (random-init control), raw (normalized taxels).
The test split requires --selection-seal from seal-probes, so test tokens are produced only
after every compared Stage B checkpoint has been selected on validation and sealed.
"""

import argparse
import json
import os
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--data-root", required=True, help="Assembly LeRobot root (lerobot_all)")
    p.add_argument("--split", choices=("train", "validation", "test"), required=True)
    p.add_argument("--source", choices=("lingbot", "random", "raw"), required=True)
    p.add_argument("--bundle", help="Converted LingBot Flax bundle (source=lingbot)")
    p.add_argument("--random-init-seed", type=int, help="source=random")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--selection-seal", help="Required for --split test")
    p.add_argument("--pads", choices=("tip", "tip_pulp"), default="tip")
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--scales-xyz", type=float, nargs=3, default=(30.0, 30.0, 30.0))
    p.add_argument("--axis-signs", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-queries", type=int, help="Smoke test only: first N queries (marked partial)")
    p.add_argument("--threads", type=int, default=16, help="CPU threads for XLA/BLAS")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    # Hard CPU-only guard: set before JAX is imported anywhere in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    if "jax" in sys.modules:
        raise RuntimeError("JAX imported before the CPU guard")
    import jax

    if any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("Touch cache building must run on CPU")

    from openpi.trlt.tacdex_lingbot import PaxiniLingBotConfig
    from openpi.trlt.touch_tokens import build_assembly_touch_cache

    adapter = PaxiniLingBotConfig(
        pads=args.pads, frames=args.frames, scales_xyz=tuple(args.scales_xyz), axis_signs=tuple(args.axis_signs)
    )
    manifest = build_assembly_touch_cache(
        args.protocol_dir,
        args.data_root,
        args.split,
        args.output_dir,
        selection_seal=args.selection_seal,
        max_queries=args.max_queries,
        source=args.source,
        adapter=adapter,
        bundle=args.bundle,
        random_init_seed=args.random_init_seed,
        batch_size=args.batch_size,
        progress=True,
    )
    print(json.dumps({k: manifest[k] for k in ("touch_identity", "count", "stats", "build_seconds")}, indent=2))


if __name__ == "__main__":
    main()
