#!/usr/bin/env python3
"""Plan or prepare the CPU-only Assembly experiment protocol, without training."""

import argparse
from dataclasses import asdict
import json
import os


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "seal-probes"))
    parser.add_argument("--config", default="configs/trlt/3dtacdex_assembly/trlt_alternating.json")
    parser.add_argument("--data-root")
    parser.add_argument("--protocol-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--base-selection-seal")
    parser.add_argument("--probe", action="append", default=[], help="VARIANT=CHECKPOINT_DIR")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    from openpi.trlt.config import TemporalConfig
    from openpi.trlt.config import root_path
    from openpi.trlt.tacdex_data import DEFAULT_PROTOCOL
    from openpi.trlt.tacdex_data import DEFAULT_ROOT
    from openpi.trlt.tacdex_data import grouped_split
    from openpi.trlt.tacdex_data import inventory
    from openpi.trlt.tacdex_data import prepare_protocol
    from openpi.trlt.tacdex_data import tactile_preprocessing_config
    from openpi.trlt.tacdex_experiment import seal_probe_selections

    if args.command == "seal-probes":
        if not args.output:
            raise ValueError("seal-probes requires --output")
        probes = {}
        for spec in args.probe:
            if "=" not in spec:
                raise ValueError(f"Expected VARIANT=CHECKPOINT_DIR, got {spec!r}")
            variant, ckpt = spec.split("=", 1)
            if not variant or not ckpt or variant in probes:
                raise ValueError(f"Invalid or duplicate --probe specification: {spec!r}")
            probes[variant] = ckpt
        sealed = seal_probe_selections(
            probes,
            args.output,
            protocol_dir=args.protocol_dir or DEFAULT_PROTOCOL,
            seed=args.seed,
            base_selection_seal=args.base_selection_seal,
        )
        print(json.dumps(sealed, indent=2))
        return

    raw = json.loads(root_path(args.config).read_text())
    if raw["dataset_kind"] != "3dtacdex_assembly":
        raise ValueError("Assembly config required")
    root = args.data_root or raw.get("data_root", DEFAULT_ROOT)
    destination = args.protocol_dir or raw.get("protocol_dir", DEFAULT_PROTOCOL)
    temporal = TemporalConfig(**raw["temporal"])
    source = inventory(root)
    split = grouped_split(source, raw.get("split_seed", 42))
    result = {
        "status": "plan-only",
        "device": "CPU",
        "temporal": asdict(temporal),
        "splits": {
            name: {
                "segments": len(ids),
                "trajectories": len({source["episodes"][eid]["source_episode_id"] for eid in ids}),
                "frames": sum(source["episodes"][eid]["length"] for eid in ids),
            }
            for name, ids in split["splits"].items()
        },
        "training_steps": 0,
        "tactile_preprocessing": tactile_preprocessing_config(raw.get("tactile_preprocessing")),
    }
    if args.command == "prepare":
        result["protocol"] = prepare_protocol(root, destination, seed=raw.get("split_seed", 42), temporal=temporal)
        result["status"] = "prepared"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
