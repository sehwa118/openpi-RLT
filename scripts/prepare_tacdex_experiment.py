#!/usr/bin/env python3
"""Plan or prepare the CPU-only Assembly experiment protocol, without training."""

import argparse
from dataclasses import asdict
import json
import os


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("plan", "prepare"))
    parser.add_argument("--config", default="configs/trlt/3dtacdex_assembly/trlt_alternating.json")
    parser.add_argument("--data-root")
    parser.add_argument("--protocol-dir")
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
    }
    if args.command == "prepare":
        result["protocol"] = prepare_protocol(root, destination, seed=raw.get("split_seed", 42), temporal=temporal)
        result["status"] = "prepared"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
