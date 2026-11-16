#!/usr/bin/env python3
"""Explicit CPU-only hand geometry source/derivation/cache stages. Never trains."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from openpi.trlt.config import root_path
from openpi.trlt.geometry import DEFAULT_SPEC
from openpi.trlt.geometry import SOURCE_REPO
from openpi.trlt.geometry import SOURCE_REVISION
from openpi.trlt.geometry import derive_spec
from openpi.trlt.geometry import prepare_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch-model", "derive", "cache"))
    parser.add_argument("--source-dir", default=f"datasets/trex_squeeze/geometry_sources/{SOURCE_REVISION}")
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--data-root", default="datasets/trex_squeeze")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--report", help="Write execution result under experiments/")
    args = parser.parse_args()
    source = root_path(args.source_dir)
    if args.command == "fetch-model":
        records = []
        for side in ("left", "right"):
            base = f"dataset_quickstart/third_party/sharpa-urdf-usd-xml/wave_01/{side}_sharpa_wave"
            for name in (
                f"{side}_sharpa_wave_with_wrist.xml",
                "meshes/elastomer_HB1_TH.STL",
                "meshes/elastomer_HB1_4F.STL",
            ):
                path = f"{base}/{name}"
                target = source / side / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with urllib.request.urlopen(
                        f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_REVISION}/{path}", timeout=60
                    ) as response:
                        payload = response.read()
                    target.write_bytes(payload)
                records.append({"side": side, "path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        result = {"repository": SOURCE_REPO, "revision": SOURCE_REVISION, "files": records}
        (source / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "derive":
        result = derive_spec(source)
        target = root_path(args.spec)
        if target.exists() and json.loads(target.read_text()) != result:
            raise ValueError("Refusing to overwrite a different geometry spec; choose a new --spec")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = prepare_geometry(args.data_root, spec=args.spec, split=args.split, episode_ids=args.episode_ids)
    if args.report:
        target = root_path(args.report)
        if not target.is_relative_to(root_path("experiments")):
            raise ValueError("Execution reports belong under experiments/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            result
            if args.command == "cache"
            else {"command": args.command, "status": "completed", "spec_hash": result.get("hash")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
