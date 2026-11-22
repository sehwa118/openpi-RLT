"""Reviewable Assembly commands and a CPU checkpoint seal before test access."""

from datetime import UTC
from datetime import datetime
import hashlib
import json

from openpi.trlt.cache import write_json
from openpi.trlt.config import MODELS
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_data import DEFAULT_PROTOCOL
from openpi.trlt.tacdex_data import read_manifest
from openpi.trlt.tacdex_data import seal


def sha256(path):
    result = hashlib.sha256()
    with root_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seal_selections(
    checkpoint_root, output, *, protocol_dir=DEFAULT_PROTOCOL, seed=0, base_selection_seal=None, replace_models=None
):
    """Seal all four selections, optionally replacing explicit models for a follow-up.

    Unchanged models retain the exact validated records from the completed base
    experiment. A follow-up still seals its new validation selections before test.
    """
    protocol = read_manifest(root_path(protocol_dir) / "protocol.json")
    records = {}
    modes = MODELS
    lineage = {}
    if (base_selection_seal is None) != (replace_models is None):
        raise ValueError("base_selection_seal and replace_models must be supplied together")
    if base_selection_seal is not None:
        modes = tuple(replace_models)
        if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODELS):
            raise ValueError("Replacement models must be nonempty, unique and supported")
        base = validate_selection_seal(base_selection_seal, protocol_dir=protocol_dir)
        if base["seed"] != seed:
            raise ValueError("Base selection seed mismatch")
        records = {mode: dict(stages) for mode, stages in base["selections"].items()}
        lineage = {"base_selection_seal_hash": base["hash"], "replaced_models": list(modes)}
    for mode in modes:
        records[mode] = {}
        for stage in ("representation", "probe"):
            directory = root_path(checkpoint_root) / mode / f"seed{seed}" / stage / "best"
            metadata_path = directory / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            if metadata["stage"] != stage or metadata["seed"] != seed:
                raise ValueError("Selected checkpoint stage/seed mismatch")
            for key in ("split_hash", "normalization_hash", "query_manifest_hash"):
                if metadata["provenance"].get(key) != protocol[key]:
                    raise ValueError(f"Selection protocol mismatch: {key}")
            if stage == "representation":
                if metadata["model_config"]["mode"] != mode:
                    raise ValueError("Selection model mismatch")
            elif (
                metadata["provenance"]["representation_checkpoint_hash"]
                != records[mode]["representation"]["parameter_hash"]
            ):
                raise ValueError("Probe was trained with another representation")
            params_path = directory / "params.msgpack"
            if sha256(params_path) != metadata["params_file_hash"]:
                raise ValueError("Selection parameter file hash mismatch")
            records[mode][stage] = {
                "directory": str(directory),
                "metadata_hash": sha256(metadata_path),
                "params_file_hash": metadata["params_file_hash"],
                "parameter_hash": metadata["parameter_hash"],
                "step": metadata["step"],
            }
    result = seal(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "protocol_hash": protocol["hash"],
            "protocol_dir": str(root_path(protocol_dir)),
            "selections": records,
            **lineage,
        }
    )
    destination = root_path(output)
    if destination.exists():
        previous = read_manifest(destination)
        if (
            previous["selections"] != records
            or previous["protocol_hash"] != protocol["hash"]
            or any(previous.get(key) != value for key, value in lineage.items())
        ):
            raise ValueError("Existing seal differs; never replace selections after test access")
        return previous
    write_json(destination, result)
    return result


def validate_selection_seal(path, *, protocol_dir=None, checkpoint=None):
    if not path:
        raise ValueError("Assembly test access requires --selection-seal after all four models are selected")
    value = read_manifest(path)
    if value.get("schema") == "lingbot_single_selection_v1":
        return validate_lingbot_selection_seal(value, protocol_dir=protocol_dir, checkpoint=checkpoint)
    protocol = read_manifest(root_path(protocol_dir or value["protocol_dir"]) / "protocol.json")
    if value["protocol_hash"] != protocol["hash"] or set(value["selections"]) != set(MODELS):
        raise ValueError("Selection seal protocol/models mismatch")
    allowed = set()
    for stages in value["selections"].values():
        for record in stages.values():
            directory = root_path(record["directory"])
            if (
                sha256(directory / "metadata.json") != record["metadata_hash"]
                or sha256(directory / "params.msgpack") != record["params_file_hash"]
            ):
                raise ValueError("Sealed checkpoint changed")
            allowed.add(str(directory.resolve()))
    if checkpoint is not None and str(root_path(checkpoint).resolve()) not in allowed:
        raise ValueError("Checkpoint is not one of the sealed selections")
    return value


def write_plan(run_id, *, device="0"):
    """Write immutable config snapshots and ordered argv arrays; execute nothing."""
    if not run_id or "/" in run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    output = root_path(f"experiments/trlt/3dtacdex_assembly/plans/{run_id}")
    if output.exists():
        raise FileExistsError(f"Plan already exists: {output}")
    checkpoint_root = f"checkpoints/trlt/3dtacdex_assembly/{run_id}"
    latent_root = f"datasets/3dtacdex_assembly/latents/{run_id}"
    logs = f"experiments/trlt/3dtacdex_assembly/runs/{run_id}"
    selection = f"{logs}/final_selections.json"
    config_paths = {}
    configs = {}
    for mode in MODELS:
        raw = json.loads(root_path(f"configs/trlt/3dtacdex_assembly/{mode}.json").read_text())
        path = output / "configs" / f"{mode}.json"
        write_json(path, raw)
        configs[mode] = raw
        config_paths[mode] = str(path)
    stages = []

    def add(name, argv, phase):
        stages.append({"name": name, "phase": phase, "argv": ["uv", "run", "--no-sync", "python", *argv]})

    first = config_paths[MODELS[0]]
    add("prepare_protocol", ["scripts/prepare_tacdex_experiment.py", "prepare", "--config", first], "cpu_prepare")
    for split in ("train", "validation"):
        add(
            f"cache_{split}",
            ["scripts/cache_trex_features.py", "--config", first, "--split", split, "--device", device, "--extract"],
            "train_validation",
        )
    for mode in MODELS:
        add(
            f"{mode}_gpu_preflight",
            [
                "scripts/tacdex_offline.py",
                "train-representation",
                "--config",
                config_paths[mode],
                "--seed",
                "0",
                "--device",
                device,
                "--wandb",
                "disabled",
                "--max-steps",
                "1",
                "--output-dir",
                f"{checkpoint_root}/preflight/{mode}/seed0/representation",
                "--log-dir",
                f"{logs}/preflight/{mode}/seed0/representation",
            ],
            "gpu_preflight_train_validation",
        )
    evaluations = []
    for mode in MODELS:
        common = ["scripts/tacdex_offline.py"]
        options = ["--config", config_paths[mode], "--seed", "0", "--device", device]
        tracking_args = ["--wandb-group", run_id]
        checkpoints = f"{checkpoint_root}/{mode}/seed0"
        log = f"{logs}/{mode}/seed0"
        latents = f"{latent_root}/{mode}/seed0"
        add(
            f"{mode}_stage_a",
            [
                *common,
                "train-representation",
                *options,
                *tracking_args,
                "--wandb-name",
                f"assembly/{mode}/seed0/stage_a",
                "--output-dir",
                f"{checkpoints}/representation",
                "--log-dir",
                f"{log}/representation",
            ],
            "train_validation",
        )
        for split in ("train", "validation"):
            add(
                f"{mode}_export_{split}",
                [
                    *common,
                    "export-latents",
                    *options,
                    "--split",
                    split,
                    "--representation-checkpoint",
                    f"{checkpoints}/representation/best",
                    "--output-dir",
                    f"{latents}/{split}",
                ],
                "train_validation",
            )
        add(
            f"{mode}_stage_b",
            [
                *common,
                "train-probe",
                *options,
                *tracking_args,
                "--wandb-name",
                f"assembly/{mode}/seed0/stage_b",
                "--train-latents",
                f"{latents}/train",
                "--validation-latents",
                f"{latents}/validation",
                "--output-dir",
                f"{checkpoints}/probe",
                "--log-dir",
                f"{log}/probe",
            ],
            "train_validation",
        )
        evaluations.append((mode, common, options, checkpoints, log, latents))
    add(
        "seal_selections",
        ["scripts/plan_tacdex_experiment.py", "seal", "--checkpoint-root", checkpoint_root, "--output", selection],
        "seal",
    )
    add(
        "cache_test",
        [
            "scripts/cache_trex_features.py",
            "--config",
            first,
            "--split",
            "test",
            "--device",
            device,
            "--extract",
            "--selection-seal",
            selection,
        ],
        "final_test",
    )
    metrics = []
    for mode, common, options, checkpoints, log, latents in evaluations:
        add(
            f"{mode}_export_test",
            [
                *common,
                "export-latents",
                *options,
                "--split",
                "test",
                "--representation-checkpoint",
                f"{checkpoints}/representation/best",
                "--output-dir",
                f"{latents}/test",
                "--selection-seal",
                selection,
            ],
            "final_test",
        )
        add(
            f"{mode}_evaluate",
            [
                *common,
                "evaluate",
                *options,
                "--latents",
                f"{latents}/test",
                "--probe-checkpoint",
                f"{checkpoints}/probe/best",
                "--normalization",
                f"{configs[mode]['protocol_dir']}/normalization.json",
                "--output-dir",
                f"{log}/evaluation",
                "--selection-seal",
                selection,
            ],
            "final_test",
        )
        metrics.append(f"{log}/evaluation/metrics.json")
    add("summarize", ["scripts/trex_offline.py", "summarize", *metrics, "--output", f"{logs}/summary.json"], "report")
    sources = sorted(
        {
            *root_path("src/openpi/trlt").glob("*.py"),
            *root_path("src/openpi/models").glob("trlt*.py"),
            root_path("src/openpi/models/pi0.py"),
            root_path("src/openpi/models/rl_token.py"),
            *(
                root_path(p)
                for p in (
                    "scripts/trex_offline.py",
                    "scripts/tacdex_offline.py",
                    "scripts/cache_trex_features.py",
                    "scripts/prepare_tacdex_experiment.py",
                    "scripts/plan_tacdex_experiment.py",
                )
            ),
        }
    )
    result = seal(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "prepared_not_launched",
            "seeds": [0],
            "wandb": configs[MODELS[0]].get("tracking", {}).get("mode", "disabled"),
            "tracking": configs[MODELS[0]].get("tracking", {}),
            "device_requested_for_future_execution": device,
            "source_sha256": {str(p.relative_to(root_path("."))): sha256(p) for p in sources},
            "config_sha256": {p: sha256(p) for p in config_paths.values()},
            "stages": stages,
            "execution_policy": "Explicit stages only; stop on failure, source/config change, or occupied GPU. No automatic resume.",
        }
    )
    write_json(output / "plan.json", result)
    return result


def seal_lingbot_selections(representation, probe, output, *, protocol_dir=DEFAULT_PROTOCOL, seed=0):
    """Seal the explicitly requested single LingBot variant before final test access."""
    protocol = read_manifest(root_path(protocol_dir) / "protocol.json")
    records = {}
    for stage, selected_dir in (("representation", representation), ("probe", probe)):
        directory = root_path(selected_dir)
        if directory.name != "best":
            raise ValueError("Seal validation-selected best checkpoints only")
        meta = json.loads((directory / "metadata.json").read_text())
        if meta["stage"] != stage or meta["seed"] != seed:
            raise ValueError("LingBot selection stage/seed mismatch")
        for key in ("split_hash", "normalization_hash", "query_manifest_hash"):
            if meta["provenance"].get(key) != protocol[key]:
                raise ValueError(f"LingBot selection protocol mismatch: {key}")
        cfg = meta["model_config"] if stage == "representation" else meta["provenance"]["model_config"]
        if (
            cfg.get("tactile_encoder") != "lingbot_touch"
            or cfg["mode"] != "trlt_self_only"
            or cfg["num_stages"] != 2
            or cfg["num_sensors"] != 8
            or not meta["provenance"].get("lingbot_cache_hash")
        ):
            raise ValueError("Expected the Assembly LingBot self-only2layer experiment")
        if stage == "representation":
            model_config = cfg
        elif (
            cfg != model_config
            or meta["provenance"].get("representation_checkpoint_hash") != records["representation"]["parameter_hash"]
        ):
            raise ValueError("LingBot probe was trained with another representation")
        if sha256(directory / "params.msgpack") != meta["params_file_hash"]:
            raise ValueError("LingBot selection params hash mismatch")
        records[stage] = {
            "directory": str(directory.resolve()),
            "metadata_hash": sha256(directory / "metadata.json"),
            "params_file_hash": meta["params_file_hash"],
            "parameter_hash": meta["parameter_hash"],
            "step": meta["step"],
        }
    result = seal(
        {
            "schema": "lingbot_single_selection_v1",
            "seed": seed,
            "protocol_hash": protocol["hash"],
            "protocol_dir": str(root_path(protocol_dir).resolve()),
            "model_config": model_config,
            "selections": {"trlt_self_only": records},
        }
    )
    destination = root_path(output)
    if destination.exists():
        if read_manifest(destination) != result:
            raise ValueError("Existing LingBot seal differs; never overwrite selections")
    else:
        write_json(destination, result)
    return result


def validate_lingbot_selection_seal(value, *, protocol_dir=None, checkpoint=None):
    protocol = read_manifest(root_path(protocol_dir or value["protocol_dir"]) / "protocol.json")
    if value["protocol_hash"] != protocol["hash"] or set(value["selections"]) != {"trlt_self_only"}:
        raise ValueError("LingBot selection seal protocol/models mismatch")
    records = value["selections"]["trlt_self_only"]
    if set(records) != {"representation", "probe"}:
        raise ValueError("LingBot requires both selected stages before test access")
    for record in records.values():
        directory = root_path(record["directory"])
        if (
            sha256(directory / "metadata.json") != record["metadata_hash"]
            or sha256(directory / "params.msgpack") != record["params_file_hash"]
        ):
            raise ValueError("Sealed LingBot checkpoint changed")
    if checkpoint is not None and str(root_path(checkpoint).resolve()) not in {
        r["directory"] for r in records.values()
    }:
        raise ValueError("Checkpoint is not one of the sealed LingBot selections")
    return value
