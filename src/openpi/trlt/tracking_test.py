"""CPU-only tracking lifecycle and trainer integration checks."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpi.trlt.config import TrainConfig
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import train_representation
from openpi.trlt.runner_test import tiny
from openpi.trlt.tracking import TrainingTracker
from openpi.trlt.tracking import grouped_metrics


def test_tracking_resume_and_failure_cleanup(tmp_path, monkeypatch):
    run = Mock(id="abc", project="trlt", entity="team", url="https://example.invalid/run")
    init = Mock(return_value=run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    options = {"mode": "online"}
    with TrainingTracker(tmp_path, {}, options=options) as tracker:
        tracker.log("train", {"step": 1, "metrics": {"tactile_per_sensor": [2.0, 3.0]}})
    assert run.log.call_args.args[0]["stage_a/train/tactile_per_sensor/1"] == 3.0
    with pytest.raises(RuntimeError), TrainingTracker(tmp_path, {}, options=options, resume=True):
        raise RuntimeError("training failed")
    assert init.call_args.kwargs["id"] == "abc"
    assert init.call_args.kwargs["resume"] == "allow"
    run.finish.assert_called_with(exit_code=1)


def test_cpu_training_with_offline_wandb(tmp_path):
    config = tiny()
    logs = tmp_path / "logs"
    result = train_representation(
        SyntheticDataset(config, size=2),
        SyntheticDataset(config, split="validation", size=1),
        config,
        TrainConfig(batch_size=2, microbatch_size=1, max_epochs=1),
        tmp_path / "checkpoints",
        max_steps=1,
        log_dir=logs,
        tracking={"mode": "offline", "project": "trlt-tracking-test"},
        progress=True,
    )
    assert result["manifest"]["actual_train_steps"] == 1
    assert json.loads((logs / "validation.jsonl").read_text())["step"] == 1
    assert (logs / "wandb_run.json").exists()
    assert list((logs / "wandb").glob("offline-run-*/*.wandb"))


@pytest.mark.parametrize(
    ("stage", "group", "metric"),
    [
        ("representation", "stage_a", "visual_loss"),
        ("probe", "stage_b", "action_loss"),
    ],
)
def test_grouping_excludes_mask_diagnostics(stage, group, metric):
    event = {
        "step": 4,
        "epoch": 1,
        "examples": 32,
        "data_seconds": 0.2,
        "update_seconds": 1.0,
        "loss": 2.0,
        "value": 1.5,
        "best_value": 1.2,
        "improved": False,
        "metrics": {
            metric: 2.0,
            "visual_count": 8,
            "tactile_sensor_count": [1, 2],
            "tactile_valid_fraction": 0.5,
            "action_count": 6,
        },
    }
    for phase in ("train", "validation"):
        logged = grouped_metrics(event, phase, stage)
        assert logged[f"{group}/{phase}/{metric}"] == 2.0
        assert logged[f"progress/{phase}/epoch"] == 1
        assert logged[f"timing/{phase}/data_seconds"] == 0.2
        assert logged["optimizer_step"] == 4
        assert not any("count" in key or "valid_fraction" in key for key in logged)
        assert all(key.split("/")[0] in {group, "progress", "timing", "optimizer_step"} for key in logged)
    assert "visual_count" in event["metrics"]  # local JSONL input is untouched


@pytest.mark.parametrize("stage", ["representation", "probe"])
def test_ten_step_train_interval_keeps_validation_and_resume_steps(tmp_path, monkeypatch, stage):
    run = Mock(id="cadence", project="trlt", entity=None, url="offline")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=Mock(return_value=run)))
    with TrainingTracker(tmp_path, {"stage": stage}, options={"mode": "offline", "log_every_steps": 10}) as tracker:
        for step in range(1, 26):
            tracker.log("train", {"step": step, "loss": 1.0})
        tracker.log("validation", {"step": 25, "value": 0.5})
    logged = [call.args[0] for call in run.log.call_args_list]
    assert [row["optimizer_step"] for row in logged] == [10, 20, 25]
    assert ("stage_b" if stage == "probe" else "stage_a") + "/validation/value" in logged[-1]
    run.log.reset_mock()
    with TrainingTracker(
        tmp_path, {"stage": stage}, options={"mode": "offline", "log_every_steps": 10}, resume=True
    ) as tracker:
        for step in range(26, 32):
            tracker.log("train", {"step": step, "loss": 1.0})
    assert [call.args[0]["optimizer_step"] for call in run.log.call_args_list] == [30]


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_logging_interval_rejected(tmp_path, interval):
    with pytest.raises(ValueError, match="positive integer"):
        TrainingTracker(tmp_path, {}, options={"log_every_steps": interval})


def read_offline_records(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    reader = datastore.DataStore()
    reader.open_for_scan(str(path))
    config, history = {}, []
    try:
        while (data := reader.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind in ("run", "config"):
                stored_config = record.run.config if kind == "run" else record.config
                for item in stored_config.update:
                    config[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
            elif kind == "history":
                history.append(
                    {item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item}
                )
    finally:
        reader.close()
    return config, history


@pytest.mark.parametrize("stage", ["representation", "probe"])
def test_real_offline_sdk_assembly_config_and_ten_step_metrics(tmp_path, stage):
    from openpi.trlt.config import root_path
    from openpi.trlt.tacdex_source import PAD_NAMES

    raw = json.loads(root_path("configs/trlt/3dtacdex_assembly/trlt_alternating.json").read_text())
    options = dict(raw["tracking"], mode="offline", project="assembly-tracking-validation", entity=None)
    manifest = {
        "stage": stage,
        "dataset": {
            "kind": raw["dataset_kind"],
            "state_dim": 22,
            "action_dim": 22,
            "camera_order": ["main", "secondary"],
            "sensor_order": list(PAD_NAMES),
            "temporal": raw["temporal"],
        },
        "representation_model_config": raw["model"],
        "train_config": raw["probe"] if stage == "probe" else raw["representation"],
        "tracking": options,
    }
    with TrainingTracker(tmp_path, manifest, options=options) as tracker:
        for step in range(1, 26):
            tracker.log(
                "train",
                {
                    "step": step,
                    "loss": 1.0,
                    "metrics": {"action_loss": 0.1}
                    if stage == "probe"
                    else {
                        "visual_loss": 0.2,
                        "tactile_per_sensor": [0.3] * 8,
                        "tactile_per_history": [0.4] * 3,
                        "visual_count": 1,
                    },
                },
            )
        tracker.log("validation", {"step": 25, "value": 0.5})
    path = next((tmp_path / "wandb").glob("offline-run-*/*.wandb"))
    config, history = read_offline_records(path)
    assert config["dataset"]["kind"] == "3dtacdex_assembly"
    assert config["dataset"]["action_dim"] == 22
    assert len(config["dataset"]["sensor_order"]) == 8
    assert config["representation_model_config"]["num_cameras"] == 2
    assert config["representation_model_config"]["use_modality_embedding"] is False
    if stage == "representation":
        assert config["train_config"]["batch_size"] == config["train_config"]["microbatch_size"] == 64
        assert config["train_config"]["max_steps"] == 8000
    assert [row["optimizer_step"] for row in history] == [10, 20, 25]
    if stage == "representation":
        keys = history[0].keys()
        assert len([k for k in keys if "tactile_per_sensor/" in k]) == 8
        assert len([k for k in keys if "tactile_per_history/" in k]) == 3
        assert not any("count" in k for k in keys)
