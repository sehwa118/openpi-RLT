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
