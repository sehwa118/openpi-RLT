"""CPU-compatible checks for step budgets and optional early stopping."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from openpi.trlt.config import MODELS
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TrainConfig
from openpi.trlt.config import root_path
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import load_checkpoint
from openpi.trlt.runner import train_action_probe
from openpi.trlt.runner import train_representation
from openpi.trlt.runner_test import tiny


@pytest.mark.parametrize(
    ("epochs", "patience", "override", "expected"),
    [(None, None, None, 7), (None, 1, None, 4), (1, None, None, 2), (None, None, 3, 3)],
)
def test_step_budget_and_plateau(tmp_path, epochs, patience, override, expected):
    model = tiny()
    # A zero learning rate gives exactly constant validation loss: stopping must
    # depend on the requested policy, not accidental improvements in the fixture.
    training = TrainConfig(
        learning_rate=0, batch_size=2, microbatch_size=1, max_epochs=epochs, patience=patience, max_steps=7
    )
    result = train_representation(
        SyntheticDataset(model, size=3),
        SyntheticDataset(model, size=1, split="validation"),
        model,
        training,
        tmp_path / "checkpoints",
        max_steps=override,
        log_dir=tmp_path / "logs",
        num_workers=0,
        progress=False,
    )
    manifest = result["manifest"]
    assert manifest["actual_train_steps"] == expected
    assert manifest["max_steps"] == (override or 7)
    _, metadata = load_checkpoint(result["last_checkpoint"])
    assert metadata["step"] == expected
    events = [json.loads(line) for line in (tmp_path / "logs/train.jsonl").read_text().splitlines()]
    assert [event["examples"] for event in events] == [2, 1, 2, 1, 2, 1, 2][:expected]
    if epochs is None and patience is None and override is None:
        assert (manifest["completed_epochs"], manifest["cursor"]) == (3, 2)
        validations = [json.loads(line) for line in (tmp_path / "logs/validation.jsonl").read_text().splitlines()]
        assert [event["step"] for event in validations] == [2, 4, 6, 7]
        assert [event["best"] for event in validations] == [True, False, False, False]
        _, best = load_checkpoint(result["best_checkpoint"])
        assert best["step"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_epochs": None}, "requires max_steps or max_epochs"),
        ({"max_steps": 0}, "max_steps must be positive"),
        ({"max_epochs": 0}, "max_epochs must be positive"),
        ({"patience": 0}, "patience must be positive or None"),
    ],
)
def test_invalid_stopping_policy(tmp_path, kwargs, message):
    model = tiny()
    with pytest.raises(ValueError, match=message):
        train_representation(
            SyntheticDataset(model, size=1),
            SyntheticDataset(model, size=1, split="validation"),
            model,
            TrainConfig(**kwargs),
            tmp_path,
            num_workers=0,
        )


@pytest.mark.parametrize("mode", MODELS)
def test_experiment_configs_keep_probe_defaults(mode):
    raw = json.loads(root_path(f"configs/trlt/{mode}.json").read_text())
    config = TrainConfig(**raw["representation"])
    assert (config.max_steps, config.max_epochs, config.patience) == (15_000, None, None)
    assert raw["probe"] == asdict(ProbeConfig())


def test_probe_keeps_epoch_budget_and_early_stopping(tmp_path):
    class FrozenSyntheticDataset(SyntheticDataset):
        def __init__(self, split):
            super().__init__(tiny(), size=1, split=split)
            self.provenance["frozen"] = True
            self.cache_hash = split

        def __getitem__(self, index):
            return dict(super().__getitem__(index), z=np.zeros((1, 8), np.float32), visual_age_seconds=np.float32(0))

    result = train_action_probe(
        FrozenSyntheticDataset("train"),
        FrozenSyntheticDataset("validation"),
        ProbeConfig(hidden_dim=8, learning_rate=0),
        tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        num_workers=0,
        progress=False,
    )
    manifest = result["manifest"]
    # One best epoch, followed by the default ten epochs without improvement.
    assert manifest["actual_train_steps"] == manifest["completed_epochs"] == 11
    assert manifest["max_steps"] is None
    assert not manifest["stopped_by_max_steps"]
    assert manifest["train_config"]["max_epochs"] == 100
    assert manifest["train_config"]["patience"] == 10
