"""CPU-only checks for worker ordering, resume, masks and numerical failures."""

import json
import os

import jax
import numpy as np
import pytest

from openpi.trlt.config import TrainConfig
from openpi.trlt.loading import OrderedBatchSampler
from openpi.trlt.loading import TrainingCollator
from openpi.trlt.loading import training_loader
from openpi.trlt.runner import OBSERVATION_FIELDS
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import load_checkpoint
from openpi.trlt.runner import train_representation
from openpi.trlt.runner_test import tiny


class EnvironmentDataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 2:
            raise ValueError("worker read failed")
        return index, os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("JAX_PLATFORMS"), jax.config.jax_platforms


def test_worker_cpu_isolation_order_reset_and_error(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    sampler = OrderedBatchSampler(1)
    sampler.indices = [1, 0]
    with training_loader(EnvironmentDataset(), sampler, list, num_workers=2) as loader:
        assert list(loader) == [[(1, "", "cpu", "cpu")], [(0, "", "cpu", "cpu")]]
        sampler.indices = [0]
        assert list(loader) == [[(0, "", "cpu", "cpu")]]
        sampler.indices = [2]
        with pytest.raises(ValueError, match="worker read failed"):
            list(loader)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_prefetch_samples_counts_masks_and_partial_batch():
    dataset = SyntheticDataset(tiny(), size=5)
    sampler = OrderedBatchSampler(3)
    sampler.indices = [4, 2, 0, 3, 1]
    collator = TrainingCollator((*OBSERVATION_FIELDS, "query_frame_index"), 2, probe=False)
    with training_loader(dataset, sampler, collator, num_workers=0) as loader:
        expected = list(loader)
    with training_loader(dataset, sampler, collator, num_workers=2) as loader:
        actual = list(loader)
    assert [batch[3] for batch in actual] == [3, 2]
    assert [part["query_frame_index"].tolist() for batch in actual for part in batch[0]] == [[4, 2], [0], [3, 1]]
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)


def test_worker_training_resume_matches_sync_including_partial_batch(tmp_path):
    config = tiny("trlt_cross_only")
    train = SyntheticDataset(config, size=7)
    val = SyntheticDataset(config, size=1, split="validation")
    cfg = TrainConfig(batch_size=4, microbatch_size=2, max_epochs=2)

    def run(name, steps, workers, *, resume=False):
        return train_representation(
            train,
            val,
            config,
            cfg,
            tmp_path / name,
            max_steps=steps,
            log_dir=tmp_path / (name + "_logs"),
            progress=False,
            resume=resume,
            num_workers=workers,
        )

    full = run("full", 2, 0)
    first = run("resume", 1, 2)
    assert first["manifest"]["cursor"] == 4
    resumed = run("resume", 2, 2, resume=True)
    a, am = load_checkpoint(full["last_checkpoint"])
    b, bm = load_checkpoint(resumed["last_checkpoint"])
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-7, rtol=1e-6)
    assert (am["epoch"], am["cursor"], am["step"]) == (bm["epoch"], bm["cursor"], bm["step"]) == (1, 0, 2)
    events = [json.loads(line) for line in (tmp_path / "resume_logs/train.jsonl").read_text().splitlines()]
    assert [event["examples"] for event in events] == [4, 3]


class NonfiniteDataset(SyntheticDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["visual_features"][:] = np.nan
        return sample


def test_nonfinite_gradient_does_not_save_update(tmp_path):
    cfg = tiny()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_representation(
            NonfiniteDataset(cfg, size=2),
            SyntheticDataset(cfg, split="validation", size=1),
            cfg,
            TrainConfig(batch_size=2, microbatch_size=1, max_epochs=1),
            tmp_path / "checkpoints",
            max_steps=1,
            log_dir=tmp_path / "logs",
            progress=False,
            num_workers=0,
        )
    assert not (tmp_path / "checkpoints/last/metadata.json").exists()
    assert not (tmp_path / "logs/train.jsonl").exists()


class ColdCacheDataset:
    """Two spawned workers request the same uncached tactile episode."""

    def __init__(self, root):
        self.root = root

    def __len__(self):
        return 4

    def __getitem__(self, index):
        from pathlib import Path

        from openpi.trlt.config import SENSORS
        import openpi.trlt.data as data

        def decode(path, *args, **kwargs):
            with (self.root / "decode_calls.txt").open("a") as handle:
                handle.write(Path(path).name + "\n")
            return np.ones((2, 64, 64, 1), np.float32), np.arange(2) / 30

        data.decode_segment = decode
        dataset = object.__new__(data.TrexDataset)
        dataset.root = self.root
        dataset.resolution = 64
        dataset.subset = {"revision": "fixture"}
        dataset.rows = {
            0: {
                "streams": {
                    sensor: {"path": sensor + ".mp4", "from_timestamp": 0.0, "to_timestamp": 2 / 30}
                    for sensor in SENSORS
                }
            }
        }
        images, timestamps = dataset._tactile_episode(0)  # noqa: SLF001
        assert all(value.shape == (2, 64, 64, 1) for value in images.values())
        assert all(len(value) == 2 for value in timestamps.values())
        return index


def test_workers_build_cold_cache_once(tmp_path):
    from openpi.trlt.config import SENSORS

    sampler = OrderedBatchSampler(1)
    sampler.indices = [0, 1, 2, 3]
    with training_loader(ColdCacheDataset(tmp_path), sampler, list, num_workers=2) as loader:
        assert list(loader) == [[0], [1], [2], [3]]
    assert len((tmp_path / "decode_calls.txt").read_text().splitlines()) == len(SENSORS)
