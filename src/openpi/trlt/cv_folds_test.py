import pickle

import numpy as np
import pytest

from openpi.trlt.cv_folds import assign_folds
from openpi.trlt.cv_folds import build_cv_datasets
from openpi.trlt.cv_folds import role_of


class _Source:
    """Minimal LatentDataset stand-in: shards on disk with episode_id/source_group, rows indexable."""

    def __init__(self, root, split, episodes, groups, cache_hash):
        self.root, self.cache_hash, self.normalization = root, cache_hash, {"hash": "n"}
        root.mkdir(parents=True)
        np.savez(root / "chunk.npz", episode_id=np.asarray(episodes), source_group=np.asarray(groups))
        self.manifest = {"shards": [{"file": "chunk.npz", "count": len(episodes)}]}
        self.rows = [{"episode_id": e, "row": i, "split": split} for i, e in enumerate(episodes)]
        self.provenance = {
            "split": split,
            "representation_checkpoint_hash": "r",
            "normalization_hash": "n",
            "split_hash": "s",
            "query_manifest_hash": "q",
        }

    def __getitem__(self, i):
        return self.rows[i]


def _sources(tmp_path, split_b="validation"):
    rng = np.random.default_rng(0)
    train_eps = np.repeat(np.arange(20), rng.integers(3, 7, size=20))
    val_eps = np.repeat(np.arange(100, 106), 4)
    group = lambda e: f"g{e % 3}"  # noqa: E731
    return [
        _Source(tmp_path / "a", "train", train_eps, [group(e) for e in train_eps], "ha"),
        _Source(tmp_path / "b", split_b, val_eps, [group(e) for e in val_eps], "hb"),
    ]


def test_assign_folds_is_deterministic_stratified_and_balanced():
    groups = {e: f"g{e % 4}" for e in range(93)}
    a, b = assign_folds(groups, 5, 0), assign_folds(groups, 5, 0)
    assert a == b
    assert set(a) == set(groups)
    assert assign_folds(groups, 5, 1) != a
    sizes = np.bincount(list(a.values()), minlength=5)
    assert sizes.max() - sizes.min() <= 1
    for g in range(4):
        per = np.bincount([f for e, f in a.items() if e % 4 == g], minlength=5)
        assert per.max() - per.min() <= 1
    with pytest.raises(ValueError, match="k >= 3"):
        assign_folds(groups, 2, 0)


def test_roles_partition_trajectories_without_leakage(tmp_path):
    sources = _sources(tmp_path)
    total = sum(len(s.rows) for s in sources)
    heldout_union = set()
    for outer in range(5):
        ds, spec = build_cv_datasets(sources, k=5, seed=0, outer=outer)
        assert sum(len(d) for d in ds.values()) == total
        eps = {role: {d[i]["episode_id"] for i in range(len(d))} for role, d in ds.items()}
        assert not (eps["train"] & eps["early_stop"])
        assert not (eps["train"] & eps["heldout"])
        assert not (eps["early_stop"] & eps["heldout"])
        assert all(spec["assignment"][str(e)] == outer for e in eps["heldout"])
        assert all(role_of(spec["assignment"][str(e)], outer, 5) == "early_stop" for e in eps["early_stop"])
        heldout_union |= {(d["split"], d["row"]) for d in (ds["heldout"][i] for i in range(len(ds["heldout"])))}
        prov = {role: d.provenance for role, d in ds.items()}
        assert (prov["train"]["split"], prov["early_stop"]["split"], prov["heldout"]["split"]) == (
            "train", "validation", "cv_heldout",
        )  # fmt: skip
        assert len({d.cache_hash for d in ds.values()}) == 3
        assert pickle.loads(pickle.dumps(ds["train"]))[0] == ds["train"][0]  # DataLoader workers
    assert len(heldout_union) == total  # every row is held out exactly once across the K outer folds


def test_refuses_test_split_and_mismatched_sources(tmp_path):
    with pytest.raises(ValueError, match="train and validation"):
        build_cv_datasets(_sources(tmp_path / "x", split_b="test"), k=5, seed=0, outer=0)
    sources = _sources(tmp_path / "y")
    sources[1].provenance["representation_checkpoint_hash"] = "other"
    with pytest.raises(ValueError, match="representation_checkpoint_hash"):
        build_cv_datasets(sources, k=5, seed=0, outer=0)
