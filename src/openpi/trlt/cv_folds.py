"""Trajectory-grouped K-fold cross-validation over the Assembly train+validation frozen caches.

Used by `scripts/run_tactile_gru_cv.py` (docs/trlt/assembly_tactile_gru_cv_plan.md). Never reads the test split:
the pooled sources must have provenance split "train" or "validation". Folds are whole source trajectories
(`episode_id`), stratified by `source_group`. For outer fold k the held-out fold is k, the early-stopping fold is
(k + 1) mod K, and the remaining K - 2 folds train. Rows are referenced in place (no cache copy or rewrite).
"""

from __future__ import annotations

import numpy as np

from openpi.trlt.config import digest

ROLES = ("train", "early_stop", "heldout")
ROLE_SPLIT = {"train": "train", "early_stop": "validation", "heldout": "cv_heldout"}


def _latents(source):
    return getattr(source, "latents", source)  # TouchLatentDataset wraps a LatentDataset


def row_metadata(source) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (episode_id, source_group) of a LatentDataset (or TouchLatentDataset), read from the shards."""
    latents = _latents(source)
    episodes, groups = [], []
    for shard in latents.manifest["shards"]:
        with np.load(latents.root / shard["file"], allow_pickle=False) as values:
            episodes.append(values["episode_id"].astype(np.int64))
            groups.append(values["source_group"].astype(str))
    return np.concatenate(episodes), np.concatenate(groups)


def assign_folds(episode_group: dict[int, str], k: int, seed: int) -> dict[int, int]:
    """Stratified round-robin assignment of whole trajectories to k folds (deterministic in seed)."""
    if k < 3:
        raise ValueError("Need k >= 3 (train, early-stop and held-out folds)")
    rng = np.random.default_rng(seed)
    folds, counter = {}, 0
    for group in sorted(set(episode_group.values())):
        members = sorted(e for e, g in episode_group.items() if g == group)
        for i in rng.permutation(len(members)):
            folds[members[i]] = counter % k
            counter += 1
    return folds


def role_of(fold: int, outer: int, k: int) -> str:
    if fold == outer:
        return "heldout"
    if fold == (outer + 1) % k:
        return "early_stop"
    return "train"


class PooledSubset:
    """Rows of several frozen sources, addressed in place; provenance marks the CV role."""

    def __init__(self, sources, index, provenance, cache_hash, normalization):
        self.sources, self.index = sources, index
        self.provenance, self.cache_hash, self.normalization = provenance, cache_hash, normalization

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, r = self.index[i]
        return self.sources[s][int(r)]


def build_cv_datasets(sources, *, k: int, seed: int, outer: int) -> tuple[dict[str, PooledSubset], dict]:
    """Return {role: PooledSubset} for one outer fold plus the (hashable) fold specification."""
    splits = [getattr(s, "provenance", {}).get("split") for s in sources]
    if sorted(splits) != ["train", "validation"]:
        raise ValueError(f"CV pools exactly the train and validation caches, got splits {splits}")
    base = dict(sources[splits.index("train")].provenance)
    for s in sources:
        other = s.provenance
        for key in ("representation_checkpoint_hash", "normalization_hash", "split_hash", "query_manifest_hash"):
            if other.get(key) != base.get(key):
                raise ValueError(f"CV sources disagree on {key}")
        if other.get("touch_identity") != base.get("touch_identity"):
            raise ValueError("CV sources use different touch caches")
    meta = [row_metadata(s) for s in sources]
    episode_group = {}
    for episodes, groups in meta:
        for e, g in zip(episodes.tolist(), groups.tolist(), strict=True):
            if episode_group.setdefault(e, g) != g:
                raise ValueError(f"Trajectory {e} has conflicting source groups")
    folds = assign_folds(episode_group, k, seed)
    spec = {
        "k": k,
        "seed": seed,
        "outer_fold": outer,
        "early_stop_fold": (outer + 1) % k,
        "sources": [{"split": sp, "latent_cache_hash": s.cache_hash} for sp, s in zip(splits, sources, strict=True)],
        "touch_cache_hashes": [s.provenance.get("touch_cache_hash") for s in sources],
        "assignment": {str(e): f for e, f in sorted(folds.items())},
    }
    spec["hash"] = digest(spec)
    datasets = {}
    for role in ROLES:
        index = [
            (si, r)
            for si, (episodes, _) in enumerate(meta)
            for r in np.flatnonzero([role_of(folds[e], outer, k) == role for e in episodes.tolist()])
        ]
        role_hash = digest({"cv_fold_hash": spec["hash"], "role": role})
        provenance = dict(
            base,
            split=ROLE_SPLIT[role],
            latent_cache_hash=role_hash,
            cv_fold_hash=spec["hash"],
            cv_role=role,
        )
        if "touch_cache_hash" in base:
            provenance["touch_cache_hash"] = digest(spec["touch_cache_hashes"])
        datasets[role] = PooledSubset(sources, index, provenance, role_hash, getattr(sources[0], "normalization", None))
    return datasets, spec
