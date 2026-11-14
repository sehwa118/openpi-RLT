"""Verified anchor feature storage and local rolling context (no model downloads)."""

from collections import deque
import json
import os
from pathlib import Path
import shutil

import numpy as np

from openpi.trlt.config import CAMERA_KEYS
from openpi.trlt.config import CAMERAS
from openpi.trlt.config import SCHEMA_VERSION
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import root_path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def checkpoint_identity(path):
    """Local checkpoint identity from metadata content and immutable file inventory.

    Includes size/mtime_ns for data shards, content hashes for metadata. Replacing
    checkpoint files invalidates the identity without reading multi-GB shards.
    """
    path = root_path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    inventory = []
    import hashlib

    for file in files:
        stat = file.stat()
        entry = [str(file.relative_to(path)) if path.is_dir() else file.name, stat.st_size, stat.st_mtime_ns]
        if stat.st_size < 1024 * 1024:
            entry.append(hashlib.sha256(file.read_bytes()).hexdigest())
        inventory.append(entry)
    return {"path": str(path), "inventory_hash": digest(inventory), "files": len(files)}


class FeatureCache:
    """One NPZ chunk per source episode/anchor; read on demand, never full RAM load."""

    def __init__(self, root, expected_metadata=None):
        self.root = root_path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["cache_hash"] != digest(self.manifest["metadata"]):
            raise ValueError("Corrupt feature-cache metadata hash")
        if expected_metadata is not None and digest(expected_metadata) != self.manifest["cache_hash"]:
            raise ValueError("Stale feature cache: metadata mismatch; use an explicit rebuild")
        self.cache_hash = self.manifest["cache_hash"]

    @classmethod
    def create(cls, root, metadata, *, force=False):
        root = root_path(root)
        if (root / "manifest.json").exists():
            if not force:
                return cls(root, metadata)
            # Only remove chunks owned by this cache, never arbitrary parent directories.
            for folder in root.glob("episode_*"):
                if folder.is_dir():
                    shutil.rmtree(folder)
        root.mkdir(parents=True, exist_ok=True)
        write_json(
            root / "manifest.json",
            {"schema_version": SCHEMA_VERSION, "metadata": metadata, "cache_hash": digest(metadata)},
        )
        return cls(root, metadata)

    def _path(self, key):
        episode, anchor = map(int, key)
        if episode < 0 or anchor < 0:
            raise ValueError("Cache keys must be nonnegative source indices")
        return self.root / f"episode_{episode:06d}" / f"anchor_{anchor:06d}.npz"

    def __contains__(self, key):
        return self._path(key).is_file()

    def __getitem__(self, key):
        with np.load(self._path(key), allow_pickle=False) as data:
            if str(data["cache_hash"]) != self.cache_hash:
                raise ValueError("Stale anchor chunk")
            features, mask = data["visual_features"], data["visual_valid_mask"]
            if features.ndim != 2 or mask.shape != features.shape[:1] or mask.dtype != np.bool_:
                raise ValueError("Invalid anchor feature/mask shape")
            result = {"visual_features": features.astype(np.float32), "visual_valid_mask": mask}
            for field in ("visual_timestamp", "camera_timestamps"):
                if field in data:
                    result[field] = data[field]
            return result

    def put(
        self, key, visual_features, visual_valid_mask, *, caption_hash, visual_timestamp=None, camera_timestamps=None
    ):
        features = np.asarray(visual_features)
        mask = np.asarray(visual_valid_mask, dtype=bool)
        if features.ndim != 2 or features.shape[:1] != mask.shape or not np.isfinite(features).all():
            raise ValueError("Nonfinite or malformed features")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        extra = {}
        if visual_timestamp is not None:
            extra["visual_timestamp"] = np.asarray(visual_timestamp, np.float32)
        if camera_timestamps is not None:
            extra["camera_timestamps"] = np.asarray(camera_timestamps, np.float32)
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                **extra,
                visual_features=features,
                visual_valid_mask=mask,
                cache_hash=np.array(self.cache_hash),
                caption_hash=np.array(caption_hash),
            )
        os.replace(temporary, path)

    def keys(self):
        return [
            (int(p.parent.name.split("_")[1]), int(p.stem.split("_")[1]))
            for p in sorted(self.root.glob("episode_*/anchor_*.npz"))
        ]


def feature_metadata(
    *,
    revision,
    episode_manifest_hash,
    backbone_checkpoint,
    caption_hash,
    tokenizer_hash,
    feature_dtype="float32",
    temporal=None,
):
    temporal = temporal or TemporalConfig()
    return {
        "revision": revision,
        "episode_manifest_hash": episode_manifest_hash,
        "backbone": checkpoint_identity(backbone_checkpoint),
        "camera_order": CAMERAS,
        "camera_mapping": dict(zip(CAMERAS, CAMERA_KEYS, strict=True)),
        "preprocessing": {
            "resize": "openpi.resize_with_pad",
            "resolution": [224, 224],
            "scale": "uint8 / 255 * 2 - 1",
            "augmentation": False,
        },
        "caption_hash": caption_hash,
        "tokenizer_hash": tokenizer_hash,
        "discrete_state_input": False,
        "jax_default_matmul_precision": "highest",
        "feature_dtype": feature_dtype,
        "features": "final-layer PaliGemma prefix image positions",
        "refresh_frames": temporal.visual_refresh_frames,
        "anchor_sampling": {"fps": temporal.fps, "max_observation_lag_seconds": temporal.max_observation_lag_seconds},
    }


class CachedContext:
    """Serving-side state separate from model parameters; reset at episode boundary.

    update_visual replaces all three cameras as a single anchor. append_tactile
    accepts observed frames only. encode is a callback to the pure tensor model;
    no VLM or action expert runs on tactile updates.
    """

    def __init__(self, temporal=None):
        self.temporal = temporal or TemporalConfig()
        self.reset()

    def reset(self, episode_id=None):
        self.episode_id = episode_id
        self.visual = None
        self.history = deque(maxlen=self.temporal.tactile_history_frames * self.temporal.tactile_history_stride_frames)

    def update_visual(self, context, *, episode_id, timestamp, camera_order=CAMERAS):
        if tuple(camera_order) != CAMERAS:
            raise ValueError("Visual context must contain all three cameras in canonical order")
        if episode_id != self.episode_id:
            self.reset(episode_id)
        self.visual = {**context, "visual_timestamp": float(timestamp)}

    def append_tactile(self, image, valid_mask, *, episode_id, timestamp):
        if episode_id != self.episode_id:
            self.reset(episode_id)
        if self.history and timestamp <= self.history[-1][0]:
            raise ValueError("Tactile observations must be strictly chronological")
        self.history.append((float(timestamp), np.asarray(image), np.asarray(valid_mask, dtype=bool)))

    def encode(self, encode_fn, *, current_state, query_time):
        if self.visual is None or self.visual["visual_timestamp"] > query_time:
            raise ValueError("Missing causal visual context")
        available = [entry for entry in self.history if entry[0] <= query_time]
        if not available:
            raise ValueError("No tactile shape available; supply invalid observed frame to establish shape")
        template = available[-1][1]
        length = self.temporal.tactile_history_frames
        stride = self.temporal.tactile_history_stride_frames
        tactile = np.zeros((length, *template.shape), np.float32)
        mask = np.zeros((length, template.shape[0]), bool)
        stamps = np.full((length, template.shape[0]), query_time, np.float32)
        for slot in range(length):
            desired = query_time - (length - 1 - slot) * stride / self.temporal.fps
            previous = [entry for entry in available if entry[0] <= desired + 1e-7]
            if previous and desired - previous[-1][0] <= self.temporal.max_observation_lag_seconds:
                stamp, value, valid = previous[-1]
                tactile[slot] = np.where(valid[:, None, None, None], value, 0)
                mask[slot] = valid
                stamps[slot] = stamp
        batch = {key: np.asarray(self.visual[key])[None] for key in ("visual_features", "visual_valid_mask")}
        batch.update(
            tactile=tactile[None],
            tactile_valid_mask=mask[None],
            tactile_timestamps=stamps[None],
            proprioception=np.asarray(current_state, np.float32)[None],
            query_timestamp=np.array([query_time], np.float32),
            visual_timestamp=np.array([self.visual["visual_timestamp"]], np.float32),
        )
        return encode_fn(batch)
