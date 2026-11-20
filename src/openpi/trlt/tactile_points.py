"""Compact 240x240->P raw tactile sampling and immutable episode caches for LingBot-Touch."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.trlt.cache import write_json
from openpi.trlt.config import SCHEMA_VERSION
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import root_path

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
SUPPORTED_VISUOTACTILE_POINTS = (200, 96, 32)
VISUOTACTILE_LAYOUT_ID = "visuotactile_deformation240_canonical_right_v1"
LINGBOT_MAP_SHA256 = "c1c6bc2390d6653984745d0cc559b82a85809d6df1e26a68f18902c129b5d57e"
LINGBOT_REFERENCE_SHA256 = "53437b563db58b6033e990e3f977b7c84d91e36e412269302b8e2512fbace9fb"
NATIVE_DEFORM_SHAPE = (240, 240)
SUPPORTED_CACHE_DTYPES = ("float16", "uint8")


@dataclass(frozen=True)
class SamplingMap:
    """Verified 240x240 -> P spatial sampling map and canonical fingertip geometry."""

    num_points: int
    indices: np.ndarray
    weights: np.ndarray
    pos_xyz: np.ndarray
    coverage_mask: np.ndarray
    map_sha256: str
    reference_sha256: str

    def __post_init__(self):
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        if self.indices.ndim != 3 or self.indices.shape[:2] != (5, self.num_points):
            raise ValueError("indices must have shape [5, P, K]")
        if self.weights.shape != self.indices.shape:
            raise ValueError("weights must match indices shape [5, P, K]")
        if self.pos_xyz.shape != (5, self.num_points, 3):
            raise ValueError("pos_xyz must have shape [5, P, 3]")
        if self.coverage_mask.shape != (5, self.num_points) or self.coverage_mask.dtype != bool:
            raise ValueError("coverage_mask must have boolean shape [5, P]")


def load_lingbot_visuotactile_map(lingbot_repo_dir: str | Path, *, num_points: int = 32) -> SamplingMap:
    """Load and checksum-verify official LingBot-Touch 240x240 sampling map and reference geometry."""
    if num_points not in SUPPORTED_VISUOTACTILE_POINTS:
        raise ValueError(f"Supported visuotactile point counts are {SUPPORTED_VISUOTACTILE_POINTS}")
    base = root_path(lingbot_repo_dir) / "data_interfaces" / "data"
    map_bytes = (base / "pretraining_sampling_maps_v1.json").read_bytes()
    ref_bytes = (base / "multires_references_v1.json").read_bytes()
    map_sha = hashlib.sha256(map_bytes).hexdigest()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()
    if map_sha != LINGBOT_MAP_SHA256:
        raise ValueError("Native sampling map asset checksum mismatch")
    if ref_sha != LINGBOT_REFERENCE_SHA256:
        raise ValueError("Reference geometry asset checksum mismatch")

    map_asset = json.loads(map_bytes)
    ref_asset = json.loads(ref_bytes)
    if (
        map_asset.get("schema") != "pretraining_sampling_maps_v1"
        or tuple(map_asset.get("finger_order", ())) != FINGER_ORDER
        or map_asset.get("canonical_right") is not True
    ):
        raise ValueError("Sampling map schema or hand convention mismatch")
    if (
        ref_asset.get("schema") != "multires_references_v1"
        or tuple(ref_asset.get("finger_order", ())) != FINGER_ORDER
        or ref_asset.get("canonical_right") is not True
    ):
        raise ValueError("Reference geometry schema or hand convention mismatch")

    source = map_asset["layouts"][VISUOTACTILE_LAYOUT_ID]
    entry = source["entries"][str(num_points)]
    ref_entry = ref_asset["entries"][f"visuotactile_{num_points:03d}"]
    return SamplingMap(
        num_points=num_points,
        indices=np.asarray(entry["indices"], dtype=np.int64),
        weights=np.asarray(entry["weights"], dtype=np.float32),
        pos_xyz=np.asarray(ref_entry["pos_xyz"], dtype=np.float32),
        coverage_mask=np.asarray(ref_entry["coverage_mask"], dtype=bool),
        map_sha256=map_sha,
        reference_sha256=ref_sha,
    )


def make_synthetic_sampling_map(num_points: int = 32) -> SamplingMap:
    """Create a deterministic grid-sampled canonical map for local unit tests only."""
    if num_points <= 0 or num_points > 240 * 240:
        raise ValueError("num_points must be in [1, 240*240]")
    cols = max(1, round(np.sqrt(num_points)))
    rows = (num_points + cols - 1) // cols
    ys = np.linspace(48, 192, rows, dtype=np.int64)
    xs = np.linspace(48, 192, cols, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    flat = (yy.reshape(-1) * 240 + xx.reshape(-1))[:num_points]
    indices = np.broadcast_to(flat[None, :, None], (5, num_points, 1)).copy()
    weights = np.ones((5, num_points, 1), dtype=np.float32)

    ny = ((yy.reshape(-1) - 120.0) / 120.0)[:num_points]
    nx = ((xx.reshape(-1) - 120.0) / 120.0)[:num_points]
    nz = np.sqrt(np.maximum(1.0 - 0.25 * (nx * nx + ny * ny), 0.1))
    unit = np.stack((nx, ny, nz), axis=-1).astype(np.float32)
    pos_xyz = np.broadcast_to(unit[None], (5, num_points, 3)).copy()
    coverage = np.ones((5, num_points), dtype=bool)
    return SamplingMap(
        num_points=num_points,
        indices=indices,
        weights=weights,
        pos_xyz=pos_xyz,
        coverage_mask=coverage,
        map_sha256=digest({"synthetic_indices": indices.tolist()}),
        reference_sha256=digest({"synthetic_pos_xyz": unit.tolist()}),
    )


def sample_deformation_240(
    raw_frames: np.ndarray, sampling_map: SamplingMap, *, flip_horizontal: bool = False
) -> np.ndarray:
    """Sample native 240x240 raw luma frames [..., 5, 240, 240] to raw scalar points [..., 5, P]."""
    values = np.asarray(raw_frames)
    if np.iscomplexobj(values) or values.dtype == bool or values.ndim < 3 or values.shape[-3:] != (5, *NATIVE_DEFORM_SHAPE):
        raise ValueError("raw_frames must be real numeric [..., 5, 240, 240]")
    values = values.astype(np.float32)
    if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 255.0:
        raise ValueError("raw_frames must be finite raw luma codes in [0, 255]")
    if flip_horizontal:
        values = values[..., ::-1]
    flattened = values.reshape(*values.shape[:-2], 240 * 240)
    fingers = np.arange(5)[:, None, None]
    prefix = (slice(None),) * (flattened.ndim - 2)
    gathered = flattened[(*prefix, fingers, sampling_map.indices)]
    return (gathered * sampling_map.weights).sum(axis=-1).astype(np.float32)


def prepare_raw_to_lingbot_signal(
    raw_points: np.ndarray, coverage_mask: np.ndarray | None = None
) -> np.ndarray:
    """Convert raw visuotactile scalar [..., 5, P] in [0, 255] to LingBot-Touch [..., 5, P, 3]."""
    values = np.asarray(raw_points, dtype=np.float32)
    if values.ndim < 2 or values.shape[-2] != 5 or not np.isfinite(values).all():
        raise ValueError("raw_points must be finite floating [..., 5, P]")
    if coverage_mask is not None:
        mask = np.asarray(coverage_mask, dtype=bool)
        if mask.shape != values.shape[-2:]:
            raise ValueError("coverage_mask must match trailing [5, P] shape")
        values = np.where(mask, values, 0.0)
    z_norm = np.clip(values / 255.0, 0.0, 1.0)
    signal = np.zeros((*values.shape, 3), dtype=np.float32)
    signal[..., 2] = z_norm
    return signal


class TactilePointCache:
    """Immutable episode-level cache storing sampled tactile points [L_ep, 10, P]."""

    def __init__(self, root: str | Path, expected_metadata: dict | None = None):
        self.root = root_path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["cache_hash"] != digest(self.manifest["metadata"]):
            raise ValueError("Corrupt tactile point cache manifest hash")
        if expected_metadata is not None and digest(expected_metadata) != self.manifest["cache_hash"]:
            raise ValueError("Stale tactile point cache: metadata mismatch; explicit rebuild required")
        self.cache_hash = self.manifest["cache_hash"]
        self.metadata = self.manifest["metadata"]
        dtype_name = self.metadata.get("dtype", "float16")
        if dtype_name not in SUPPORTED_CACHE_DTYPES:
            raise ValueError(f"Unsupported tactile cache dtype: {dtype_name}")
        self.storage_dtype = np.float16 if dtype_name == "float16" else np.uint8

    @classmethod
    def create(cls, root: str | Path, metadata: dict, *, force: bool = False) -> "TactilePointCache":
        root_dir = root_path(root)
        dtype_name = metadata.get("dtype", "float16")
        if dtype_name not in SUPPORTED_CACHE_DTYPES:
            raise ValueError(f"Unsupported tactile cache dtype: {dtype_name}")
        if (root_dir / "manifest.json").exists():
            if not force:
                return cls(root_dir, metadata)
            for path in root_dir.glob("episode_*.npz"):
                path.unlink()
            shutil.rmtree(root_dir / "tmp", ignore_errors=True)
        root_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            root_dir / "manifest.json",
            {"schema_version": SCHEMA_VERSION, "metadata": metadata, "cache_hash": digest(metadata)},
        )
        return cls(root_dir, metadata)

    def _path(self, episode_id: int) -> Path:
        episode = int(episode_id)
        if episode < 0:
            raise ValueError("episode_id must be nonnegative")
        return self.root / f"episode_{episode:06d}.npz"

    def __contains__(self, episode_id: int) -> bool:
        return self._path(episode_id).is_file()

    def put(self, episode_id: int, points: np.ndarray, timestamps: np.ndarray, valid_mask: np.ndarray) -> None:
        values = np.asarray(points, dtype=np.float32)
        times = np.asarray(timestamps, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        if values.ndim != 3 or values.shape[1] != 10 or not np.isfinite(values).all():
            raise ValueError("points must be finite [L_ep, 10, P]")
        if times.shape != values.shape[:2] or not np.isfinite(times).all():
            raise ValueError("timestamps must be finite [L_ep, 10]")
        if valid.shape != values.shape[:2]:
            raise ValueError("valid_mask must have boolean shape [L_ep, 10]")
        if float(values.min()) < 0.0 or float(values.max()) > 255.0:
            raise ValueError("points must lie in raw [0, 255] range")
        if self.storage_dtype == np.uint8:
            stored = np.rint(values).clip(0, 255).astype(np.uint8)
        else:
            stored = values.astype(np.float16)

        path = self._path(episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez(
                stream,
                points=stored,
                timestamps=times,
                valid_mask=valid,
                cache_hash=np.array(self.cache_hash),
            )
        tmp.replace(path)

    def load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        with np.load(self._path(episode_id), allow_pickle=False) as data:
            if str(data["cache_hash"]) != self.cache_hash:
                raise ValueError("Stale tactile point episode chunk")
            points = data["points"].astype(np.float32)
            timestamps = data["timestamps"].astype(np.float64)
            valid_mask = data["valid_mask"].astype(bool)
            if points.ndim != 3 or points.shape[1] != 10 or not np.isfinite(points).all():
                raise ValueError("Invalid tactile point chunk shape or nonfinite values")
            return {"points": points, "timestamps": timestamps, "valid_mask": valid_mask}

    def window(
        self,
        episode_id: int,
        query_time: float,
        temporal: TemporalConfig | None = None,
        *,
        tubelet_size: int = 2,
    ) -> dict[str, np.ndarray]:
        """Extract causal nearest-previous tactile window matching openpi.trlt.data timing."""
        temporal = temporal or TemporalConfig()
        length = temporal.tactile_history_frames
        if tubelet_size <= 0 or length % tubelet_size:
            raise ValueError("tactile_history_frames must be divisible by tubelet_size")

        episode = self.load_episode(episode_id)
        num_points = episode["points"].shape[2]
        out = np.zeros((length, 10, num_points), dtype=np.float32)
        valid = np.zeros((length, 10), dtype=bool)
        observed_times = np.zeros((length, 10), dtype=np.float32)

        for si in range(10):
            sensor_mask = episode["valid_mask"][:, si]
            if not sensor_mask.any():
                continue
            times = episode["timestamps"][sensor_mask, si].astype(np.float32).astype(np.float64)
            pts = episode["points"][sensor_mask, si]
            tolerance = max(1e-7, 2 * abs(float(np.spacing(np.float32(query_time)))))
            for hi in range(length):
                target = query_time - (length - 1 - hi) * temporal.tactile_history_stride_frames / temporal.fps
                if target < -1e-6:
                    continue
                idx = int(np.searchsorted(times, target + tolerance, side="right")) - 1
                if idx < 0 or target - times[idx] > temporal.max_observation_lag_seconds + tolerance:
                    continue
                out[hi, si] = pts[idx]
                valid[hi, si] = True
                observed_times[hi, si] = np.float32(times[idx])

        tubelet_valid = valid.reshape((length // tubelet_size, tubelet_size, 10)).any(axis=1)
        return {
            "left_raw": out[:, :5],
            "right_raw": out[:, 5:],
            "tactile_valid_mask": valid,
            "tubelet_valid_mask": tubelet_valid,
            "tactile_timestamps": observed_times,
        }
