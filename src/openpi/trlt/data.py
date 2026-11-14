"""Pinned, exact-file T-Rex squeeze preparation and causal offline adapter.

No LeRobot construction is used: missing local files always raise instead of downloading.
Native deformation maps are decoded directly from the Y plane, preserving luma codes.
"""

from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
import fcntl
from functools import lru_cache
import hashlib
from itertools import pairwise
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import polars as pl

from openpi.trlt.config import CAMERA_KEYS
from openpi.trlt.config import CAMERAS
from openpi.trlt.config import SCHEMA_VERSION
from openpi.trlt.config import SENSORS
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import root_path

REPO_ID = "zekaiwang/trex_dataset"
DEFAULT_ROOT = "datasets/trex_squeeze"
DEFAULT_TEMPORAL = TemporalConfig()


def _read(value):
    result = json.loads(root_path(value).read_text()) if isinstance(value, str | Path) else value
    if isinstance(result, dict) and "hash" in result:
        actual = digest({key: item for key, item in result.items() if key != "hash"})
        if result["hash"] != actual:
            raise ValueError("Manifest/statistics self hash mismatch; edited or corrupt metadata cannot reuse caches")
    return result


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True))
        temporary = Path(handle.name)
    temporary.replace(path)


def _seal(value):
    value = dict(value)
    value.pop("hash", None)
    return {**value, "hash": digest(value)}


def build_subset_manifest(data_root=DEFAULT_ROOT, dataset_revision="main"):
    """Resolve once; fetch only info/tasks/episode metadata at the immutable SHA."""
    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download

    root = root_path(data_root)
    api = HfApi()
    remote = api.dataset_info(REPO_ID, revision=dataset_revision, files_metadata=True)
    revision = remote.sha
    inventory = {f.rfilename: {"size": f.size, "blob_id": f.blob_id} for f in remote.siblings}
    metadata = sorted(
        p
        for p in inventory
        if p in ("meta/info.json", "meta/tasks.parquet") or (p.startswith("meta/episodes/") and p.endswith(".parquet"))
    )
    if not metadata or "meta/info.json" not in metadata:
        raise ValueError("Required episode metadata/info is absent")
    revision_path = root / "source_revision.json"
    if revision_path.exists() and _read(revision_path)["revision"] != revision:
        raise ValueError("data_root belongs to another revision; choose a fresh data_root")
    _write(revision_path, {"repo_id": REPO_ID, "revision": revision})
    for path in metadata:
        hf_hub_download(REPO_ID, path, repo_type="dataset", revision=revision, local_dir=root / "source")
    info = _read(root / "source/meta/info.json")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [58]:
            raise ValueError(f"Unexpected {key} shape")
    if info["fps"] != 30:
        raise ValueError("Expected source fps=30; review temporal configuration")
    episode_files = [root / "source" / p for p in metadata if p.startswith("meta/episodes/")]
    # The full-dataset stats columns are intentionally never consumed.
    frames = [pl.read_parquet(p).select(pl.exclude("^stats/.*$")) for p in episode_files]
    all_episodes = pl.concat(frames)
    selected = all_episodes.filter(pl.col("motor_primitive") == "squeeze").sort("episode_index")
    episodes = []
    files = set(metadata)
    for row in selected.to_dicts():
        row["data_path"] = info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        files.add(row["data_path"])
        streams = {}
        for stream in (*CAMERAS, *SENSORS):
            prefix = f"videos/{stream}/"
            path = info["video_path"].format(
                video_key=stream, chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"]
            )
            streams[stream] = {
                "path": path,
                "from_timestamp": row[prefix + "from_timestamp"],
                "to_timestamp": row[prefix + "to_timestamp"],
            }
            files.add(path)
        row["streams"] = streams
        episodes.append(row)
    if len({e["episode_index"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate source episode IDs")
    counts = Counter(e["object"] for e in episodes)
    manifest = _seal(
        {
            "schema_version": SCHEMA_VERSION,
            "repo_id": REPO_ID,
            "revision": revision,
            "total_source_episodes": all_episodes.height,
            "episode_count": len(episodes),
            "frame_count": sum(e["length"] for e in episodes),
            "object_counts": dict(sorted(counts.items())),
            "camera_order": CAMERAS,
            "sensor_order": SENSORS,
            "episodes": episodes,
            "files": [{"path": p, **inventory[p]} for p in sorted(files)],
            "reference_counts_match": all_episodes.height == 5464
            and len(episodes) == 173
            and sum(e["length"] for e in episodes) == 152169,
            "joint_schema": {k: info["features"][k] for k in ("observation.state", "action")},
            "joint_units": "source metadata does not declare physical units; retained source joint units without conversion",
        }
    )
    _write(root / "subset_manifest.json", manifest)
    return manifest


def plan_download(data_root=DEFAULT_ROOT, manifest=None):
    root = root_path(data_root)
    manifest = _read(manifest or root / "subset_manifest.json")
    files = []
    for item in manifest["files"]:
        path = root / "source" / item["path"]
        size = int(item["size"])
        complete = path.is_file() and path.stat().st_size == size
        files.append({**item, "existing": complete, "new_bytes": 0 if complete else size})
    new = sum(f["new_bytes"] for f in files)
    root.mkdir(parents=True, exist_ok=True)
    plan = _seal(
        {
            "revision": manifest["revision"],
            "subset_hash": manifest["hash"],
            "episode_count": manifest["episode_count"],
            "frame_count": manifest["frame_count"],
            "object_counts": manifest["object_counts"],
            "files": files,
            "file_count": len(files),
            "data_file_count": sum(f["path"].startswith("data/") for f in files),
            "video_file_count": sum(f["path"].startswith("videos/") for f in files),
            "total_bytes": sum(f["size"] for f in files),
            "existing_bytes": sum(f["size"] for f in files if f["existing"]),
            "new_bytes": new,
            "free_bytes": shutil.disk_usage(root).free,
            "required_free_bytes": math.ceil(new * 1.05),
            "space_note": "download+5% headroom only; decoded/feature caches planned separately",
        }
    )
    _write(root / "download_plan.json", plan)
    return plan


def download_subset(
    data_root=DEFAULT_ROOT, *, max_download_gb=100.0, workers=6, retry_failed=False, priority_episode_ids=()
):
    from huggingface_hub import hf_hub_download

    root = root_path(data_root)
    manifest = _read(root / "subset_manifest.json")
    plan = plan_download(root, manifest)
    if plan["new_bytes"] > max_download_gb * 1e9:
        raise ValueError(
            f"Download needs {plan['new_bytes'] / 1e9:.3f} GB, exceeds max_download_gb={max_download_gb}; no data downloaded"
        )
    if plan["free_bytes"] < plan["required_free_bytes"]:
        raise ValueError("Insufficient disk space; no data downloaded")
    pending = [f for f in plan["files"] if not f["existing"]]
    failure_path = root / "download_failures.json"
    if retry_failed and failure_path.exists():
        failed = {f["path"] for f in _read(failure_path)["failures"]}
        pending = [f for f in pending if f["path"] in failed]
    priorities = set()
    for row in manifest["episodes"]:
        if row["episode_index"] in priority_episode_ids:
            priorities.add(row["data_path"])
            priorities.update(s["path"] for s in row["streams"].values())
    pending.sort(key=lambda f: (f["path"] not in priorities, f["path"]))
    failures = []

    def fetch(item):
        local = Path(
            hf_hub_download(
                REPO_ID, item["path"], repo_type="dataset", revision=manifest["revision"], local_dir=root / "source"
            )
        )
        if local.stat().st_size != item["size"]:
            raise OSError("Downloaded file size mismatch")
        return item["path"]

    # Complete priority episode files before scheduling the remaining allowlist.
    groups = ([f for f in pending if f["path"] in priorities], [f for f in pending if f["path"] not in priorities])
    for group in groups:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, f): f for f in group}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                except Exception as exc:  # failure messages intentionally omit request/auth details
                    failures.append({"path": item["path"], "error_type": type(exc).__name__})
                _write(failure_path, {"revision": manifest["revision"], "failures": failures})
        if group and priorities:
            _write(root / "priority_ready.json", {"episode_ids": list(priority_episode_ids), "failures": failures})
            print("priority download group complete", list(priority_episode_ids), flush=True)
    complete = plan_download(root, manifest)
    report = {
        "revision": manifest["revision"],
        "subset_hash": manifest["hash"],
        "failures": failures,
        "missing": [f["path"] for f in complete["files"] if not f["existing"]],
        "complete_bytes": complete["existing_bytes"],
        "total_bytes": complete["total_bytes"],
    }
    _write(root / "download_manifest.json", report)
    return report


def make_episode_splits(manifest, split_seed=42, output_path=None):
    """Constrained largest-deficit allocation, deterministic per-object shuffling.

    Start each object/split at one. Add seats minimizing squared deviation from
    object proportional quotas, subject to exact global and per-object totals.
    """
    manifest = _read(manifest)
    episodes = manifest["episodes"]
    objects = sorted({e["object"] for e in episodes})
    grouped = {obj: sorted(e["episode_index"] for e in episodes if e["object"] == obj) for obj in objects}
    if any(len(ids) < 3 for ids in grouped.values()):
        raise ValueError("Each object needs at least three episodes")
    n = len(episodes)
    fractions = np.array([0.7, 0.15, 0.15])
    target = np.floor(fractions * n).astype(int)
    remainder_order = sorted(range(3), key=lambda i: (-(fractions[i] * n - target[i]), i))
    for i in remainder_order[: n - int(target.sum())]:
        target[i] += 1
    if np.any(target < len(objects)):
        raise ValueError("Global targets cannot satisfy per-object minima")
    quota = np.ones((len(objects), 3), dtype=int)
    sizes = np.array([len(grouped[obj]) for obj in objects])
    ideal = sizes[:, None] * fractions
    while int(quota.sum()) < n:
        candidates = []
        for oi, obj in enumerate(objects):
            for si in range(3):
                if quota[oi].sum() < sizes[oi] and quota[:, si].sum() < target[si]:
                    cost = 2 * (quota[oi, si] - ideal[oi, si]) + 1
                    candidates.append((cost, obj, si, oi))
        if not candidates:
            raise RuntimeError("Constrained quota allocation infeasible")
        _, _, si, oi = min(candidates)
        quota[oi, si] += 1
    names = ("train", "validation", "test")
    splits = {name: [] for name in names}
    for oi, obj in enumerate(objects):
        ids = np.array(grouped[obj])
        object_seed = int(hashlib.sha256(f"{split_seed}:{obj}".encode()).hexdigest()[:16], 16)
        np.random.default_rng(object_seed).shuffle(ids)
        start = 0
        for si, name in enumerate(names):
            end = start + quota[oi, si]
            splits[name].extend(int(x) for x in ids[start:end])
            start = end
    for ids in splits.values():
        ids.sort()
    summaries = {}
    for name, ids in splits.items():
        rows = [e for e in episodes if e["episode_index"] in ids]
        lengths = [e["length"] for e in rows]
        summaries[name] = {
            "episodes": len(rows),
            "frames": sum(lengths),
            "objects": dict(sorted(Counter(e["object"] for e in rows).items())),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_mean": float(np.mean(lengths)),
        }
    result = _seal(
        {
            "subset_hash": manifest["hash"],
            "revision": manifest["revision"],
            "split_seed": split_seed,
            "splits": splits,
            "quotas": {obj: dict(zip(names, map(int, quota[oi]), strict=True)) for oi, obj in enumerate(objects)},
            "summary": summaries,
            "algorithm": "One/object/split; greedy minimum squared-quota-error increment under row/global capacities; tie object lexicographic then train/validation/test. SHA256(seed:object) seeded PCG64 shuffle.",
            "limitations": "Source lacks verified collection session/physical-object identity; evaluation is unseen demonstrations of seen categories, not object-held-out.",
        }
    )
    if output_path:
        _write(root_path(output_path), result)
    return result


def build_query_manifest(subset, split_manifest, temporal=DEFAULT_TEMPORAL, output_root=None):
    """Persist a global, split-independent identity for the common query schedule.

    Each row is [source episode ID, query frame, anchor frame, valid horizon count].
    All models and seeds use the same rows; targets never cross an episode edge.
    """
    subset, split = _read(subset), _read(split_manifest)
    _validate_split(subset, split)
    rows = {episode["episode_index"]: episode for episode in subset["episodes"]}
    queries = {}
    for name, ids in split["splits"].items():
        queries[name] = [
            [
                episode_id,
                query,
                temporal.visual_refresh_frames * (query // temporal.visual_refresh_frames),
                min(temporal.action_horizon_frames, rows[episode_id]["length"] - query),
            ]
            for episode_id in sorted(ids)
            for query in range(0, rows[episode_id]["length"], temporal.query_stride_frames)
        ]
    manifest = _seal(
        {
            "schema_version": SCHEMA_VERSION,
            "revision": subset["revision"],
            "subset_hash": subset["hash"],
            "split_hash": split["hash"],
            "temporal": asdict(temporal),
            "timestamp_unit": "episode-local seconds",
            "columns": ["episode_id", "query_frame_index", "anchor_frame_index", "valid_horizon_count"],
            "queries": queries,
            "query_counts": {name: len(values) for name, values in queries.items()},
        }
    )
    if output_root is not None:
        path = root_path(output_root) / (manifest["hash"] + ".json")
        if path.exists():
            if _read(path) != manifest:
                raise ValueError("Stored query manifest does not match its derived contract")
        else:
            _write(path, manifest)
    return manifest


def read_episode(data_root, row):
    path = root_path(data_root) / "source" / row["data_path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing allowlisted file {path}; run explicit download, no automatic download")
    frame = pl.read_parquet(path).filter(pl.col("episode_index") == row["episode_index"]).sort("timestamp")
    if frame.height != row["length"]:
        raise ValueError(f"Episode {row['episode_index']} length mismatch")
    times = frame["timestamp"].to_numpy().astype(np.float64)
    indices = frame["frame_index"].to_numpy()
    if np.any(np.diff(times) <= 0) or not np.array_equal(indices, np.arange(len(times))):
        raise ValueError("Non-monotonic timestamps or non-contiguous frame_index")
    if not np.allclose(np.diff(times), 1 / 30, atol=2e-4):
        raise ValueError("Unexpected frame period")
    result = {"timestamp": times, "frame_index": indices}
    for key in ("observation.state", "action", "observation.tactile_force"):
        if key in frame.columns:
            result[key] = np.asarray(frame[key].to_list(), np.float32)
    for key in ("observation.state", "action"):
        if result[key].shape != (len(times), 58):
            raise ValueError(f"{key} must preserve all 58 source joints")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite source {key} in episode {row['episode_index']}")
    if "index" in frame.columns:
        actual = frame["index"].to_numpy()
        if actual[0] != row["dataset_from_index"] or actual[-1] + 1 != row["dataset_to_index"]:
            raise ValueError("Global dataset pointer does not match episode rows")
    return result


def compute_train_normalization(data_root=DEFAULT_ROOT, split_manifest=None, epsilon=1e-6, output_path=None):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Normalization epsilon must be positive and finite")
    root = root_path(data_root)
    subset = _read(root / "subset_manifest.json")
    split = _read(split_manifest or root / "splits.json")
    _validate_split(subset, split)
    rows = [e for e in subset["episodes"] if e["episode_index"] in split["splits"]["train"]]
    accum = {k: [0, np.zeros(58), np.zeros(58)] for k in ("observation.state", "action")}
    fingerprints = defaultdict(list)
    for row in rows:
        ep = read_episode(root, row)
        fingerprints[hashlib.sha256(ep["observation.state"].tobytes() + ep["action"].tobytes()).hexdigest()].append(
            row["episode_index"]
        )
        for key, (count, total, total_sq) in accum.items():
            x = ep[key].astype(np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite {key} in train episode")
            accum[key] = [count + len(x), total + x.sum(0), total_sq + np.square(x).sum(0)]
    result = {
        "revision": subset["revision"],
        "subset_hash": subset["hash"],
        "split_hash": split["hash"],
        "epsilon": epsilon,
        "unique_train_frames": accum["action"][0],
        "episode_ids": split["splits"]["train"],
        "duplicates_in_train": [v for v in fingerprints.values() if len(v) > 1],
    }
    for src, dst in (("observation.state", "state"), ("action", "action")):
        count, total, total_sq = accum[src]
        mean = total / count
        raw_std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0))
        result[dst] = {
            "mean": mean.tolist(),
            "std": np.maximum(raw_std, epsilon).tolist(),
            "floored_dimensions": np.where(raw_std < epsilon)[0].tolist(),
        }
    result = _seal(result)
    _write(root_path(output_path) if output_path else root / "normalization.json", result)
    return result


def _validate_split(subset, split):
    if split["subset_hash"] != subset["hash"]:
        raise ValueError("Stale split: subset hash mismatch")
    ids = [x for part in split["splits"].values() for x in part]
    expected = {e["episode_index"] for e in subset["episodes"]}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Split overlap or missing/extra episode IDs")


def _luma(frame):
    """Copy native 8-bit Y codes; never swscale limited/full range or convert RGB."""
    if not frame.format.name.startswith(("yuv", "gray")):
        raise ValueError(f"Unsupported deformation pixel format {frame.format.name}")
    plane = frame.planes[0]
    return np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)[: frame.height, : frame.width].copy()


def decode_segment(path, start, end, *, tactile=False, resolution=64):
    """Decode selected episode segment using source timestamps and native Y plane."""
    import av
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected video {path}; explicit download required")
    images, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((start - 0.1) / float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < start - 1e-5:
                continue
            if timestamp >= end - 1e-5:
                break
            if tactile:
                image = _luma(frame)
                if image.shape != (240, 240):
                    raise ValueError(f"Expected native deformation 240x240, got {image.shape}")
                image = (
                    cv2.resize(image.astype(np.float32), (resolution, resolution), interpolation=cv2.INTER_AREA)[
                        ..., None
                    ]
                    / 255.0
                )
            else:
                image = frame.to_ndarray(format="rgb24")
            images.append(image)
            timestamps.append(timestamp - start)
    if not images:
        raise ValueError(f"No decodable frame in selected interval {path.name}:{start}:{end}")
    return np.asarray(images), np.asarray(timestamps, np.float64)


def build_tactile_window(tactile, timestamps, query_time, temporal=DEFAULT_TEMPORAL, *, resolution=64):
    """Causal nearest-previous sampling; streams contain (images,timestamps)."""
    length = temporal.tactile_history_frames
    out = np.zeros((length, len(SENSORS), resolution, resolution, 1), np.float32)
    valid = np.zeros((length, len(SENSORS)), bool)
    observed_times = np.zeros((length, len(SENSORS)), np.float32)
    for hi in range(length):
        target = query_time - (length - 1 - hi) * temporal.tactile_history_stride_frames / temporal.fps
        if target < -1e-6:
            continue
        for si, stream in enumerate(SENSORS):
            if stream not in tactile:
                continue
            # Source Parquet and tensor timestamps are float32. Canonicalize PTS
            # to that same precision; tolerate two ULPs in relative-time subtraction,
            # never anything approaching a frame interval.
            times = np.asarray(timestamps[stream], np.float32).astype(np.float64)
            tolerance = max(1e-7, 2 * abs(float(np.spacing(np.float32(query_time)))))
            index = int(np.searchsorted(times, target + tolerance, side="right")) - 1
            if (
                index < 0
                or target - times[index] > temporal.max_observation_lag_seconds
                or times[index] > query_time + 1e-7
            ):
                continue
            image = tactile[stream][index]
            if not np.isfinite(image).all():
                continue
            out[hi, si] = image
            valid[hi, si] = True
            observed_times[hi, si] = times[index]
    return {"tactile": out, "tactile_valid_mask": valid, "tactile_timestamps": observed_times}


class TrexDataset:
    """Lazy episode-local streaming adapter; trainable CNN features are never cached."""

    def __init__(
        self,
        data_root=DEFAULT_ROOT,
        split_manifest=None,
        normalization=None,
        split="train",
        temporal=DEFAULT_TEMPORAL,
        tactile_resolution=64,
        visual_cache=None,
    ):
        self.root = root_path(data_root)
        self.subset = _read(self.root / "subset_manifest.json")
        self.split_manifest = _read(split_manifest or self.root / "splits.json")
        _validate_split(self.subset, self.split_manifest)
        norm_path = normalization or self.root / "normalization.json"
        self.normalization = _read(norm_path) if isinstance(norm_path, dict) or root_path(norm_path).exists() else None
        if self.normalization is not None and self.normalization["split_hash"] != self.split_manifest["hash"]:
            raise ValueError("Normalization/split mismatch")
        self.temporal = temporal
        self.resolution = tactile_resolution
        self.visual_cache = visual_cache
        if hasattr(visual_cache, "manifest"):
            metadata = visual_cache.manifest["metadata"]
            expected = {
                "revision": self.subset["revision"],
                "episode_manifest_hash": self.subset["hash"],
                "refresh_frames": temporal.visual_refresh_frames,
                "anchor_sampling": {
                    "fps": temporal.fps,
                    "max_observation_lag_seconds": temporal.max_observation_lag_seconds,
                },
            }
            if (
                any(metadata.get(key) != value for key, value in expected.items())
                or tuple(metadata.get("camera_order", ())) != CAMERAS
            ):
                raise ValueError("Visual cache provenance mismatch: revision/subset/camera order/temporal refresh")
        self.episodes = [
            e for e in self.subset["episodes"] if e["episode_index"] in self.split_manifest["splits"][split]
        ]
        self.rows = {e["episode_index"]: e for e in self.episodes}
        self.query_manifest = build_query_manifest(
            self.subset, self.split_manifest, temporal, self.root / "query_manifests"
        )
        self.queries = [(row[0], row[1]) for row in self.query_manifest["queries"][split]]
        visual_metadata = getattr(visual_cache, "manifest", {}).get("metadata", {})
        self.provenance = {
            "revision": self.subset["revision"],
            "split_seed": self.split_manifest.get("split_seed"),
            "file_manifest_hash": digest(self.subset.get("files", [])),
            "backbone_checkpoint": visual_metadata.get("backbone"),
            "visual_preprocessing": visual_metadata.get("preprocessing"),
            "feature_dtype": visual_metadata.get("feature_dtype"),
            "subset_hash": self.subset["hash"],
            "split_hash": self.split_manifest["hash"],
            "normalization_hash": self.normalization["hash"] if self.normalization else None,
            "split": split,
            "episode_manifest_hash": self.subset["hash"],
            "split_manifest_hash": self.split_manifest["hash"],
            "query_manifest_hash": self.query_manifest["hash"],
            "caption_hash": digest([(e["episode_index"], e["caption"]) for e in self.subset["episodes"]]),
            "temporal": asdict(temporal),
            "tactile_resolution": tactile_resolution,
            "camera_order": CAMERAS,
            "sensor_order": SENSORS,
            "feature_cache_hash": getattr(visual_cache, "cache_hash", None),
        }

    def __len__(self):
        return len(self.queries)

    # Keep the small state/action tables across shuffled batches. Two entries
    # caused almost every random sample to reread an entire shared Parquet shard.
    # This bounded cache fits all 173 squeeze episodes; video arrays stay mmap'd
    # separately and are not retained here.
    @lru_cache(maxsize=256)  # noqa: B019 -- bounded dataset lifetime, cleared on reset
    def _episode(self, episode_id):
        return read_episode(self.root, self.rows[episode_id])

    @lru_cache(maxsize=64)  # noqa: B019 -- bounded mmap handles, no full image copies
    def _tactile_episode(self, episode_id):
        row = self.rows[episode_id]
        images, times = {}, {}
        for sensor in SENSORS:
            p = row["streams"][sensor]
            identity = {
                "revision": self.subset["revision"],
                "episode_id": episode_id,
                "stream": sensor,
                "pointer": p,
                "resolution": self.resolution,
                "decode": "native_Y_float32_area_div255_v1",
            }
            cache = self.root / "decoded" / digest(identity)
            values_path, times_path, metadata_path = (
                cache / "values.npy",
                cache / "timestamps.npy",
                cache / "metadata.json",
            )
            if not metadata_path.exists():
                cache.mkdir(parents=True, exist_ok=True)
                with (cache / "build.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not metadata_path.exists():
                        values, stamps = decode_segment(
                            self.root / "source" / p["path"],
                            p["from_timestamp"],
                            p["to_timestamp"],
                            tactile=True,
                            resolution=self.resolution,
                        )
                        # Atomic replace plus metadata last prevents partial arrays being accepted.
                        import uuid

                        for target, value in ((values_path, values), (times_path, stamps)):
                            temporary = cache / (target.name + "." + uuid.uuid4().hex + ".tmp")
                            with temporary.open("wb") as handle:
                                np.save(handle, value, allow_pickle=False)
                            temporary.replace(target)
                        _write(metadata_path, identity)
            if _read(metadata_path) != identity:
                raise ValueError("Stale deformation preprocessing cache")
            images[sensor] = np.load(values_path, mmap_mode="r", allow_pickle=False)
            times[sensor] = np.load(times_path, mmap_mode="r", allow_pickle=False)
        return images, times

    def read_anchor(self, episode_id, anchor_frame_index):
        row = self.rows[episode_id]
        ep = self._episode(episode_id)
        time = ep["timestamp"][anchor_frame_index]
        images = {}
        observed = []
        for camera, key in zip(CAMERAS, CAMERA_KEYS, strict=True):
            p = row["streams"][camera]
            # A short interval includes the previous frame if source PTS rounding differs.
            begin = max(0, time - self.temporal.max_observation_lag_seconds)
            values, times = decode_segment(
                self.root / "source" / p["path"],
                p["from_timestamp"] + begin,
                min(p["to_timestamp"], p["from_timestamp"] + time + 1 / self.temporal.fps + 1e-5),
            )
            times = times + begin
            index = int(np.searchsorted(times, time + 1e-7, side="right")) - 1
            if index < 0 or time - times[index] > self.temporal.max_observation_lag_seconds:
                raise ValueError("Missing causal RGB anchor")
            images[key] = values[index]
            observed.append(float(times[index]))
        return {
            "images": images,
            "caption": row["caption"],
            "visual_timestamp": float(min(observed)),
            "camera_timestamps": observed,
        }

    def __getitem__(self, index):
        if self.normalization is None:
            raise ValueError("Compute train normalization before reading training samples")
        episode_id, query = self.queries[index]
        ep = self._episode(episode_id)
        row = self.rows[episode_id]
        anchor = self.temporal.visual_refresh_frames * (query // self.temporal.visual_refresh_frames)
        query_time = float(ep["timestamp"][query])
        images, times = self._tactile_episode(episode_id)
        result = build_tactile_window(images, times, query_time, self.temporal, resolution=self.resolution)
        state_mean = np.asarray(self.normalization["state"]["mean"], np.float32)
        state_std = np.asarray(self.normalization["state"]["std"], np.float32)
        action_mean = np.asarray(self.normalization["action"]["mean"], np.float32)
        action_std = np.asarray(self.normalization["action"]["std"], np.float32)
        horizon = self.temporal.action_horizon_frames
        action = np.zeros((horizon, 58), np.float32)
        count = min(horizon, len(ep["action"]) - query)
        action[:count] = (ep["action"][query : query + count] - action_mean) / action_std
        result.update(
            {
                "proprioception": (ep["observation.state"][query] - state_mean) / state_std,
                "action_target": action,
                "action_valid_mask": np.arange(horizon) < count,
                "query_timestamp": np.float32(query_time),
                "visual_timestamp": np.float32(ep["timestamp"][anchor]),
                "episode_id": np.int32(episode_id),
                "anchor_frame_index": np.int32(anchor),
                "query_frame_index": np.int32(query),
                "object": row["object"],
                "caption": row["caption"],
            }
        )
        if self.visual_cache is not None:
            context = self.visual_cache[(episode_id, anchor)]
            result.update({k: context[k] for k in ("visual_features", "visual_valid_mask")})
            if "visual_timestamp" in context:
                result["visual_timestamp"] = np.float32(context["visual_timestamp"])
        return result


def validate_dataset(
    data_root=DEFAULT_ROOT, *, split_manifest=None, decode_videos=True, max_episodes=None, workers=4, packet_workers=8
):
    root = root_path(data_root)
    subset = _read(root / "subset_manifest.json")
    split = _read(split_manifest or root / "splits.json")
    _validate_split(subset, split)
    selected = subset["episodes"][:max_episodes]
    errors, fingerprints = [], defaultdict(list)
    episode_split = {eid: name for name, ids in split["splits"].items() for eid in ids}
    luma_report = None

    def check_row(row):
        fingerprint = None
        luma_report = None
        error = None
        try:
            if row["motor_primitive"] != "squeeze":
                raise ValueError("Non-squeeze episode")
            ep = read_episode(root, row)
            fingerprint = hashlib.sha256(ep["observation.state"].tobytes() + ep["action"].tobytes()).hexdigest()
            for stream, pointer in row["streams"].items():
                duration = pointer["to_timestamp"] - pointer["from_timestamp"]
                if not np.isclose(duration, row["length"] / 30, atol=1e-3):
                    raise ValueError(f"Video pointer duration mismatch {stream}")
                if decode_videos:
                    # Check both boundaries, not just MP4 header/open success.
                    for offset in (0, max(0, duration - 1 / 30 - 1e-5)):
                        values, times = decode_segment(
                            root / "source" / pointer["path"],
                            pointer["from_timestamp"] + offset,
                            min(pointer["to_timestamp"], pointer["from_timestamp"] + offset + 1 / 30 + 1e-5),
                            tactile=stream in SENSORS,
                        )
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite decoded video")
                    if stream in SENSORS and luma_report is None:
                        import av

                        with av.open(str(root / "source" / pointer["path"])) as container:
                            frame = next(container.decode(video=0))
                            native = _luma(frame)
                            luma_report = {
                                "pixel_format": frame.format.name,
                                "color_range": str(frame.color_range),
                                "native_shape": list(native.shape),
                                "native_min": int(native.min()),
                                "native_max": int(native.max()),
                                "decode": "direct Y-plane code copy; no range conversion",
                                "resize": "float32 INTER_AREA /255",
                            }
        except Exception as exc:
            error = {"episode_id": row["episode_index"], "error": str(exc)}
        return fingerprint, row["episode_index"], luma_report, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fingerprint, episode_id, luma, error in pool.map(check_row, selected):
            if fingerprint is not None:
                fingerprints[fingerprint].append(episode_id)
            if error is not None:
                errors.append(error)
            if luma_report is None and luma is not None:
                luma_report = luma
    duplicates = [ids for ids in fingerprints.values() if len(ids) > 1]
    cross_split_duplicates = [ids for ids in duplicates if len({episode_split[i] for i in ids}) > 1]
    if cross_split_duplicates:
        errors.append(
            {
                "error": "Exact duplicate trajectories cross splits; regroup before training",
                "episode_ids": cross_split_duplicates,
            }
        )
    # Detect overlap of selected source row and stream intervals (shared files alone are allowed).
    for field in ("data_path", *CAMERAS, *SENSORS):
        groups = defaultdict(list)
        for row in subset["episodes"]:
            if field == "data_path":
                groups[row[field]].append((row["dataset_from_index"], row["dataset_to_index"], row["episode_index"]))
            else:
                p = row["streams"][field]
                groups[p["path"]].append((p["from_timestamp"], p["to_timestamp"], row["episode_index"]))
        for ranges in groups.values():
            ranges.sort()
            for a, b in pairwise(ranges):
                if a[1] > b[0] + 1e-5:
                    errors.append(
                        {"error": f"Overlapping selected source ranges: {field}", "episode_ids": [a[2], b[2]]}
                    )
    result = {
        "status": "pass" if not errors else "fail",
        "revision": subset["revision"],
        "subset_hash": subset["hash"],
        "split_hash": split["hash"],
        "checked_episodes": len(selected),
        "decode_videos": decode_videos,
        "errors": errors,
        "exact_duplicate_trajectories": duplicates,
        "luma_validation": luma_report,
        "limitations": "Session and physical object identity unavailable; near-duplicate trajectories not certified absent.",
    }
    if decode_videos:
        packet_report = validate_video_packet_index(root, episodes=selected, workers=packet_workers)
        result["packet_index_validation"] = {
            key: value for key, value in packet_report.items() if key != "selected_interval_summaries"
        }
        if packet_report["status"] != "pass":
            result["status"] = "fail"
            result["errors"].extend(packet_report["errors"])
    _write(root / "validation.json", result)
    return result


def validate_packet_timestamps(
    packet_timestamps, start, end, expected_count, *, fps=30.0, tolerance=1e-5, index_verified=False
):
    """Validate an MP4 H264 sample/packet PTS index for one selected episode.

    MP4/H264 video samples are access units (one frame per timestamped packet).
    Decode boundary checks separately establish that actual frames are readable.
    """
    timestamps = np.asarray(packet_timestamps, np.float64)
    if timestamps.ndim != 1 or (
        not index_verified and (not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0))
    ):
        raise ValueError("Packet PTS index must be finite, unique and sorted")
    left = int(np.searchsorted(timestamps, start - tolerance, side="left"))
    right = int(np.searchsorted(timestamps, end - tolerance, side="left"))
    selected = timestamps[left:right]
    if len(selected) != expected_count:
        raise ValueError(f"Selected interval frame count {len(selected)} != expected {expected_count}")
    if not len(selected):
        raise ValueError("Empty selected video interval")
    if not np.isclose(selected[0], start, atol=tolerance, rtol=0) or not np.isclose(
        selected[-1], end - 1 / fps, atol=tolerance, rtol=0
    ):
        raise ValueError("Selected video first/last PTS does not align with episode pointers")
    differences = np.diff(selected)
    if not np.allclose(differences, 1 / fps, atol=tolerance, rtol=0):
        raise ValueError("Selected video cadence is not the expected source fps")
    return {
        "packet_frame_count": len(selected),
        "first_timestamp": float(selected[0]),
        "last_timestamp": float(selected[-1]),
        "maximum_cadence_error_seconds": float(np.max(np.abs(differences - 1 / fps))) if len(differences) else 0.0,
    }


def _check_packet_file(item):
    """Top-level fork worker: AV demux and NumPy only; never imports a model."""
    import av

    actual_path, path, intervals = item
    summaries, errors = [], []
    started = time.monotonic()
    scanned = 0
    try:
        with av.open(actual_path) as container:
            stream = container.streams.video[0]
            if stream.codec_context.name != "h264":
                raise ValueError("Packet/frame correspondence is validated only for source H264 MP4")
            time_base = float(stream.time_base)
            # Temporary source-file index is discarded before returning. No
            # non-selected frame/feature/PTS cache is written.
            pts = np.fromiter(
                (packet.pts for packet in container.demux(stream) if packet.pts is not None), dtype=np.int64
            ).astype(np.float64)
            pts *= time_base
        scanned = len(pts)
        pts.sort()
        if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
            raise ValueError("Packet PTS index must be finite, unique and sorted")
        for episode_id, sensor, pointer, count in intervals:
            try:
                summary = validate_packet_timestamps(
                    pts, pointer["from_timestamp"], pointer["to_timestamp"], count, index_verified=True
                )
                summaries.append(
                    {"episode_id": episode_id, "stream": sensor, "file": path, "expected_count": count, **summary}
                )
            except ValueError as exc:
                errors.append({"episode_id": episode_id, "stream": sensor, "file": path, "error": str(exc)})
    except Exception as exc:
        errors.append({"file": path, "error": str(exc)})
    return summaries, errors, {"file": path, "packets_scanned": scanned, "seconds": time.monotonic() - started}


def validate_video_packet_index(data_root=DEFAULT_ROOT, *, episodes=None, workers=8):
    """CPU-only preparation: demux unique files once in at most eight fork workers.

    Run in a dedicated host preparation process before initializing JAX/CUDA.
    Only selected-interval summaries and aggregate file profiling are persisted.
    """
    root = root_path(data_root)
    subset = _read(root / "subset_manifest.json")
    selected = subset["episodes"] if episodes is None else episodes
    grouped = defaultdict(list)
    for episode in selected:
        for stream, pointer in episode["streams"].items():
            grouped[pointer["path"]].append((episode["episode_index"], stream, pointer, episode["length"]))
    if workers < 1:
        raise ValueError("packet workers must be positive")
    workers = min(workers, 8)
    summaries, errors, profiles = [], [], []
    started = time.monotonic()
    tasks = [(str(root / "source" / path), path, intervals) for path, intervals in sorted(grouped.items())]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for records, failures, profile in pool.map(_check_packet_file, tasks):
            summaries.extend(records)
            errors.extend(failures)
            profiles.append(profile)
    result = {
        "status": "pass" if not errors else "fail",
        "revision": subset["revision"],
        "subset_hash": subset["hash"],
        "unique_files_checked": len(grouped),
        "selected_intervals_checked": len(summaries),
        "selected_packet_frame_count": sum(row["packet_frame_count"] for row in summaries),
        "method": "MP4 H264 access-unit PTS, sorted for B-frame reordering; each file demuxed once in CPU fork workers; boundary frames separately decoded",
        "packet_workers": workers,
        "wall_seconds": time.monotonic() - started,
        "source_packets_scanned": sum(row["packets_scanned"] for row in profiles),
        "file_profiles": profiles,
        "errors": errors,
        "selected_interval_summaries": summaries,
    }
    _write(root / "packet_index_validation.json", result)
    return result
