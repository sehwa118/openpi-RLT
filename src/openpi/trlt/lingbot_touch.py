"""Frozen LingBot-Touch query-local features. Importing this module never initializes Torch/CUDA.

T-Rex uses source-native identity pixel order on BOTH hands, explicitly an exploratory
correspondence assumption, not a calibrated left/right canonical mapping.
"""

from dataclasses import asdict
from datetime import UTC
from datetime import datetime
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from openpi.trlt.cache import write_json
from openpi.trlt.config import SENSORS
from openpi.trlt.config import digest
from openpi.trlt.config import root_path

SOURCE_COMMIT = "dec8025b212e7099b235cc2ef13a4921a7e59545"
WEIGHT_SHA256 = "ddb8b435963348fff5b8a83bb48f995978dc8841d82f87be007c775bf5a88055"
MAP_SHA256 = "c1c6bc2390d6653984745d0cc559b82a85809d6df1e26a68f18902c129b5d57e"
REFERENCE_SHA256 = "53437b563db58b6033e990e3f977b7c84d91e36e412269302b8e2512fbace9fb"
CONFIG_SHA256 = "233a1ed8ba7b9654d41586f5bdb34b13aa869032a921a8c8916caac217e9bc57"
FIELDS = ("tactile_features", "tactile_feature_valid_mask", "tactile_feature_timestamps")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_upstream(source="third_party/lingbot-touch"):
    source = root_path(source).resolve()
    if not (source / "tactile_jepa_inference/model.py").is_file():
        raise FileNotFoundError(f"Download pinned LingBot source into {source}; see docs/trlt/lingbot_touch.md")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import tactile_jepa_inference

    if Path(tactile_jepa_inference.__file__).resolve().parent != source / "tactile_jepa_inference":
        raise ValueError("A different LingBot source is already imported")
    return source


def bundle_identity(source="third_party/lingbot-touch", bundle="checkpoints/lingbot-touch-tiny"):
    """Content-bind actual source, weights, config and spatial maps, not mutable path names."""
    source, bundle = root_path(source).resolve(), root_path(bundle)
    files = {
        "config.json": (bundle / "config.json", CONFIG_SHA256),
        "encoder.pt": (bundle / "encoder.pt", WEIGHT_SHA256),
        "sampling_maps": (source / "data_interfaces/data/pretraining_sampling_maps_v1.json", MAP_SHA256),
        "reference_geometry": (source / "data_interfaces/data/multires_references_v1.json", REFERENCE_SHA256),
    }
    hashes = {}
    for name, (path, expected) in files.items():
        hashes[name] = sha256(path)
        if hashes[name] != expected:
            raise ValueError(f"Pinned LingBot asset mismatch: {name}")
    hashes["python_source"] = digest(
        [
            (str(p.relative_to(source)), sha256(p))
            for folder in ("tactile_jepa_inference", "data_interfaces")
            for p in sorted((source / folder).glob("*.py"))
        ]
    )
    return {"upstream_commit": SOURCE_COMMIT, "hashes": hashes, "feature_dim": 768, "tubelet_size": 2}


def cache_metadata(dataset, settings):
    source = settings.get("source", "third_party/lingbot-touch")
    bundle = settings.get("bundle", "checkpoints/lingbot-touch-tiny")
    identity = bundle_identity(source, bundle)
    # Stat inventory detects ordinary replacement; this is not a content hash of videos.
    inventory = []
    relative_files = {p["path"] for row in dataset.subset["episodes"] for p in row["streams"].values()}
    relative_files.update(row["data_path"] for row in dataset.subset["episodes"])
    for relative in sorted(relative_files):
        path = dataset.root / "source" / relative
        stat = path.stat()
        inventory.append((relative, stat.st_size, stat.st_mtime_ns))
    return {
        "schema_version": 1,
        "encoder": identity,
        "dataset_kind": "trex_squeeze",
        "adapter_sha256": sha256(__file__),
        "revision": dataset.subset["revision"],
        "subset_hash": dataset.subset["hash"],
        "split_hash": dataset.split_manifest["hash"],
        "query_manifest_hash": dataset.query_manifest["hash"],
        "normalization_hash": dataset.provenance["normalization_hash"],
        "source_inventory_hash": digest(inventory),
        "temporal": asdict(dataset.temporal),
        "sensor_order": list(SENSORS),
        "feature_shape": [5, 10, 768],
        "feature_dtype": "float32",
        "sampling": {
            "layout": "visuotactile_deformation240_canonical_right_v1",
            "points": 96,
            "signal": "native_Y_weighted_sample_div255",
            "left_transform": "identity",
            "right_transform": "identity",
            "calibration": "source_native_assumed_unverified",
        },
        "window_policy": "query_local_nearest_previous; leading_earliest_valid_then_forward_fill; absent_sensor_coverage_false",
        "pooling": "per_tubelet_sensor_valid_taxel_mean; invalid_original_tubelets_excluded; timestamp_mean_of_original_valid_frames",
    }


def _array_digest(values):
    return digest(
        [
            (
                k,
                str(np.asarray(v).dtype),
                np.asarray(v).shape,
                hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
            )
            for k, v in sorted(values.items())
        ]
    )


class TactileFeatureCache:
    """Atomic resumable query chunks; stable identity is shared across train/validation/test."""

    def __init__(self, directory, expected_metadata=None):
        self.root = root_path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.cache_hash = self.manifest["cache_hash"]
        if self.cache_hash != digest(self.manifest["metadata"]):
            raise ValueError("Corrupt tactile feature manifest")
        if expected_metadata is not None and digest(expected_metadata) != self.cache_hash:
            raise ValueError("Stale tactile feature cache: metadata mismatch; choose a new cache directory")

    @classmethod
    def create(cls, directory, metadata):
        directory = root_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "manifest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (directory / "manifest.json").exists():
                write_json(directory / "manifest.json", {"cache_hash": digest(metadata), "metadata": metadata})
        return cls(directory, metadata)

    def path(self, key):
        episode, query = map(int, key)
        if min(episode, query) < 0:
            raise ValueError("Negative source cache key")
        return self.root / f"episode_{episode:06d}" / f"query_{query:06d}.npz"

    def __contains__(self, key):
        return self.path(key).is_file()

    def __getitem__(self, key):
        with np.load(self.path(key), allow_pickle=False) as chunk:
            if str(chunk["cache_hash"]) != self.cache_hash or tuple(chunk["source_key"]) != tuple(key):
                raise ValueError("Tactile chunk identity/key mismatch")
            values = {field: chunk[field] for field in FIELDS}
            self._validate(values)
            if str(chunk["payload_hash"]) != _array_digest(values):
                raise ValueError("Corrupt tactile feature chunk content")
            return values

    def _validate(self, values):
        shape = tuple(self.manifest["metadata"]["feature_shape"])
        feature, valid, stamps = (values[field] for field in FIELDS)
        if (
            feature.shape != shape
            or feature.dtype != np.float32
            or valid.shape != shape[:-1]
            or valid.dtype != np.bool_
            or stamps.shape != valid.shape
            or stamps.dtype != np.float32
            or not np.isfinite(feature).all()
            or not np.isfinite(stamps).all()
        ):
            raise ValueError("Invalid frozen tactile feature arrays")
        if np.any(feature[~valid]) or np.any(stamps[~valid]):
            raise ValueError("Invalid tactile slots must be zero")

    def put(self, key, values):
        self._validate(values)
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                **values,
                cache_hash=self.cache_hash,
                source_key=np.asarray(key, np.int64),
                payload_hash=_array_digest(values),
            )
        temporary.replace(path)

    def require_complete(self, dataset):
        path = self.root / f"{dataset.provenance['split']}_complete.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed tactile cache: {path}; run cache_lingbot_touch_features.py")
        record = json.loads(path.read_text())
        keys = [list(map(int, key)) for key in dataset.queries]
        if record.get("cache_hash") != self.cache_hash or record.get("query_keys_hash") != digest(keys):
            raise ValueError("Tactile completion manifest does not match requested split/query set")
        missing = [key for key in dataset.queries if key not in self]
        if missing:
            raise FileNotFoundError(f"Missing tactile query chunks: {missing[:5]}")


class WithTactileFeatures:
    """Spawn-safe dataset wrapper: no live Torch model or GPU runtime in loader workers."""

    def __init__(self, dataset, directory, settings):
        self.dataset = dataset
        self.cache = TactileFeatureCache(directory, cache_metadata(dataset, settings))
        self.cache.require_complete(dataset)
        self.normalization = dataset.normalization
        self.provenance = dict(dataset.provenance, tactile_feature_cache_hash=self.cache.cache_hash)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        features = self.cache[(int(sample["episode_id"]), int(sample["query_frame_index"]))]
        valid = features["tactile_feature_valid_mask"]
        if np.any(features["tactile_feature_timestamps"][valid] > sample["query_timestamp"] + 1e-6):
            raise ValueError("Future tactile features in causal query cache")
        return dict(sample, **features)


def sampled_map(source="third_party/lingbot-touch"):
    path = root_path(source) / "data_interfaces/data/pretraining_sampling_maps_v1.json"
    if sha256(path) != MAP_SHA256:
        raise ValueError("Sampling map checksum mismatch")
    asset = json.loads(path.read_text())
    entry = asset["layouts"]["visuotactile_deformation240_canonical_right_v1"]["entries"]["96"]
    return np.asarray(entry["indices"], np.int64), np.asarray(entry["weights"], np.float32)


def sample_native_frame(image, finger, indices, weights):
    if image.shape != (240, 240) or image.dtype != np.uint8:
        raise ValueError("Native tactile frame must be240x240 uint8 Y codes")
    return (image.reshape(-1)[indices[finger]].astype(np.float32) * weights[finger]).sum(axis=-1) / np.float32(255)


class TrexSampledInput:
    """Decode once per episode/sensor directly into96 scalar points; never save full240 maps."""

    def __init__(self, dataset, source="third_party/lingbot-touch"):
        self.dataset = dataset
        self.indices, self.weights = sampled_map(source)

    @lru_cache(maxsize=2)  # noqa: B019
    def episode(self, episode_id):
        import av

        from openpi.trlt.data import _luma

        result = {}
        row = self.dataset.rows[episode_id]
        for sensor_index, sensor in enumerate(SENSORS):
            pointer = row["streams"][sensor]
            identity = {
                "revision": self.dataset.subset["revision"],
                "pointer": pointer,
                "finger": sensor_index % 5,
                "map_sha": MAP_SHA256,
                "decode": "native_Y_sample96_div255_identity_both_hands_checksummed_v2",
            }
            original = self.dataset.root / "source" / pointer["path"]
            stat = original.stat()
            identity["source_stat"] = [stat.st_size, stat.st_mtime_ns]
            directory = self.dataset.root / "lingbot_sampled96" / digest(identity)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "build.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not (directory / "metadata.json").exists():
                    values, times = [], []
                    start, end = pointer["from_timestamp"], pointer["to_timestamp"]
                    with av.open(str(original)) as container:
                        stream = container.streams.video[0]
                        container.seek(
                            max(0, int((start - 0.1) / float(stream.time_base))), stream=stream, backward=True
                        )
                        for frame in container.decode(stream):
                            if frame.pts is None:
                                continue
                            ts = float(frame.pts * stream.time_base)
                            if ts < start - 1e-5:
                                continue
                            if ts >= end - 1e-5:
                                break
                            values.append(
                                sample_native_frame(_luma(frame), sensor_index % 5, self.indices, self.weights)
                            )
                            times.append(ts - start)
                    if not values:
                        raise ValueError("No sampled tactile frames")
                    for name, value in (
                        ("values", np.asarray(values, np.float32)),
                        ("timestamps", np.asarray(times, np.float64)),
                    ):
                        temporary = directory / (name + "." + uuid.uuid4().hex + ".tmp")
                        with temporary.open("wb") as handle:
                            np.save(handle, value, allow_pickle=False)
                        temporary.replace(directory / (name + ".npy"))
                    write_json(
                        directory / "metadata.json",
                        {
                            "identity": identity,
                            "arrays": {name: sha256(directory / (name + ".npy")) for name in ("values", "timestamps")},
                        },
                    )
            stored = json.loads((directory / "metadata.json").read_text())
            if stored.get("identity") != identity:
                raise ValueError("Stale sampled tactile preprocessing cache")
            for name in ("values", "timestamps"):
                if stored.get("arrays", {}).get(name) != sha256(directory / (name + ".npy")):
                    raise ValueError("Corrupt sampled tactile preprocessing arrays")
            result[sensor] = (
                np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False),
                np.load(directory / "timestamps.npy", mmap_mode="r", allow_pickle=False),
            )
        return result

    def window(self, episode_id, query_index):
        episode = self.dataset._episode(episode_id)  # noqa: SLF001
        query_time = float(episode["timestamp"][query_index])
        data = self.episode(episode_id)
        temporal = self.dataset.temporal
        length = temporal.tactile_history_frames
        values = np.zeros((length, 10, 96), np.float32)
        valid = np.zeros((length, 10), bool)
        stamps = np.zeros((length, 10), np.float32)
        tolerance = max(1e-7, 2 * abs(float(np.spacing(np.float32(query_time)))))
        for sensor_index, sensor in enumerate(SENSORS):
            images, times = data[sensor]
            times = np.asarray(times, np.float32).astype(np.float64)
            for history in range(length):
                target = query_time - (length - 1 - history) * temporal.tactile_history_stride_frames / temporal.fps
                if target < -1e-6:
                    continue
                index = int(np.searchsorted(times, target + tolerance, side="right")) - 1
                if (
                    index < 0
                    or target - times[index] > temporal.max_observation_lag_seconds
                    or times[index] > query_time + 1e-7
                ):
                    continue
                values[history, sensor_index] = images[index]
                valid[history, sensor_index] = True
                stamps[history, sensor_index] = times[index]
        return values, valid, stamps, np.float32(query_time)


def fill_history(values, valid):
    """Explicit causal-query padding: earliest valid for prefix, then last observed for gaps.

    Only original valid frames count in output tubelet masks/times. No observation after
    query enters; missing sensors have no coverage and never generate attention tokens.
    """
    filled = values.copy()
    for sensor in range(valid.shape[1]):
        observed = np.flatnonzero(valid[:, sensor])
        if not len(observed):
            continue
        previous = observed[0]
        for index in range(len(valid)):
            if valid[index, sensor]:
                previous = index
            filled[index, sensor] = values[previous, sensor]
    return filled


class FrozenTactileExtractor:
    def __init__(self, *, source="third_party/lingbot-touch", bundle="checkpoints/lingbot-touch-tiny", device="cpu"):
        import_upstream(source)
        from data_interfaces.sampling import load_reference_geometry
        from tactile_jepa_inference import load_encoder
        import torch

        self.identity = bundle_identity(source, bundle)
        self.device = torch.device(device)
        self.encoder = load_encoder(root_path(bundle), device=self.device)
        reference = load_reference_geometry("visuotactile", 96)
        self.positions = reference["pos_xyz"].to(self.device)
        self.coverage = reference["coverage_mask"].to(self.device)

    def encode_windows(self, windows):
        import torch

        result = [
            {
                "tactile_features": np.zeros((5, 10, 768), np.float32),
                "tactile_feature_valid_mask": np.zeros((5, 10), bool),
                "tactile_feature_timestamps": np.zeros((5, 10), np.float32),
            }
            for _ in windows
        ]
        # Batch only identical coverage layouts; upstream coverage is shared across B.
        groups = {}
        for wi, (values, valid, stamps, query_time) in enumerate(windows):
            if values.shape != (10, 10, 96) or valid.shape != (10, 10) or stamps.shape != valid.shape:
                raise ValueError("T-Rex LingBot requires10 history,10 sensors,96 points")
            if (
                not np.isfinite(values).all()
                or not np.isfinite(stamps).all()
                or np.any(stamps[valid] > query_time + 1e-6)
            ):
                raise ValueError("Nonfinite/future tactile window")
            filled = fill_history(values, valid)
            for hand in range(2):
                selection = slice(hand * 5, hand * 5 + 5)
                coverage = np.any(valid[:, selection], axis=0)
                if not coverage.any():
                    continue
                groups.setdefault(tuple(map(bool, coverage)), []).append(
                    (wi, hand, filled[:, selection], valid[:, selection], stamps[:, selection])
                )
        for mask, entries in groups.items():
            scalar = torch.as_tensor(np.stack([e[2] for e in entries]), device=self.device)
            signal = torch.stack((torch.zeros_like(scalar), torch.zeros_like(scalar), scalar), dim=-1)
            coverage = self.coverage & torch.tensor(mask, dtype=torch.bool, device=self.device)[:, None]
            with torch.inference_mode():
                output = self.encoder.encode(signal, self.positions, coverage, modality="visuotactile")
            # Compact order is time-major, then canonical finger, then point.
            for finger in range(5):
                if not mask[finger]:
                    continue
                token_mask = output.address.finger_id == finger
                latent = output.latent[:, token_mask].reshape(len(entries), 5, 96, 768).mean(dim=2).cpu().numpy()
                for bi, (wi, hand, _, valid, stamps) in enumerate(entries):
                    v = valid[:, finger].reshape(5, 2)
                    ts = stamps[:, finger].reshape(5, 2)
                    keep = v.any(axis=1)
                    times = np.sum(np.where(v, ts, 0), axis=1) / np.maximum(v.sum(axis=1), 1)
                    si = hand * 5 + finger
                    result[wi]["tactile_features"][:, si] = np.where(keep[:, None], latent[bi], 0)
                    result[wi]["tactile_feature_valid_mask"][:, si] = keep
                    result[wi]["tactile_feature_timestamps"][:, si] = np.where(keep, times, 0)
        return result


def extract_trex_tactile_cache(dataset, settings, *, device="cpu", batch_size=4, max_samples=None, plan_only=False):
    """Explicit split extraction only; no training or next-stage execution."""
    if batch_size <= 0 or (max_samples is not None and max_samples <= 0):
        raise ValueError("batch_size/max_samples must be positive")
    metadata = cache_metadata(dataset, settings)
    destination = root_path(settings["cache"])
    keys = dataset.queries[:max_samples] if max_samples else dataset.queries
    estimate = {
        "queries": len(keys),
        "shape_per_query": [5, 10, 768],
        "estimated_bytes": len(keys) * (5 * 10 * 768 * 4 + 5 * 10 * 5),
        "cache_hash": digest(metadata),
        "plan_only": plan_only,
        "device": device,
    }
    if plan_only:
        return estimate
    cache = TactileFeatureCache.create(destination, metadata)
    # An advisory lock prevents two extractors duplicating or overwriting split work.
    with (destination / "extraction.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extractor = FrozenTactileExtractor(
            source=settings.get("source", "third_party/lingbot-touch"),
            bundle=settings.get("bundle", "checkpoints/lingbot-touch-tiny"),
            device=device,
        )
        sampled = TrexSampledInput(dataset, settings.get("source", "third_party/lingbot-touch"))
        started = time.monotonic()
        pending, written, reused = [], 0, 0
        from tqdm.auto import tqdm

        progress = tqdm(total=len(keys), desc=f"LingBot {dataset.provenance['split']}")
        for key in keys:
            if key in cache:
                cache[key]  # Validate before trusting a resumed chunk.
                reused += 1
                progress.update(1)
                continue
            pending.append((key, sampled.window(*key)))
            if len(pending) >= batch_size:
                for (source_key, _), value in zip(
                    pending, extractor.encode_windows([w for _, w in pending]), strict=True
                ):
                    cache.put(source_key, value)
                    written += 1
                progress.update(len(pending))
                pending.clear()
        if pending:
            for (source_key, _), value in zip(pending, extractor.encode_windows([w for _, w in pending]), strict=True):
                cache.put(source_key, value)
                written += 1
            progress.update(len(pending))
        progress.close()
        report = dict(
            estimate,
            plan_only=False,
            written=written,
            reused=reused,
            elapsed_seconds=time.monotonic() - started,
            complete=max_samples is None or len(keys) == len(dataset.queries),
        )
        if cache_metadata(dataset, settings) != metadata:
            raise ValueError("Source/config changed during tactile extraction; completion was not sealed")
        report["date_utc"] = datetime.now(UTC).isoformat()
        report["frozen"] = all(not p.requires_grad for p in extractor.encoder.parameters())
        report["torch_version"] = __import__("torch").__version__
        if not report["frozen"]:
            raise AssertionError("Frozen tactile encoder parameters became trainable")
        if report["complete"]:
            write_json(
                destination / f"{dataset.provenance['split']}_complete.json",
                {
                    "cache_hash": cache.cache_hash,
                    "query_keys_hash": digest([list(map(int, k)) for k in dataset.queries]),
                    "count": len(keys),
                    "report": report,
                },
            )
        report_dir = root_path(settings.get("reports", "experiments/trlt/lingbot_touch/extraction"))
        write_json(report_dir / f"{dataset.provenance['split']}.json", report)
        return report
