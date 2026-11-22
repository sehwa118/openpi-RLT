"""Frozen LingBot-Touch extraction and immutable query-local Assembly caches.

No torch/JAX imports at module import: loader workers only read numpy shards.
The user-selected signed raw_clip30 differs from upstream's positive-Z adapter.
"""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys

from filelock import FileLock
import numpy as np

from openpi.trlt.config import digest
from openpi.trlt.config import root_path

SOURCE_COMMIT = "dec8025b212e7099b235cc2ef13a4921a7e59545"
WEIGHT_SHA256 = "ddb8b435963348fff5b8a83bb48f995978dc8841d82f87be007c775bf5a88055"
FEATURE_FIELDS = ("lingbot_features", "lingbot_valid_mask", "lingbot_timestamps", "lingbot_observation_counts")
DEFAULT_SOURCE = "third_party/lingbot-touch"
DEFAULT_BUNDLE = "checkpoints/lingbot-touch-tiny"
DEFAULT_GEOMETRY = "configs/trlt/lingbot_touch/paxini_v1.json"
DEFAULT_CACHE = "datasets/3dtacdex_assembly/lingbot_touch_cache/paxini_joint_v1"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def read_geometry(path=DEFAULT_GEOMETRY):
    value = read_json(root_path(path))
    if value.get("schema") != "lingbot_paxini_static_geometry_v1":
        raise ValueError("Unsupported PaXini geometry schema")
    if value.get("hash") != digest({k: v for k, v in value.items() if k != "hash"}):
        raise ValueError("PaXini geometry manifest integrity failure")
    positions = np.asarray(value["pos_xyz"], np.float32)
    coverage = np.asarray(value["coverage_mask"], bool)
    sensors = np.asarray(value["sensor_ids"], np.int32)
    if positions.shape != (5, 30, 3) or coverage.shape != (5, 30) or sensors.shape != (5, 30):
        raise ValueError("PaXini geometry requires [5,30,3] positions and [5,30] coverage/sensor IDs")
    expected = np.repeat(np.arange(8).reshape(4, 2), 15, axis=1)
    if not np.isfinite(positions).all() or not coverage[:4].all() or coverage[4].any():
        raise ValueError("PaXini requires four real fingers and an excluded pinky")
    if not np.array_equal(sensors[:4], expected):
        raise ValueError("PaXini pad order must be thumb/index/middle/ring tip15 then pulp15")
    return value, positions, coverage, sensors


def encoder_identity(source=DEFAULT_SOURCE, bundle=DEFAULT_BUNDLE, geometry=DEFAULT_GEOMETRY):
    source, bundle = root_path(source), root_path(bundle)
    if file_sha256(bundle / "encoder.pt") != WEIGHT_SHA256:
        raise ValueError("Unexpected LingBot-Touch-Tiny checkpoint SHA256")
    config = read_json(bundle / "config.json")
    if config["tubelet_size"] != 2 or config["embed_dim"] * len(config["output_layers"]) != 768:
        raise ValueError("Expected Tiny tubelet2 and 768D output")
    read_geometry(geometry)
    # Hash executed source and reference assets, rather than trusting a git label.
    files = sorted(
        [
            *source.glob("tactile_jepa_inference/*.py"),
            *source.glob("data_interfaces/*.py"),
            *source.glob("data_interfaces/data/*.json"),
        ]
    )
    if not files or not (source / "tactile_jepa_inference/model.py").is_file():
        raise FileNotFoundError("Download the pinned LingBot source before extraction/training")
    return {
        "schema": "lingbot_paxini_joint_v1",
        "upstream_commit": SOURCE_COMMIT,
        "adapter_sha256": file_sha256(__file__),
        "precision": "float32",
        "source_files": {str(p.relative_to(source)): file_sha256(p) for p in files},
        "weight_sha256": WEIGHT_SHA256,
        "config_sha256": file_sha256(bundle / "config.json"),
        "geometry_sha256": file_sha256(root_path(geometry)),
        "feature_dim": 768,
        "tubelet_size": 2,
        "modality": "magnetic",
        "input_layout": "four_fingers_tip15_pulp15_missing_pinky_one_joint_encode",
        "normalization": {"mode": "raw_clip", "scale": 30.0, "all_axes_signed": True},
        "padding": "oldest_valid_prefix_then_previous_valid_per_pad_within_query_window",
        "readout": "taxel_mean_per_pad_per_tubelet",
        "time": "mean_unique_real_observation_timestamps_per_tubelet",
    }


def cache_identity(dataset, identity):
    # Visual cache and split are independent of frozen tactile extraction.
    p = {k: v for k, v in dataset.provenance.items() if k not in ("split", "feature_cache_hash")}
    if p.get("tactile_preprocessing") != {"mode": "raw_clip", "scale": 30.0}:
        raise ValueError("PaXini LingBot input requires exact signed raw_clip30")
    if dataset.temporal.tactile_history_frames != 3:
        raise ValueError("PaXini LingBot adapter is pinned to the existing three-row history")
    return {"encoder": identity, "dataset": p}


def prepare_windows(samples, positions, sensor_ids):
    """Prepare Bx4x5x30 input, shared-per-sample spatial coverage, and true counts.

    Fill values only from this query's observed history; masks/counts never treat
    copies as new observations. All source times must be <= query time. The
    noncausal encoder may mix these copies; the padding policy is provenance.
    """
    n = len(samples)
    signals = np.zeros((n, 4, 5, 30, 3), np.float32)
    coverage = np.zeros((n, 5, 30), bool)
    times = np.zeros((n, 2, 8), np.float32)
    counts = np.zeros((n, 2, 8), np.int32)
    for b, sample in enumerate(samples):
        x = np.asarray(sample["tactile"], np.float32)
        valid = np.asarray(sample["tactile_valid_mask"], bool)
        stamps = np.asarray(sample["tactile_timestamps"], np.float32)
        query = float(sample["query_timestamp"])
        if x.shape != (3, 8, 15, 1, 3) or valid.shape != (3, 8) or stamps.shape != (3, 8):
            raise ValueError("Expected native Assembly three-row windows")
        if not np.isfinite(query) or not np.isfinite(x[valid]).all() or not np.isfinite(stamps[valid]).all():
            raise ValueError("Nonfinite real tactile observations")
        if np.any(stamps[valid] > query + 1e-5) or np.any(np.abs(x[valid]) > 1 + 1e-6):
            raise ValueError("Future timestamps or non-raw_clip30 input")
        times[b] = query
        for pad in range(8):
            indices = np.flatnonzero(valid[:, pad])
            if not len(indices):
                continue
            if np.any(np.diff(stamps[indices, pad]) <= 0):
                raise ValueError("Real history timestamps must increase")
            filled = []
            for row in (0, 0, 1, 2):
                past = indices[indices <= row]
                index = past[-1] if len(past) else indices[0]
                filled.append(x[index, pad, :, 0])
            f, region = divmod(pad, 2)
            sl = slice(15 * region, 15 * (region + 1))
            signals[b, :, f, sl] = np.stack(filled)
            coverage[b, f, sl] = True
            for k, rows in enumerate(((0,), (1, 2))):
                real = [r for r in rows if valid[r, pad]]
                counts[b, k, pad] = len(real)
                if real:
                    times[b, k, pad] = stamps[real, pad].mean()
    return signals, coverage, times, counts


class FrozenPaxiniEncoder:
    def __init__(self, *, source=DEFAULT_SOURCE, bundle=DEFAULT_BUNDLE, geometry=DEFAULT_GEOMETRY, device="cpu"):
        self.identity = encoder_identity(source, bundle, geometry)
        _, self.positions, self.coverage, self.sensor_ids = read_geometry(geometry)
        expected_source = root_path(source).resolve()
        loaded = sys.modules.get("tactile_jepa_inference")
        if loaded is not None and Path(loaded.__file__).resolve().parent.parent != expected_source:
            raise ValueError("Already-imported LingBot source differs from pinned extraction source")
        sys.path.insert(0, str(expected_source))
        from tactile_jepa_inference import load_encoder
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.runtime = {
            "torch": torch.__version__,
            "device": str(self.device),
            "precision": "float32",
            "matmul_tf32": False,
        }
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.encoder = load_encoder(root_path(bundle), device=self.device)
        if self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError("LingBot target encoder must stay frozen")

    def __call__(self, samples):
        torch = self.torch
        signals, coverage, stamps, counts = prepare_windows(samples, self.positions, self.sensor_ids)
        features = np.zeros((len(samples), 2, 8, 768), np.float32)
        groups = {}
        for i, mask in enumerate(coverage):
            groups.setdefault(mask.tobytes(), []).append(i)
        with torch.inference_mode():
            for indices in groups.values():
                mask = coverage[indices[0]]
                if not mask.any():
                    continue
                output = self.encoder.encode(
                    torch.from_numpy(signals[indices]).to(self.device),
                    torch.from_numpy(self.positions).to(self.device),
                    torch.from_numpy(mask).to(self.device),
                    modality="magnetic",
                )
                compact_ids = self.sensor_ids[mask]
                latent = output.latent.reshape(len(indices), 2, len(compact_ids), 768)
                for pad in np.unique(compact_ids):
                    select = torch.as_tensor(np.flatnonzero(compact_ids == pad), device=self.device)
                    features[indices, :, pad] = latent.index_select(2, select).mean(2).float().cpu().numpy()
        valid = counts > 0
        features = np.where(valid[..., None], features, 0)
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite frozen tactile features")
        return {
            "lingbot_features": features,
            "lingbot_valid_mask": valid,
            "lingbot_timestamps": stamps,
            "lingbot_observation_counts": counts,
        }


class LingbotCache:
    def __init__(self, root, dataset, *, identity=None, require_complete=True):
        self.root = root_path(root) / dataset.provenance["split"]
        self.manifest = read_json(self.root / "manifest.json")
        expected = dict(self.manifest)
        recorded = expected.pop("hash", None)
        if recorded != digest(expected):
            raise ValueError("LingBot cache manifest integrity failure")
        if identity is not None and digest(self.manifest["identity"]) != digest(cache_identity(dataset, identity)):
            raise ValueError("LingBot cache identity/provenance mismatch")
        if self.manifest["queries_hash"] != digest(dataset.queries):
            raise ValueError("LingBot cache query order/split mismatch")
        if require_complete and (not self.manifest["complete"] or self.manifest["count"] != len(dataset)):
            raise ValueError("Incomplete LingBot cache; finish explicit extraction first")
        self.cache_hash = digest(self.manifest["identity"])
        self.entries = self.manifest["shards"]
        self.ends = np.cumsum([s["count"] for s in self.entries])
        self._cache = OrderedDict()

    def __getitem__(self, index):
        if not 0 <= index < self.manifest["count"]:
            raise IndexError(index)
        i = int(np.searchsorted(self.ends, index, side="right"))
        if i not in self._cache:
            entry = self.entries[i]
            if file_sha256(self.root / entry["file"]) != entry["sha256"]:
                raise ValueError("Corrupted LingBot shard")
            with np.load(self.root / entry["file"], allow_pickle=False) as shard:
                self._cache[i] = {k: shard[k] for k in shard.files}
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(i)
        offset = index - (self.ends[i - 1] if i else 0)
        return {k: v[offset] for k, v in self._cache[i].items()}


class LingbotDataset:
    """Attach immutable features to the existing reconstruction/action sample."""

    def __init__(self, dataset, cache_root, *, identity):
        self.dataset = dataset
        self.cache = LingbotCache(cache_root, dataset, identity=identity)
        self.provenance = dict(
            dataset.provenance,
            lingbot_cache_hash=self.cache.cache_hash,
            lingbot_runtime_hash=digest(self.cache.manifest["runtime"]),
        )
        self.normalization = dataset.normalization

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        cached = self.cache[index]
        if (
            int(cached["segment_id"]) != int(sample["segment_id"])
            or int(cached["query_frame_index"]) != int(sample["query_frame_index"])
            or not np.isclose(cached["query_timestamp"], sample["query_timestamp"], rtol=0, atol=1e-6)
        ):
            raise ValueError("LingBot cache sample identity mismatch")
        features = cached["lingbot_features"]
        valid = cached["lingbot_valid_mask"]
        times = cached["lingbot_timestamps"]
        counts = cached["lingbot_observation_counts"]
        if (
            features.shape != (2, 8, 768)
            or features.dtype != np.float32
            or valid.shape != (2, 8)
            or valid.dtype != bool
            or times.shape != (2, 8)
            or times.dtype != np.float32
            or counts.shape != (2, 8)
            or counts.dtype != np.int32
            or not np.isfinite(features).all()
            or not np.isfinite(times).all()
            or np.any(times > float(sample["query_timestamp"]) + 1e-5)
        ):
            raise ValueError("Invalid LingBot cached tensor shape/dtype/value/time")
        raw_valid = sample["tactile_valid_mask"]
        expected_counts = np.stack((raw_valid[0].astype(np.int32), raw_valid[1:].sum(axis=0, dtype=np.int32)))
        if not np.array_equal(counts, expected_counts) or not np.array_equal(valid, counts > 0):
            raise ValueError("LingBot cached observation mask/count mismatch")
        return dict(sample, **{k: cached[k] for k in FEATURE_FIELDS})


def extract_cache(dataset, root, encoder, *, batch_size=32, shard_size=256, max_samples=None, progress=True):
    """Resume completed immutable shards; no next stage or test access is implicit."""
    from tqdm.auto import tqdm

    if batch_size < 1 or shard_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("Positive batch/shard/sample sizes required")
    output = root_path(root) / dataset.provenance["split"]
    output.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(dataset, encoder.identity)
    with FileLock(str(output / "extraction.lock")):
        path = output / "manifest.json"
        if path.exists():
            old = LingbotCache(root, dataset, identity=encoder.identity, require_complete=False)
            manifest = dict(old.manifest)
            if manifest.get("runtime") != getattr(encoder, "runtime", {"device": "fixture"}):
                raise ValueError("Extraction runtime differs from committed cache; use a separate cache root")
            # Verify existing data before resuming or claiming completion.
            for shard in manifest["shards"]:
                if file_sha256(output / shard["file"]) != shard["sha256"]:
                    raise ValueError("Corrupted existing LingBot shard")
        else:
            manifest = {
                "schema": "lingbot_tactile_cache_v1",
                "identity": identity,
                "queries_hash": digest(dataset.queries),
                "count": 0,
                "complete": False,
                "shards": [],
                "runtime": getattr(encoder, "runtime", {"device": "fixture"}),
            }
        limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        start = manifest["count"]
        bar = tqdm(
            total=limit, initial=min(start, limit), desc=f"LingBot {dataset.provenance['split']}", disable=not progress
        )
        try:
            while start < limit:
                stop = min(start + shard_size, limit)
                values = {k: [] for k in (*FEATURE_FIELDS, "segment_id", "query_frame_index", "query_timestamp")}
                for offset in range(start, stop, batch_size):
                    samples = [dataset[i] for i in range(offset, min(offset + batch_size, stop))]
                    encoded = encoder(samples)
                    for key in FEATURE_FIELDS:
                        values[key].append(encoded[key])
                    for key in ("segment_id", "query_frame_index", "query_timestamp"):
                        values[key].append(np.stack([row[key] for row in samples]))
                    bar.update(len(samples))
                arrays = {k: np.concatenate(v) for k, v in values.items()}
                name = f"chunk_{len(manifest['shards']):06d}.npz"
                temp = output / (name + ".tmp")
                with temp.open("wb") as stream:
                    np.savez(stream, **arrays)
                os.replace(temp, output / name)
                manifest["shards"].append({"file": name, "count": stop - start, "sha256": file_sha256(output / name)})
                manifest["count"] = stop
                manifest["complete"] = stop == len(dataset)
                manifest.pop("hash", None)
                manifest["hash"] = digest(manifest)
                write_json(path, manifest)
                start = stop
        finally:
            bar.close()
        return manifest
