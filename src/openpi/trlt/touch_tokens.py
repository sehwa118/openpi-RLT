"""Frozen per-finger tactile token caches for Stage B, joined onto frozen latent rows.

Sources (all CPU-computable):
  lingbot  Pretrained LingBot-Touch tokens from a converted Flax bundle (experiment H1).
  random   Same encoder with random-init parameters (H2: architecture control).
  raw      Normalized PaXini taxel vectors, one token per taxel with T*3 features (H3: no encoder).

Cache layout (one directory per split):
  tokens.npy [N, 4, K, C] float16, finger order thumb,index,middle,ring (LingBot order)
  valid.npy  [N, 4, K] bool      token usable (pad valid at the query frame)
  keys.npz   segment_id [N] int32, query_frame_index [N] int64, query_timestamp [N] float32
  pos_xyz.npy [4, K, 3] float32  canonical coordinates of every token
  manifest.json                  identity, provenance and file hashes

`TouchLatentDataset` reorders fingers to the probe's body-part finger order and exposes
touch_tokens/touch_valid_mask/touch_pos_xyz next to the LatentDataset fields.
"""

from collections.abc import Callable, Iterable
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from openpi.trlt.cache import write_json
from openpi.trlt.config import digest
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_lingbot import PaxiniLingBotConfig
from openpi.trlt.tacdex_lingbot import canonical_geometry
from openpi.trlt.tacdex_lingbot import pad_channels
from openpi.trlt.tacdex_lingbot import prepare_signal

TOUCH_SOURCES = ("lingbot", "random", "raw")
CACHE_FINGER_ORDER = ("thumb", "index", "middle", "ring")
CACHE_VERSION = 1
# Provenance that must agree between a touch cache and the latent rows it is joined to.
JOIN_PROVENANCE_KEYS = ("dataset_kind", "split_hash", "query_manifest_hash", "episode_manifest_hash", "temporal")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def query_windows(
    episode: dict[str, np.ndarray], query_row: int, frames: int, contiguous: Callable[..., bool], temporal
) -> tuple[list[int], int]:
    """Causal source rows for one query, using the dataset's own contiguity rule.

    Slots whose row is not contiguous with the query replicate the earliest contiguous row
    (at worst the query row itself). Returns (rows oldest->newest, replicated slot count).
    """
    rows = [query_row - (frames - 1 - slot) for slot in range(frames)]
    ok = [contiguous(episode, row, query_row, temporal) for row in rows]
    if not ok[-1]:
        # The query row itself is always usable for observations; contiguous() also requires
        # a future-action row, so fall back to an explicit bounds check.
        ok[-1] = 0 <= query_row < len(episode["observation.tactile"])
    first = next(i for i, value in enumerate(ok) if value)
    replicated = 0
    for slot in range(frames):
        if not ok[slot] or slot < first:
            rows[slot] = rows[first]
            replicated += 1
    return rows, replicated


def touch_window(episode: dict[str, np.ndarray], rows: list[int], query_row: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw window [T, 8, 15, 3] (invalid pads zeroed) and per-pad validity at the query row [8]."""
    raw = np.asarray(episode["observation.tactile"][rows], dtype=np.float64).reshape(len(rows), 8, 15, 3)
    masks = np.asarray(episode["observation.tactile_valid_mask"][rows], dtype=bool).reshape(len(rows), 8)
    masks &= np.isfinite(raw).all(axis=(-2, -1))
    raw = np.where(masks[:, :, None, None], np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return raw, masks[rows.index(query_row)]


def token_valid_from_pads(pad_valid: np.ndarray, adapter: PaxiniLingBotConfig, times: int) -> np.ndarray:
    """Per-token validity [4, K] from per-pad validity [8], K ordered (time, slot)."""
    channels = pad_channels(adapter)  # [4, P] source pad per finger
    per_slot = np.repeat(pad_valid[channels], 15, axis=1)  # [4, M]
    return np.tile(per_slot, (1, times))


class _Encoder:
    """Batched token producer for one source kind."""

    def __init__(self, source: str, adapter: PaxiniLingBotConfig, *, bundle=None, random_init_seed=None):
        if source not in TOUCH_SOURCES:
            raise ValueError(f"source must be one of {TOUCH_SOURCES}")
        self.source, self.adapter = source, adapter
        self.pos, self.coverage = canonical_geometry(adapter)
        self.m = adapter.taxels_per_finger
        self.model: dict[str, Any] = {"kind": source}
        if source == "raw":
            if bundle is not None or random_init_seed is not None:
                raise ValueError("raw source takes no model")
            self.times = 1
            self.channels = adapter.frames * 3
            return
        import jax.numpy as jnp

        from openpi.models.lingbot_touch_flax import LingBotTouchConfig
        from openpi.models.lingbot_touch_flax import init_random_params
        from openpi.models.lingbot_touch_flax import load_flax_bundle

        if source == "lingbot":
            if bundle is None or random_init_seed is not None:
                raise ValueError("lingbot source requires --bundle only")
            self.config, self.params, manifest = load_flax_bundle(root_path(bundle))
            self.model.update(params_sha256=manifest["params_sha256"], source_sha256=manifest["source_sha256"])
        else:
            if random_init_seed is None or bundle is not None:
                raise ValueError("random source requires --random-init-seed only")
            self.config = LingBotTouchConfig.tiny()
            self.params = init_random_params(self.config, seed=random_init_seed)
            self.model.update(random_init_seed=int(random_init_seed))
        self.model["encoder_config"] = self.config.to_dict()
        if adapter.frames % self.config.tubelet_size:
            raise ValueError(f"frames must be divisible by tubelet_size={self.config.tubelet_size}")
        self.times = adapter.frames // self.config.tubelet_size
        self.channels = self.config.target_dim
        self._pos = jnp.asarray(self.pos)

    @property
    def k(self) -> int:
        return self.times * self.m

    def token_pos(self) -> np.ndarray:
        return np.tile(self.pos[:4], (1, self.times, 1)).astype(np.float32)  # [4, K, 3]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """signal [B, T, 5, M, 3] -> tokens [B, 4, K, C] float32 (finger-major, then time, then slot)."""
        batch = signal.shape[0]
        if self.source == "raw":
            # [B, T, 4, M, 3] -> [B, 4, M, T*3]
            return signal[:, :, :4].transpose(0, 2, 3, 1, 4).reshape(batch, 4, self.m, -1).astype(np.float32)
        import jax.numpy as jnp

        from openpi.models.lingbot_touch_flax import encode_single_hand

        latent, _, finger_id, _ = encode_single_hand(
            self.params, self.config, jnp.asarray(signal), self._pos, self.coverage, modality=self.adapter.modality
        )
        expected = np.tile(np.repeat(np.arange(4), self.m), self.times)
        if not np.array_equal(np.asarray(finger_id), expected):
            raise AssertionError("Unexpected LingBot token finger ordering")
        latent = np.asarray(latent).reshape(batch, self.times, 4, self.m, self.channels)
        return latent.transpose(0, 2, 1, 3, 4).reshape(batch, 4, self.k, self.channels)


def touch_identity(adapter: PaxiniLingBotConfig, model: dict[str, Any], k: int, channels: int) -> dict[str, Any]:
    """Split-independent identity shared by train/validation/test caches of one touch source."""
    return {
        "cache_version": CACHE_VERSION,
        "adapter": adapter.to_dict(),
        "model": model,
        "finger_order": list(CACHE_FINGER_ORDER),
        "tokens_per_finger": k,
        "channels": channels,
    }


def build_touch_cache(
    *,
    queries: Iterable[tuple[int, int]],
    episode: Callable[[int], dict[str, np.ndarray]],
    contiguous: Callable[..., bool],
    temporal,
    provenance: dict[str, Any],
    output_dir,
    source: str,
    adapter: PaxiniLingBotConfig,
    bundle=None,
    random_init_seed=None,
    batch_size: int = 256,
    progress: bool = False,
) -> dict[str, Any]:
    """Compute and write a touch cache for one split (all inputs CPU-side)."""
    queries = list(queries)
    if not queries:
        raise ValueError("No queries")
    output = root_path(output_dir)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"{output}/manifest.json exists; choose a new output directory")
    encoder = _Encoder(source, adapter, bundle=bundle, random_init_seed=random_init_seed)
    identity = touch_identity(adapter, encoder.model, encoder.k, encoder.channels)
    output.mkdir(parents=True, exist_ok=True)
    n = len(queries)
    tokens = np.lib.format.open_memmap(
        output / "tokens.npy.tmp", mode="w+", dtype=np.float16, shape=(n, 4, encoder.k, encoder.channels)
    )
    valid = np.zeros((n, 4, encoder.k), bool)
    segment = np.zeros(n, np.int32)
    frame_index = np.zeros(n, np.int64)
    stamp = np.zeros(n, np.float32)
    replicated = np.zeros(n, np.int16)
    clip_counts = np.zeros(4, np.float64)  # xy clipped, z high, z negative, elements
    started = time.perf_counter()
    for begin in range(0, n, batch_size):
        chunk = queries[begin : begin + batch_size]
        windows, pad_valid = [], []
        for i, (eid, q) in enumerate(chunk):
            ep = episode(eid)
            rows, rep = query_windows(ep, q, adapter.frames, contiguous, temporal)
            raw, pads = touch_window(ep, rows, q)
            windows.append(raw)
            pad_valid.append(pads)
            j = begin + i
            segment[j], frame_index[j] = eid, int(ep["source.frame_index"][q])
            stamp[j], replicated[j] = float(ep["source.elapsed_seconds"][q]), rep
        signal, stats = prepare_signal(np.stack(windows), adapter)  # [B, T, 5, M, 3]
        clip_counts += [
            stats.frac_xy_clipped * stats.elements * 2 / 3,
            stats.frac_z_clipped_high * stats.elements / 3,
            stats.frac_z_negative_clipped * stats.elements / 3,
            stats.elements,
        ]
        out = encoder(signal.astype(np.float32))
        if not np.isfinite(out).all():
            raise FloatingPointError(f"Nonfinite touch tokens in batch starting at {begin}")
        tokens[begin : begin + len(chunk)] = out.astype(np.float16)
        valid[begin : begin + len(chunk)] = np.stack(
            [token_valid_from_pads(p, adapter, encoder.times) for p in pad_valid]
        )
        if progress and (begin // batch_size) % 20 == 0:
            print(f"touch cache {source}: {min(begin + batch_size, n)}/{n}", flush=True)
    tokens.flush()
    del tokens
    (output / "tokens.npy.tmp").replace(output / "tokens.npy")
    np.save(output / "valid.npy", valid)
    np.save(output / "pos_xyz.npy", encoder.token_pos())
    np.savez(output / "keys.npz", segment_id=segment, query_frame_index=frame_index, query_timestamp=stamp)
    keys = set(zip(segment.tolist(), frame_index.tolist(), strict=True))
    if len(keys) != n:
        raise ValueError("Duplicate (segment_id, query_frame_index) keys in touch cache")
    manifest = {
        "identity": identity,
        "touch_identity": digest(identity),
        "provenance": {key: provenance.get(key) for key in (*JOIN_PROVENANCE_KEYS, "split", "normalization_hash")},
        "count": n,
        "files": {name: _sha256(output / name) for name in ("tokens.npy", "valid.npy", "pos_xyz.npy", "keys.npz")},
        "stats": {
            "replicated_history_queries": int(np.sum(replicated > 0)),
            "valid_token_fraction": float(valid.mean()),
            "frac_xy_clipped": float(clip_counts[0] / max(clip_counts[3] * 2 / 3, 1)),
            "frac_z_clipped_high": float(clip_counts[1] / max(clip_counts[3] / 3, 1)),
            "frac_z_negative_clipped": float(clip_counts[2] / max(clip_counts[3] / 3, 1)),
        },
        "build_seconds": time.perf_counter() - started,
        "device": "cpu",
    }
    manifest["cache_hash"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def build_assembly_touch_cache(
    protocol_dir, data_root, split, output_dir, *, selection_seal=None, max_queries=None, **kwargs
) -> dict[str, Any]:
    """Assembly wrapper: identical split/queries/contiguity as AssemblyDataset (no visual cache)."""
    from openpi.trlt.tacdex_data import AssemblyDataset
    from openpi.trlt.tacdex_data import contiguous

    if split == "test":
        from openpi.trlt.tacdex_experiment import validate_selection_seal

        # Test tokens are produced only after every compared Stage B selection is sealed.
        validate_selection_seal(selection_seal, protocol_dir=protocol_dir, allow_probe_seal=True)
    elif selection_seal is not None:
        raise ValueError("--selection-seal applies to the test split only")
    dataset = AssemblyDataset(data_root, protocol_dir, split=split)
    queries = dataset.queries[:max_queries] if max_queries else dataset.queries
    provenance = dict(dataset.provenance, max_queries=max_queries)
    manifest = build_touch_cache(
        queries=queries,
        episode=dataset._episode,  # noqa: SLF001  (shared, validated episode reader)
        contiguous=contiguous,
        temporal=dataset.temporal,
        provenance=provenance,
        output_dir=output_dir,
        **kwargs,
    )
    if max_queries:
        manifest["partial"] = True
    return manifest


class TouchLatentDataset:
    """LatentDataset rows plus frozen touch tokens joined by (segment_id, query_frame_index)."""

    def __init__(self, latents, touch_dir, finger_order: tuple[str, ...] | None = None):
        from openpi.models.trlt_actor import part_specs_for_dim

        self.latents = latents
        self.root = root_path(touch_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if digest({k: v for k, v in self.manifest.items() if k != "cache_hash"}) != self.manifest["cache_hash"]:
            raise ValueError("Touch cache manifest integrity failure")
        for name, sha in self.manifest["files"].items():
            if _sha256(self.root / name) != sha:
                raise ValueError(f"Touch cache file changed: {name}")
        lat_prov, touch_prov = getattr(latents, "provenance", {}), self.manifest["provenance"]
        for key in (*JOIN_PROVENANCE_KEYS, "split"):
            if touch_prov.get(key) is not None and digest(lat_prov.get(key)) != digest(touch_prov.get(key)):
                raise ValueError(f"Touch cache / latent provenance mismatch: {key}")
        if finger_order is None:
            action_dim = int(np.asarray(latents[0]["proprioception"]).shape[-1])
            finger_order = tuple(p.name for p in part_specs_for_dim(action_dim) if p.role == "finger")
        cache_order = tuple(self.manifest["identity"]["finger_order"])
        if sorted(finger_order) != sorted(cache_order):
            raise ValueError(f"Probe fingers {finger_order} do not match cache fingers {cache_order}")
        self.finger_order = finger_order
        self.reorder = np.asarray([cache_order.index(name) for name in finger_order])
        with np.load(self.root / "keys.npz") as keys:
            index = {
                (int(s), int(f)): i
                for i, (s, f) in enumerate(zip(keys["segment_id"], keys["query_frame_index"], strict=True))
            }
            stamps = keys["query_timestamp"]
        self.valid = np.load(self.root / "valid.npy")[:, self.reorder]
        self.pos_xyz = np.load(self.root / "pos_xyz.npy")[self.reorder].astype(np.float32)
        self.rows = np.zeros(len(latents), np.int64)
        for i in range(len(latents)):
            row = latents[i]
            key = (int(row["segment_id"]), int(row["query_frame_index"]))
            if key not in index:
                raise KeyError(f"Latent row {i} {key} missing from touch cache {self.root}")
            j = index[key]
            if abs(float(stamps[j]) - float(row["query_timestamp"])) > 1e-4:
                raise ValueError(f"Touch/latent timestamp mismatch at latent row {i}")
            self.rows[i] = j
        self.cache_hash = latents.cache_hash
        self.normalization = getattr(latents, "normalization", None)
        self.provenance = dict(
            lat_prov,
            touch_identity=self.manifest["touch_identity"],
            touch_cache_hash=self.manifest["cache_hash"],
            touch_finger_order=list(finger_order),
        )
        self._tokens = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tokens"] = None  # reopen the memmap lazily in each DataLoader worker
        return state

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index):
        if self._tokens is None:
            self._tokens = np.load(self.root / "tokens.npy", mmap_mode="r")
        row = dict(self.latents[index])
        j = self.rows[index]
        row["touch_tokens"] = np.asarray(self._tokens[j][self.reorder], dtype=np.float32)
        row["touch_valid_mask"] = self.valid[j]
        row["touch_pos_xyz"] = self.pos_xyz
        return row
