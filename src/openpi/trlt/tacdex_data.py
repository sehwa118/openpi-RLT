"""Local Assembly adapter: grouped splits, masked train statistics, native taxels.

Temporal slots use consecutive recorded rows, with actual source timestamps and
frame continuity checks. Video PTS is used only to locate that row's RGB pixels.
"""

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from functools import lru_cache
import json

import numpy as np
import pyarrow.parquet as pq

from openpi.trlt.cache import write_json
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_source import PAD_NAMES

DEFAULT_ROOT = "datasets/3dtacdex_assembly/lerobot_all"
DEFAULT_PROTOCOL = "datasets/3dtacdex_assembly/protocol"
CAMERAS = ("observation.images.main", "observation.images.secondary")
# Both cameras are external; these explicit keys do not imply wrist placement.
CAMERA_KEYS = ("base_0_rgb", "secondary_0_rgb")
DEFAULT_TEMPORAL = TemporalConfig(
    fps=5,
    visual_refresh_frames=3,
    query_stride_frames=1,
    tactile_history_frames=3,
    tactile_history_stride_frames=1,
    action_horizon_frames=3,
    max_observation_lag_seconds=0.3,
)


def seal(value):
    return dict(value, hash=digest(value))


def read_manifest(path):
    value = json.loads(root_path(path).read_text())
    if value.get("hash") != digest({k: v for k, v in value.items() if k != "hash"}):
        raise ValueError(f"Manifest hash mismatch: {path}")
    return value


def parquet_path(root, episode):
    return root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def video_path(root, episode, camera):
    return root / f"videos/chunk-{episode // 1000:03d}/{camera}/episode_{episode:06d}.mp4"


def read_episode(root, episode):
    table = pq.read_table(parquet_path(root_path(root), episode))
    return {
        k: np.asarray(table[k].to_pylist())
        for k in (
            "observation.state",
            "observation.tactile",
            "observation.tactile_valid_mask",
            "action",
            "action_valid_mask",
            "source.elapsed_seconds",
            "source.frame_index",
            "source.episode_id",
            "source.group",
            "source.label",
            "source.instruction",
        )
    }


def inventory(data_root=DEFAULT_ROOT):
    root = root_path(data_root)
    conversion = json.loads((root / "meta/assembly_conversion.json").read_text())
    if conversion["status"] != "complete" or digest(conversion["identity"]) != conversion["identity_hash"]:
        raise ValueError("Incomplete/corrupt Assembly conversion")
    if conversion["identity"]["cohort"] != "all":
        raise ValueError("This comparison requires all source versions, including nouse")
    rows, files = [], []
    for eid, segment in enumerate(conversion["segments"]):
        # Metadata only: do not read held-out action values to construct the split.
        rows.append(
            {
                "episode_index": eid,
                "length": segment["length"],
                "source_episode_id": segment["source_episode_id"],
                "source_group": segment["source_group"],
                "source_label": segment["source_label"],
            }
        )
        for path in (parquet_path(root, eid), *(video_path(root, eid, c) for c in CAMERAS)):
            stat = path.stat()
            files.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    return seal(
        {
            "dataset_kind": "3dtacdex_assembly",
            "conversion_hash": conversion["identity_hash"],
            "revision": conversion["identity"]["source_revision"],
            "source_sha256": conversion["identity"]["source_sha256"],
            "episodes": rows,
            "local_file_inventory": files,
        }
    )


def grouped_split(source, seed=42):
    strata, trajectory_strata = defaultdict(set), {}
    for row in source["episodes"]:
        key = row["source_episode_id"]
        stratum = (row["source_group"], row["source_label"])
        if key in trajectory_strata and trajectory_strata[key] != stratum:
            raise ValueError("One source trajectory has conflicting strata")
        trajectory_strata[key] = stratum
        strata[stratum].add(key)
    assignments = {}
    rng = np.random.default_rng(seed)
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError("Each source stratum needs at least three trajectories")
        nval = ntest = max(1, round(n * 0.15))
        for name, part in zip(
            ("train", "validation", "test"),
            (ids[: n - nval - ntest], ids[n - nval - ntest : n - ntest], ids[n - ntest :]),
            strict=True,
        ):
            assignments.update(dict.fromkeys(part, name))
    return seal(
        {
            "source_hash": source["hash"],
            "seed": seed,
            "grouping": "source_episode_id",
            "stratification": ["source_group", "source_label"],
            "ratios": [0.7, 0.15, 0.15],
            "trajectory_assignments": assignments,
            "splits": {
                name: [r["episode_index"] for r in source["episodes"] if assignments[r["source_episode_id"]] == name]
                for name in ("train", "validation", "test")
            },
        }
    )


def validate_split(source, split):
    if split["source_hash"] != source["hash"]:
        raise ValueError("Stale Assembly split")
    ids = [eid for part in split["splits"].values() for eid in part]
    if len(ids) != len(set(ids)) or set(ids) != {r["episode_index"] for r in source["episodes"]}:
        raise ValueError("Split overlap/missing episodes")
    seen = {}
    for name, part in split["splits"].items():
        for eid in part:
            key = source["episodes"][eid]["source_episode_id"]
            if key in seen and seen[key] != name:
                raise ValueError("Source trajectory crosses experimental splits")
            seen[key] = name
    if seen != split["trajectory_assignments"]:
        raise ValueError("Trajectory assignment mismatch")


def _statistics(values, masks, epsilon=1e-6):
    count = np.zeros(values[0].shape[1:], np.int64)
    total = np.zeros_like(count, np.float64)
    total2 = np.zeros_like(total)
    for values_part, mask_part in zip(values, masks, strict=True):
        mask = np.broadcast_to(mask_part, values_part.shape) & np.isfinite(values_part)
        x = np.where(mask, values_part, 0)
        count += mask.sum(axis=0)
        total += x.sum(axis=0)
        total2 += (x * x).sum(axis=0)
    if np.any(count == 0):
        raise ValueError("A dimension has no valid train observations")
    mean = total / count
    std = np.sqrt(np.maximum(total2 / count - mean * mean, 0))
    return {
        "mean": mean.tolist(),
        "std": np.maximum(std, epsilon).tolist(),
        "count": count.tolist(),
        "epsilon": epsilon,
        "floored": (std < epsilon).tolist(),
    }


def prepare_protocol(data_root=DEFAULT_ROOT, protocol_dir=DEFAULT_PROTOCOL, *, seed=42, temporal=DEFAULT_TEMPORAL):
    """CPU preparation only. Reuse an identical protocol; never overwrite another split."""
    root, output = root_path(data_root), root_path(protocol_dir)
    source = inventory(root)
    split = grouped_split(source, seed)
    if (output / "protocol.json").exists():
        existing = read_manifest(output / "protocol.json")
        if (
            existing["source_hash"] != source["hash"]
            or existing["split_hash"] != split["hash"]
            or existing["temporal"] != asdict(temporal)
        ):
            raise ValueError("Existing protocol differs; use a distinct protocol directory")
        validate_split(source, read_manifest(output / "splits.json"))
        norm = read_manifest(output / "normalization.json")
        queries = read_manifest(output / "queries.json")
        if norm["hash"] != existing["normalization_hash"] or queries["hash"] != existing["query_manifest_hash"]:
            raise ValueError("Stale protocol artifacts")
        return existing
    # An interrupted partial protocol must be inspected, not silently replaced.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Partial/unrelated protocol directory")
    validate_split(source, split)
    train = [read_episode(root, eid) for eid in split["splits"]["train"]]
    norm = {
        "source_hash": source["hash"],
        "split_hash": split["hash"],
        "fit_split": "train",
        "train_episode_ids": split["splits"]["train"],
        "action_groups": {"arm": list(range(6)), "hand": list(range(6, 22))},
    }
    for dst, field, mask in (
        ("state", "observation.state", lambda ep: np.ones_like(ep["observation.state"], bool)),
        ("action", "action", lambda ep: ep["action_valid_mask"]),
        ("tactile", "observation.tactile", lambda ep: ep["observation.tactile_valid_mask"][..., None, None]),
    ):
        norm[dst] = _statistics([ep[field] for ep in train], [mask(ep) for ep in train])
    norm = seal(norm)
    queries = seal(
        {
            "source_hash": source["hash"],
            "split_hash": split["hash"],
            "temporal": asdict(temporal),
            "queries": [
                [r["episode_index"], q]
                for r in source["episodes"]
                for q in range(0, r["length"], temporal.query_stride_frames)
            ],
        }
    )
    protocol = seal(
        {
            "dataset_kind": "3dtacdex_assembly",
            "source_hash": source["hash"],
            "split_hash": split["hash"],
            "normalization_hash": norm["hash"],
            "query_manifest_hash": queries["hash"],
            "temporal": asdict(temporal),
            "fit_split": "train",
            "training_steps": 0,
        }
    )
    for name, value in (
        ("source", source),
        ("splits", split),
        ("normalization", norm),
        ("queries", queries),
        ("protocol", protocol),
    ):
        write_json(output / f"{name}.json", value)
    return protocol


def contiguous(ep, start, end, temporal):
    if start < 0 or end >= len(ep["action"]):
        return False
    stamps = ep["source.elapsed_seconds"]
    frames = ep["source.frame_index"]
    delta = np.diff(stamps[start : end + 1])
    return bool(
        np.all(np.diff(frames[start : end + 1]) == 1)
        and np.all(delta > 0)
        and np.all(delta <= temporal.max_observation_lag_seconds)
    )


class AssemblyDataset:
    camera_order = CAMERAS
    camera_keys = CAMERA_KEYS

    def __init__(
        self,
        data_root=DEFAULT_ROOT,
        protocol_dir=DEFAULT_PROTOCOL,
        *,
        split="train",
        temporal=DEFAULT_TEMPORAL,
        visual_cache=None,
    ):
        self.root = root_path(data_root)
        directory = root_path(protocol_dir)
        self.source = read_manifest(directory / "source.json")
        if inventory(self.root)["hash"] != self.source["hash"]:
            raise ValueError("Converted source files changed since protocol preparation")
        self.split = read_manifest(directory / "splits.json")
        validate_split(self.source, self.split)
        self.normalization = read_manifest(directory / "normalization.json")
        query = read_manifest(directory / "queries.json")
        protocol = read_manifest(directory / "protocol.json")
        if (
            self.normalization["split_hash"] != self.split["hash"]
            or self.normalization["train_episode_ids"] != self.split["splits"]["train"]
            or self.normalization["fit_split"] != "train"
            or protocol["normalization_hash"] != self.normalization["hash"]
            or protocol["query_manifest_hash"] != query["hash"]
            or query["temporal"] != asdict(temporal)
        ):
            raise ValueError("Assembly normalization/query/protocol mismatch")
        self.temporal = temporal
        ids = set(self.split["splits"][split])
        self.queries = [tuple(row) for row in query["queries"] if row[0] in ids]
        self.original_ids = sorted({r["source_episode_id"] for r in self.source["episodes"]})
        self.visual_cache = visual_cache
        self.provenance = {
            "dataset_kind": "3dtacdex_assembly",
            "tactile_format": "native_ordered_taxels_xyz",
            "revision": self.source["revision"],
            "episode_manifest_hash": self.source["hash"],
            "split_hash": self.split["hash"],
            "normalization_hash": self.normalization["hash"],
            "query_manifest_hash": query["hash"],
            "temporal": asdict(temporal),
            "camera_order": CAMERAS,
            "sensor_order": PAD_NAMES,
            "caption_hash": digest("Assemble the object with the dexterous hand."),
            "feature_cache_hash": getattr(visual_cache, "cache_hash", None),
            "split": split,
        }
        if visual_cache is not None:
            meta = visual_cache.manifest["metadata"]
            for key, value in (
                ("episode_manifest_hash", self.source["hash"]),
                ("camera_order", list(CAMERAS)),
                ("caption_hash", self.provenance["caption_hash"]),
                ("refresh_frames", temporal.visual_refresh_frames),
            ):
                if digest(meta.get(key)) != digest(value):
                    raise ValueError(f"Assembly visual cache mismatch: {key}")
            if digest(meta.get("anchor_sampling")) != digest(
                {"fps": temporal.fps, "max_observation_lag_seconds": temporal.max_observation_lag_seconds}
            ):
                raise ValueError("Assembly visual temporal mismatch")
            missing = {
                (e, q // temporal.visual_refresh_frames * temporal.visual_refresh_frames)
                for e, q in self.queries
                if (e, q // temporal.visual_refresh_frames * temporal.visual_refresh_frames) not in visual_cache
            }
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} Assembly visual anchors")

    def __len__(self):
        return len(self.queries)

    @lru_cache(maxsize=128)  # noqa: B019 -- bounded by dataset lifetime
    def _episode(self, episode):
        ep = read_episode(self.root, episode)
        stamps = ep["source.elapsed_seconds"]
        if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError("Nonmonotonic/nonfinite source timestamps")
        if not np.all(ep["source.instruction"] == "Assemble the object with the dexterous hand."):
            raise ValueError("Unexpected instruction; cache caption identity needs updating")
        return ep

    def read_anchor(self, episode, anchor):
        import av

        ep = self._episode(episode)
        images = {}
        # Frame counter -> nominal video PTS locates exactly the original row.
        for camera, key in zip(CAMERAS, CAMERA_KEYS, strict=True):
            with av.open(str(video_path(self.root, episode, camera))) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                container.seek(max(0, int((anchor / 5 - 0.01) / float(stream.time_base))), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is not None and round(float(frame.time) * 5) == anchor:
                        images[key] = frame.to_ndarray(format="rgb24")
                        break
            if key not in images:
                raise ValueError("Missing exact Assembly video row")
        timestamp = float(ep["source.elapsed_seconds"][anchor])
        return {
            "images": images,
            "caption": str(ep["source.instruction"][anchor]),
            "visual_timestamp": timestamp,
            "camera_timestamps": [timestamp] * 2,
        }

    def iter_anchors(self, episode, anchors):
        """Decode each camera once from row0, yielding only exact requested rows."""
        import av

        anchors = list(anchors)
        if not anchors or anchors != sorted(set(anchors)) or anchors[0] < 0:
            raise ValueError("Anchor indices must be nonempty, unique, sorted and nonnegative")
        ep = self._episode(episode)
        if anchors[-1] >= len(ep["source.elapsed_seconds"]):
            raise IndexError("Anchor outside source segment")
        with ExitStack() as stack:
            streams = {}
            for camera, key in zip(CAMERAS, CAMERA_KEYS, strict=True):
                container = stack.enter_context(av.open(str(video_path(self.root, episode, camera))))
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                streams[key] = iter(container.decode(stream))
            previous = dict.fromkeys(CAMERA_KEYS, -1)
            for anchor in anchors:
                images = {}
                for key in CAMERA_KEYS:
                    for frame in streams[key]:
                        if frame.pts is None:
                            raise ValueError("Missing video PTS")
                        row = round(float(frame.time) * 5)
                        if row != previous[key] + 1 or abs(float(frame.time) - row / 5) > 1e-6:
                            raise ValueError("Noncontiguous/misaligned native video PTS")
                        previous[key] = row
                        if row == anchor:
                            images[key] = frame.to_ndarray(format="rgb24")
                            break
                        if row > anchor:
                            raise ValueError("Missing requested source video row")
                    if key not in images:
                        raise ValueError("Truncated video before requested anchor")
                stamp = float(ep["source.elapsed_seconds"][anchor])
                yield (
                    (episode, anchor),
                    {
                        "images": images,
                        "caption": str(ep["source.instruction"][anchor]),
                        "visual_timestamp": stamp,
                        "camera_timestamps": [stamp] * 2,
                    },
                )

    def __getitem__(self, index):
        eid, q = self.queries[index]
        ep = self._episode(eid)
        t = self.temporal
        n = self.normalization
        stamps = ep["source.elapsed_seconds"]
        tactile = np.zeros((t.tactile_history_frames, 8, 15, 1, 3), np.float32)
        valid = np.zeros(tactile.shape[:2], bool)
        times = np.full(valid.shape, stamps[q], np.float32)
        for slot in range(t.tactile_history_frames):
            row = q - (t.tactile_history_frames - 1 - slot) * t.tactile_history_stride_frames
            if contiguous(ep, row, q, t):
                mask = ep["observation.tactile_valid_mask"][row].astype(bool)
                x = (ep["observation.tactile"][row] - n["tactile"]["mean"]) / n["tactile"]["std"]
                mask &= np.isfinite(x).all(axis=(-2, -1))
                tactile[slot, :, :, 0, :] = np.where(mask[:, None, None], x, 0)
                valid[slot] = mask
                times[slot] = stamps[row]
        target = np.zeros((t.action_horizon_frames, 22), np.float32)
        action_valid = np.zeros_like(target, bool)
        for offset in range(t.action_horizon_frames):
            row = q + offset
            if contiguous(ep, q, row, t):
                mask = ep["action_valid_mask"][row].astype(bool) & np.isfinite(ep["action"][row])
                target[offset] = np.where(mask, (ep["action"][row] - n["action"]["mean"]) / n["action"]["std"], 0)
                action_valid[offset] = mask
        anchor = q // t.visual_refresh_frames * t.visual_refresh_frames
        # Original trajectory IDs group multiple LeRobot segments in final metrics.
        result = {
            "tactile": tactile,
            "tactile_valid_mask": valid,
            "tactile_timestamps": times,
            "proprioception": np.asarray(
                (ep["observation.state"][q] - n["state"]["mean"]) / n["state"]["std"], np.float32
            ),
            "action_target": target,
            "action_valid_mask": action_valid,
            "query_timestamp": np.float32(stamps[q]),
            "visual_timestamp": np.float32(stamps[anchor]),
            "episode_id": np.int32(self.original_ids.index(self.source["episodes"][eid]["source_episode_id"])),
            "query_frame_index": np.int64(ep["source.frame_index"][q]),
            "anchor_frame_index": np.int64(ep["source.frame_index"][anchor]),
            "segment_id": np.int32(eid),
            "source_episode_id": str(ep["source.episode_id"][q]),
            "source_group": str(ep["source.group"][q]),
            "source_label": str(ep["source.label"][q]),
            "object": str(ep["source.group"][q]),
            "caption": str(ep["source.instruction"][q]),
        }
        if self.visual_cache is not None:
            context = self.visual_cache[eid, anchor]
            result.update({k: context[k] for k in ("visual_features", "visual_valid_mask")})
            if context["visual_features"].shape != (512, 2048) or not np.isclose(
                context["visual_timestamp"], stamps[anchor], rtol=0, atol=1e-4
            ):
                raise ValueError("Invalid Assembly anchor dimensions/timestamp")
        return result
