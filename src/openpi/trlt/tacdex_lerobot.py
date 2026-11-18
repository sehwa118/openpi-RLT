"""Streaming lossless Assembly conversion to the installed LeRobot v2.1 format.

Only public LeRobot metadata/schema APIs are used. RGB is streamed directly to
CPU H.264 RGB lossless videos, avoiding an intermediate full PNG extraction.
"""

from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from openpi.trlt.config import digest
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_source import ARCHIVE_SHA256
from openpi.trlt.tacdex_source import CAMERAS
from openpi.trlt.tacdex_source import DEFAULT_REPORT
from openpi.trlt.tacdex_source import DEFAULT_ROOT
from openpi.trlt.tacdex_source import PAD_NAMES
from openpi.trlt.tacdex_source import REVISION
from openpi.trlt.tacdex_source import STATE_NAMES
from openpi.trlt.tacdex_source import archive_path
from openpi.trlt.tacdex_source import command_arrays
from openpi.trlt.tacdex_source import open_assembly
from openpi.trlt.tacdex_source import selected_segments
from openpi.trlt.tacdex_source import sha256_file
from openpi.trlt.tacdex_source import utc_now
from openpi.trlt.tacdex_source import verify_archive
from openpi.trlt.tacdex_source import write_json

FPS = 5
REPO_ID = "local/3dtacdex_assembly"
FORMAT_VERSION = 1


def feature_spec(group):
    def numeric(dtype, shape, names=None):
        return {"dtype": dtype, "shape": shape, "names": names}

    features = {
        "observation.state": numeric("float64", (22,), [list(STATE_NAMES)]),
        "action": numeric("float64", (22,), [list(STATE_NAMES)]),
        "action_valid_mask": numeric("bool", (22,)),
        "observation.wrist_pose": numeric("float64", (6,)),
        "observation.tactile": numeric("float64", (8, 15, 3)),
        "observation.tactile_valid_mask": numeric("bool", (8,)),
        "source.timestamp": numeric("float64", (1,)),
        "source.elapsed_seconds": numeric("float64", (1,)),
        "source.global_frame_index": numeric("int64", (1,)),
        "source.frame_index": numeric("int64", (1,)),
        "source.position": numeric("int64", (1,)),
        "source.segment_id": numeric("int64", (1,)),
        "source.episode_id": numeric("string", (1,)),
        "source.group": numeric("string", (1,)),
        "source.label": numeric("string", (1,)),
        "source.instruction": numeric("string", (1,)),
        "source.record_sha256": numeric("string", (1,)),
    }
    for name, key in CAMERAS.items():
        _, h, w, c = group[key].shape
        if c != 3 or group[key].dtype != np.dtype("uint8"):
            raise ValueError("RGB must have three uint8 channels")
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
            # The pinned LeRobot get_video_info does not recognize planar gbrp.
            # Describe the actual bitstream explicitly; the reader decodes RGB.
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "h264",
                "video.pix_fmt": "gbrp",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    return features


def episode_columns(group, segment, action, action_valid):
    start, end = segment["output_start"], segment["output_end"]
    n = end - start
    sl = slice(start, end)
    timestamp = np.asarray(group["data/timestamps"][sl], np.float64)
    return {
        "observation.state": np.concatenate(
            (group["data/right_arm_joints"][sl], group["data/right_hand_joints"][sl]), axis=1
        ).astype(np.float64),
        "action": action[sl],
        "action_valid_mask": action_valid[sl],
        "observation.wrist_pose": np.asarray(group["data/right_wrist_pose"][sl], np.float64),
        "observation.tactile": np.asarray(group["data/right_tactile_data_fingers"][sl], np.float64),
        "observation.tactile_valid_mask": np.ones((n, 8), bool),
        "source.timestamp": timestamp,
        "source.elapsed_seconds": timestamp - timestamp[0],
        "source.global_frame_index": np.arange(start, end, dtype=np.int64),
        "source.frame_index": np.asarray(group["data/source_frame_index"][sl], np.int64),
        "source.position": np.asarray(segment["source_positions"], np.int64),
        "source.segment_id": np.full(n, segment["segment_id"], np.int64),
        "source.episode_id": np.asarray(group["data/source_episode_id"][sl]).astype(str),
        "source.group": np.asarray(group["data/source_group"][sl]).astype(str),
        "source.label": np.full(n, segment["source_label"]),
        "source.instruction": np.asarray(group["data/sub_task_instruction"][sl]).astype(str),
        "source.record_sha256": np.asarray(group["data/source_record_sha256"][sl]).astype(str),
    }


def conversion_identity(group, segments, cohort):
    return {
        "format_version": FORMAT_VERSION,
        "source_revision": REVISION,
        "source_sha256": ARCHIVE_SHA256,
        "cohort": cohort,
        "source_segment_ids": [s["segment_id"] for s in segments],
        "features": feature_spec(group),
        "fps": FPS,
        "codec": {"name": "libx264rgb", "crf": 0, "preset": "ultrafast", "pixel_format": "rgb24"},
        "timing": "LeRobot/video timestamp=frame_index/5; source timestamps/elapsed preserved separately; no resampling.",
        "action": "Source joint commands, unchanged order/sign/units; missing elements zero-padded with false mask.",
    }


def encode_segment_videos(group, segment, meta, episode_index, *, threads=2):
    import av

    statistics = {}
    with ExitStack() as stack:
        outputs = {}
        for name in CAMERAS:
            key = f"observation.images.{name}"
            path = meta.root / meta.get_video_file_path(episode_index, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + ".partial.mp4")
            container = stack.enter_context(av.open(str(temporary), "w"))
            stream = container.add_stream("libx264rgb", rate=FPS, options={"crf": "0", "preset": "ultrafast"})
            _, h, w, _ = group[CAMERAS[name]].shape
            stream.width, stream.height, stream.pix_fmt = w, h, "rgb24"
            stream.codec_context.thread_count = threads
            outputs[name] = (container, stream, temporary, path)
            statistics[key] = {
                "sum": np.zeros(3),
                "sum2": np.zeros(3),
                "pixels": 0,
                "min": np.full(3, np.inf),
                "max": np.full(3, -np.inf),
            }
        for i, source_index in enumerate(range(segment["output_start"], segment["output_end"])):
            for name, (container, stream, _, _) in outputs.items():
                rgb = np.asarray(group[CAMERAS[name]][source_index])
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, FPS)
                for packet in stream.encode(frame):
                    container.mux(packet)
                # Descriptive LeRobot image statistics, spatially sampled only.
                sample = rgb[::8, ::8].reshape(-1, 3).astype(np.float64) / 255
                stats = statistics[f"observation.images.{name}"]
                stats["sum"] += sample.sum(axis=0)
                stats["sum2"] += (sample**2).sum(axis=0)
                stats["pixels"] += len(sample)
                stats["min"] = np.minimum(stats["min"], sample.min(axis=0))
                stats["max"] = np.maximum(stats["max"], sample.max(axis=0))
        for container, stream, _, _ in outputs.values():
            for packet in stream.encode():
                container.mux(packet)
    for _, _, temporary, path in outputs.values():
        temporary.replace(path)
    result = {}
    for key, stats in statistics.items():
        mean = stats["sum"] / stats["pixels"]
        std = np.sqrt(np.maximum(stats["sum2"] / stats["pixels"] - mean**2, 0))
        result[key] = {
            k: np.asarray(v).reshape(3, 1, 1)
            for k, v in {"mean": mean, "std": std, "min": stats["min"], "max": stats["max"]}.items()
        }
        result[key]["count"] = np.array([segment["length"]])
    return result


def convert_group(group, output_dir, *, cohort="all", threads=2, resume=False):
    from filelock import FileLock
    from lerobot.common.datasets.compute_stats import compute_episode_stats
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import get_hf_features_from_features

    import datasets

    if threads < 1:
        raise ValueError("threads must be positive")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir) + ".lock"):
        segments = selected_segments(group, cohort)
        if not segments:
            raise ValueError("Selected cohort is empty")
        identity = conversion_identity(group, segments, cohort)
        identity_hash = digest(identity)
        manifest_path = output_dir / "meta/assembly_conversion.json"
        if output_dir.exists():
            if not manifest_path.exists():
                raise ValueError("Output exists without Assembly provenance; choose a new output directory")
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity_hash"] != identity_hash:
                raise ValueError("Existing conversion provenance differs; choose a new output directory")
            if manifest["status"] == "complete":
                return manifest
            if not resume:
                raise ValueError("Incomplete output preserved; use --resume to continue committed episodes")
            meta = LeRobotDatasetMetadata(REPO_ID, root=output_dir)
            if len(meta.episodes) != meta.total_episodes or len(meta.episodes_stats) != meta.total_episodes:
                raise ValueError("Interrupted metadata commit; inspect output before recovery")
        else:
            meta = LeRobotDatasetMetadata.create(
                REPO_ID, fps=FPS, root=output_dir, robot_type="jaka_minicobo_leap_right", features=feature_spec(group)
            )
            manifest = {
                "identity": identity,
                "identity_hash": identity_hash,
                "status": "converting",
                "created_at": utc_now(),
                "segments": segments,
                "tactile_pad_order": PAD_NAMES,
                "source_attributes": dict(group.attrs),
                "statistics_scope": "LeRobot descriptive per-episode metadata only; not training normalization.",
                "implementation_sha256": sha256_file(__file__),
            }
            for source_key, label in (
                ("data/right_hand_joints_idx", "hand_joint_oxt_indices"),
                ("data/right_tactile_area_fingers", "tactile_area_oxt_indices"),
            ):
                if source_key in group:
                    indices = np.asarray(group[source_key][:])
                    if not np.all(indices == indices[0]):
                        raise ValueError(f"Source mapping changes across frames: {source_key}")
                    manifest[label] = indices[0].tolist()
            write_json(manifest_path, manifest)
        action, valid = command_arrays(group)
        hf_features = get_hf_features_from_features(meta.features)
        for episode_index, segment in enumerate(segments):
            if episode_index < meta.total_episodes:
                continue
            print(
                f"Converting {episode_index + 1}/{len(segments)}: source segment {segment['segment_id']} "
                f"({segment['length']} frames)",
                flush=True,
            )
            columns = episode_columns(group, segment, action, valid)
            n = segment["length"]
            episode_tasks = sorted(set(columns["source.instruction"].tolist()))
            for task in episode_tasks:
                if not task.strip():
                    raise ValueError("Source task instruction is empty")
                if meta.get_task_index(task) is None:
                    meta.add_task(task)
            columns.update(
                timestamp=np.arange(n, dtype=np.float32) / FPS,
                frame_index=np.arange(n, dtype=np.int64),
                episode_index=np.full(n, episode_index, np.int64),
                index=np.arange(meta.total_frames, meta.total_frames + n, dtype=np.int64),
                task_index=np.asarray([meta.get_task_index(task) for task in columns["source.instruction"]], np.int64),
            )
            stats = compute_episode_stats(columns, meta.features)
            stats.update(encode_segment_videos(group, segment, meta, episode_index, threads=threads))
            table = datasets.Dataset.from_dict(columns, features=hf_features)
            path = output_dir / meta.get_data_file_path(episode_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial.parquet")
            table.to_parquet(temporary)
            temporary.replace(path)
            meta.save_episode(episode_index, n, episode_tasks, stats)
            manifest.update(
                completed_episodes=meta.total_episodes, completed_frames=meta.total_frames, updated_at=utc_now()
            )
            write_json(manifest_path, manifest)
        manifest.update(status="complete", completed_at=utc_now())
        write_json(manifest_path, manifest)
        return manifest


def convert_assembly(data_root=DEFAULT_ROOT, *, output_dir=None, cohort="all", threads=2, resume=False):
    verify_archive(data_root)
    root = root_path(data_root)
    output_dir = root_path(output_dir) if output_dir else root / f"lerobot_{cohort}"
    with open_assembly(archive_path(root)) as group:
        return convert_group(group, output_dir, cohort=cohort, threads=threads, resume=resume)


def validate_conversion(group, output_dir, *, all_rgb=True):
    """Compare every numeric row and optionally every decoded RGB pixel to source."""
    import av
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "meta/assembly_conversion.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Conversion is not complete")
    # Real local reader checks official metadata, Arrow schema, temporal sync,
    # torch transforms and PyAV camera decoding. It must never download files.
    dataset = LeRobotDataset(REPO_ID, root=output_dir, video_backend="pyav")
    segments = selected_segments(group, manifest["identity"]["cohort"])
    if digest(conversion_identity(group, segments, manifest["identity"]["cohort"])) != manifest["identity_hash"]:
        raise ValueError("Source/feature conversion identity mismatch")
    if len(segments) != dataset.num_episodes or len(dataset) != sum(s["length"] for s in segments):
        raise ValueError("Episode/frame totals differ")
    action, valid = command_arrays(group)
    checked_rgb, checked_rows, source_ids = 0, 0, set()
    file_hashes = {}
    for episode_index, segment in enumerate(segments):
        expected = episode_columns(group, segment, action, valid)
        path = output_dir / dataset.meta.get_data_file_path(episode_index)
        table = pq.read_table(path)
        for key, values in expected.items():
            actual = np.asarray(table[key].to_pylist(), dtype=values.dtype)
            if not np.array_equal(actual, values):
                raise ValueError(f"Numeric/source mismatch: episode {episode_index}, {key}")
        tasks = np.asarray([dataset.meta.tasks[int(i)] for i in table["task_index"].to_pylist()])
        if not np.array_equal(tasks, expected["source.instruction"]):
            raise ValueError("LeRobot task mapping differs from source instruction")
        times = np.asarray(table["timestamp"].to_pylist())
        if not np.allclose(times, np.arange(segment["length"]) / FPS, atol=1e-5, rtol=0):
            raise ValueError("LeRobot timestamps differ from video frame times")
        checked_rows += segment["length"]
        source_ids.add(segment["source_episode_id"])
        file_hashes[str(path.relative_to(output_dir))] = sha256_file(path)
        for name, source_key in CAMERAS.items():
            video = output_dir / dataset.meta.get_video_file_path(episode_index, f"observation.images.{name}")
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                if (stream.width, stream.height) != tuple(group[source_key].shape[2:0:-1]):
                    raise ValueError("Video dimensions differ")
                count = 0
                for frame in container.decode(stream):
                    if count >= segment["length"] or abs(float(frame.time) - count / FPS) > 1e-6:
                        raise ValueError("Unexpected video frame count or PTS")
                    if all_rgb or count in (0, segment["length"] - 1):
                        rgb = frame.to_ndarray(format="rgb24")
                        source = np.asarray(group[source_key][segment["output_start"] + count])
                        if not np.array_equal(rgb, source):
                            raise ValueError(f"RGB mismatch: episode {episode_index}, {name}, frame {count}")
                        checked_rgb += 1
                    count += 1
                if count != segment["length"]:
                    raise ValueError("Video frame count differs")
            file_hashes[str(video.relative_to(output_dir))] = sha256_file(video)
        # Check both episode boundaries through the actual LeRobot __getitem__.
        offset = int(dataset.episode_data_index["from"][episode_index])
        for i in sorted({0, segment["length"] - 1}):
            item = dataset[offset + i]
            if item["task"] != expected["source.instruction"][i]:
                raise ValueError("LeRobot reader task instruction differs")
            for key in ("observation.state", "action", "observation.tactile", "action_valid_mask"):
                actual = item[key].numpy()
                # Pinned LeRobot's torch transform infers float32 from Arrow
                # Python lists. Storage precision is checked separately above.
                if not np.array_equal(actual, expected[key][i].astype(actual.dtype)):
                    raise ValueError(f"LeRobot reader mismatch: {key}")
            for name in CAMERAS:
                key = f"observation.images.{name}"
                rgb = np.rint(item[key].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                source = np.asarray(group[CAMERAS[name]][segment["output_start"] + i])
                if not np.array_equal(rgb, source):
                    raise ValueError("Actual LeRobot RGB reader differs")
        print(f"Validated {episode_index + 1}/{len(segments)}", flush=True)
    return {
        "status": "pass",
        "validated_at": utc_now(),
        "device": "CPU",
        "identity_hash": manifest["identity_hash"],
        "episodes": len(segments),
        "source_trajectories": len(source_ids),
        "numeric_rows": checked_rows,
        "rgb_frames_compared_exactly": checked_rgb,
        "all_rgb": all_rgb,
        "lerobot_reader_boundary_samples": sum(min(2, s["length"]) for s in segments),
        "complete_action_frames": sum(
            int(valid[s["output_start"] : s["output_end"]].all(axis=1).sum()) for s in segments
        ),
        "file_sha256": file_hashes,
        "training_steps": 0,
        "split_created": False,
        "normalization_fitted": False,
    }


def validate_assembly(data_root=DEFAULT_ROOT, *, output_dir=None, report_dir=DEFAULT_REPORT, all_rgb=True):
    verify_archive(data_root)
    root = root_path(data_root)
    output_dir = root_path(output_dir) if output_dir else root / "lerobot_all"
    with open_assembly(archive_path(root)) as group:
        result = validate_conversion(group, output_dir, all_rgb=all_rgb)
    write_json(root_path(report_dir) / f"{output_dir.name}_validation.json", result)
    return result
