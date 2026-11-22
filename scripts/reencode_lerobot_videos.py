"""Create a resumable training copy using the installed LeRobot encoder defaults.

Run with CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu. Source videos, vector
parquets, frame order, timestamps, resolution and episode selection are preserved.
Only the destination videos and their image statistics are regenerated.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

import av
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.datasets.video_utils import get_video_info
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def init_worker(cpus_per_job):
    available = sorted(os.sched_getaffinity(0))
    index = int(multiprocessing.current_process().name.rsplit("-", 1)[-1]) - 1
    groups = max(1, len(available) // cpus_per_job)
    start = (index % groups) * cpus_per_job
    os.sched_setaffinity(0, available[start : start + cpus_per_job])


def convert_video(job):
    source, destination, fps, expected_frames, key, episode, progress_dir = job
    source, destination = Path(source), Path(destination)
    record_path = Path(progress_dir) / f"{episode:06d}_{key}.json"
    identity = {"source": str(source), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
    if record_path.exists() and destination.exists():
        record = json.loads(record_path.read_text())
        if record["source_identity"] == identity and record["output_sha256"] == sha256(destination):
            return record
        raise ValueError(f"Resume identity mismatch: {destination}")
    start = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = destination.with_suffix(".encoding.mp4")
    try:
        with tempfile.TemporaryDirectory(prefix=".frames-", dir=destination.parent) as temporary_frames:
            frame_count = 0
            reference_frames = {}
            with av.open(str(source)) as container:
                source_stream = container.streams.video[0]
                width, height = source_stream.width, source_stream.height
                for i, frame in enumerate(container.decode(video=0)):
                    timestamp = float(frame.pts * frame.time_base)
                    if abs(timestamp - i / fps) > 1e-4:
                        raise ValueError(f"Nonuniform source timestamp: {source}: {i}: {timestamp}")
                    rgb = frame.to_ndarray(format="rgb24")
                    if i in (0, expected_frames // 2, expected_frames - 1):
                        reference_frames[i] = rgb
                    Image.fromarray(rgb).save(Path(temporary_frames) / f"frame_{i:06d}.png", compress_level=0)
                    frame_count += 1
            if frame_count != expected_frames:
                raise ValueError(f"Source length mismatch: {source}: {frame_count} != {expected_frames}")
            # Codec, pixel format, GOP, quality and fast_decode use LeRobot defaults.
            # overwrite=True is needed by this version when the parent already exists.
            encode_video_frames(temporary_frames, temporary_video, fps, overwrite=True)
        sums = np.zeros(3)
        sums2 = np.zeros(3)
        minimum = np.full(3, np.inf)
        maximum = np.full(3, -np.inf)
        pixels = 0
        count = 0
        keyframes = []
        squared_errors = []
        with av.open(str(temporary_video)) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (width, height):
                raise ValueError(f"Resolution changed: {destination}")
            for i, frame in enumerate(container.decode(video=0)):
                if abs(float(frame.pts * frame.time_base) - i / fps) > 1e-4:
                    raise ValueError(f"Output timestamp mismatch: {destination}: {i}")
                if frame.key_frame:
                    keyframes.append(i)
                rgb = frame.to_ndarray(format="rgb24")
                # Match the original converter's spatially sampled image statistics.
                sample = rgb[::8, ::8].reshape(-1, 3).astype(np.float64) / 255
                sums += sample.sum(axis=0)
                sums2 += (sample**2).sum(axis=0)
                minimum = np.minimum(minimum, sample.min(axis=0))
                maximum = np.maximum(maximum, sample.max(axis=0))
                pixels += len(sample)
                if i in reference_frames:
                    error = rgb.astype(np.float32) - reference_frames[i].astype(np.float32)
                    squared_errors.append(float(np.mean(error**2)))
                count += 1
        max_gap = max(np.diff([*keyframes, count]).tolist()) if keyframes else count
        if count != expected_frames or not keyframes or keyframes[0] != 0 or max_gap > 2:
            raise ValueError(f"Invalid output count/GOP: {destination}: {count}, {max_gap}")
        video_info = get_video_info(temporary_video)
        mean = sums / pixels
        std = np.sqrt(np.maximum(sums2 / pixels - mean**2, 0))
        stats = {
            k: v[:, None, None].tolist() for k, v in (("mean", mean), ("std", std), ("min", minimum), ("max", maximum))
        }
        stats["count"] = [count]
        temporary_video.replace(destination)
        mse = float(np.mean(squared_errors))
        record = {
            "episode": episode,
            "key": key,
            "source_identity": identity,
            "output_sha256": sha256(destination),
            "frames": count,
            "max_keyframe_gap": int(max_gap),
            "video_info": video_info,
            "image_stats": stats,
            "sample_rgb_psnr_db": float(10 * np.log10(255**2 / mse)) if mse else None,
            "source_bytes": identity["size"],
            "output_bytes": destination.stat().st_size,
            "seconds": time.perf_counter() - start,
        }
        write_json(record_path, record)
        return record
    finally:
        temporary_video.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="datasets/3dtacdex_assembly/lerobot_all")
    parser.add_argument("--destination", default="datasets/3dtacdex_assembly/lerobot_default")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cpus-per-job", type=int, default=3)
    parser.add_argument("--report-dir", default="experiments/pi05_tacdex/debug/default_encoding")
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    destination = (ROOT / args.destination).resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate dataset directories")
    if args.jobs < 1 or args.cpus_per_job < 1:
        raise ValueError("jobs and cpus-per-job must be positive")
    defaults = {
        name: p.default
        for name, p in inspect.signature(encode_video_frames).parameters.items()
        if name in ("vcodec", "pix_fmt", "g", "crf", "fast_decode")
    }
    if defaults["g"] != 2:
        raise ValueError(f"Expected installed LeRobot GOP default 2, got {defaults}")
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    identity = {
        "source": str(source),
        "source_info_sha256": sha256(source / "meta/info.json"),
        "encoder": "lerobot.common.datasets.video_utils.encode_video_frames",
        "defaults": defaults,
    }
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists():
        manifest = json.loads((destination / "meta/video_reencoding.json").read_text())
        if manifest["identity"] == identity and manifest["status"] == "complete":
            print(f"Already complete: {destination}", flush=True)
            return
        raise FileExistsError(destination)
    staging.mkdir(parents=True, exist_ok=True)
    identity_path = staging / "reencoding_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Staging dataset belongs to a different conversion")
    write_json(identity_path, identity)
    report_dir.mkdir(parents=True, exist_ok=True)
    progress = staging / ".video_progress"
    progress.mkdir(exist_ok=True)
    jobs = []
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for episode in episodes:
        index = episode["episode_index"]
        for key in cameras:
            relative = info["video_path"].format(
                episode_chunk=index // info["chunks_size"], episode_index=index, video_key=key
            )
            jobs.append(
                (
                    str(source / relative),
                    str(staging / relative),
                    info["fps"],
                    episode["length"],
                    key,
                    index,
                    str(progress),
                )
            )
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(args.cpus_per_job,),
    ) as pool:
        futures = [pool.submit(convert_video, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            summary = {
                "completed": len(records),
                "total": len(jobs),
                "elapsed_seconds": time.perf_counter() - start,
                "last_episode": record["episode"],
                "last_camera": record["key"],
            }
            write_json(report_dir / "progress.json", summary)
            print(json.dumps(summary), flush=True)
    shutil.copytree(source / "data", staging / "data", dirs_exist_ok=True)
    shutil.copytree(source / "meta", staging / "meta", dirs_exist_ok=True)
    for record in records:
        info["features"][record["key"]]["info"] = record["video_info"]
    write_json(staging / "meta/info.json", info)
    episode_stats = [json.loads(line) for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines()]
    by_episode = {row["episode_index"]: row for row in episode_stats}
    for record in records:
        by_episode[record["episode"]]["stats"][record["key"]] = record["image_stats"]
    (staging / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_stats))
    # The original manifest describes historical conversion, not the new video encoding.
    original_manifest = staging / "meta/assembly_conversion.json"
    if original_manifest.exists():
        original_manifest.rename(staging / "meta/source_assembly_conversion.json")
    for path in (source / "data").rglob("*.parquet"):
        if sha256(path) != sha256(staging / path.relative_to(source)):
            raise ValueError(f"Vector parquet changed: {path}")
    report = {
        "identity": identity,
        "status": "complete",
        "videos": len(records),
        "frames": info["total_frames"],
        "elapsed_seconds": time.perf_counter() - start,
        "resolution_and_timestamps_preserved": True,
        "vector_parquets_sha256_verified": True,
        "image_stats_spatial_stride": 8,
        "records": sorted(records, key=lambda r: (r["episode"], r["key"])),
    }
    write_json(staging / "meta/video_reencoding.json", report)
    staging.rename(destination)
    write_json(report_dir / "conversion.json", report)
    print(f"Complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
