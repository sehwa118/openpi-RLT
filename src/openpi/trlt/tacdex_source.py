"""Pinned Assembly download and CPU-only source audit; no training side effects."""

from collections import Counter
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.trlt.config import root_path

REPO_ID = "kingchou007/3dtacdex"
REVISION = "c58157b2acea2b8bf7fea2767e2c00c609b8c579"
ARCHIVE = "assembly/assembly.zarr.zip"
ARCHIVE_BYTES = 81101454421
ARCHIVE_SHA256 = "fbe888bbce81553f9afd05c50430656c66de992109749bfa3a26124a4d0e1652"
DEFAULT_ROOT = "datasets/3dtacdex_assembly"
DEFAULT_REPORT = "experiments/trlt/3dtacdex_assembly/preparation"
CAMERAS = {"main": "data/camera_main_rgb", "secondary": "data/camera_secondary_rgb"}
PAD_NAMES = tuple(f"{finger}_{pad}" for finger in ("thumb", "index", "middle", "ring") for pad in ("tip", "pulp"))
STATE_NAMES = tuple(f"arm_joint_{i}" for i in range(6)) + tuple(f"leap_source_joint_{i}" for i in range(16))


def utc_now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def archive_path(data_root=DEFAULT_ROOT):
    return root_path(data_root) / "source" / ARCHIVE


def verify_archive(data_root=DEFAULT_ROOT, *, rehash=False):
    """Reuse a checksum only for the identical local file identity, size and mtime."""
    path = archive_path(data_root)
    stat = path.stat()
    if stat.st_size != ARCHIVE_BYTES:
        raise ValueError(f"Incomplete Assembly archive: {stat.st_size} != {ARCHIVE_BYTES}")
    identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    record = path.parents[1] / "checksum.json"
    if record.exists() and not rehash:
        previous = json.loads(record.read_text())
        if previous.get("file_identity") == identity and previous.get("sha256") == ARCHIVE_SHA256:
            return previous
    print("Verifying Assembly SHA256 (81.1 GB)...", flush=True)
    actual = sha256_file(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"Assembly SHA256 mismatch: {actual}")
    result = {"path": str(path), "sha256": actual, "file_identity": identity, "verified_at": utc_now()}
    write_json(record, result)
    return result


def download_plan(data_root=DEFAULT_ROOT):
    root = root_path(data_root)
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": ["README.md", ARCHIVE],
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "available_bytes": shutil.disk_usage(ancestor).free,
        "data_root": str(root),
        "status": "plan-only",
    }


def download_assembly(data_root=DEFAULT_ROOT):
    from filelock import FileLock
    from huggingface_hub import get_hf_file_metadata
    from huggingface_hub import hf_hub_download
    from huggingface_hub import hf_hub_url

    from openpi.trlt.tacdex_download import download_ranges

    root = root_path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".download.lock")):
        plan = download_plan(root)
        if not archive_path(root).exists() and plan["available_bytes"] < ARCHIVE_BYTES * 1.05:
            raise ValueError("Insufficient disk space for Assembly download")
        hf_hub_download(REPO_ID, "README.md", repo_type="dataset", revision=REVISION, local_dir=root / "source")
        if not archive_path(root).exists():

            def url_provider():
                metadata = get_hf_file_metadata(hf_hub_url(REPO_ID, ARCHIVE, repo_type="dataset", revision=REVISION))
                if metadata.size != ARCHIVE_BYTES or metadata.etag != ARCHIVE_SHA256:
                    raise ValueError("Remote Assembly file identity differs from pinned source")
                return metadata.location

            download_ranges(archive_path(root), url_provider, total_bytes=ARCHIVE_BYTES, expected_sha256=ARCHIVE_SHA256)
        checksum = verify_archive(root)
        result = {**plan, "status": "complete", "completed_at": utc_now(), "checksum": checksum}
        write_json(root / "download_manifest.json", result)
        return result


@contextmanager
def open_assembly(path):
    """Zarr 3 reads this Zarr v2 store without changing the existing environment."""
    import zarr

    with zarr.storage.ZipStore(Path(path), mode="r") as store:
        yield zarr.open_group(store=store, path="assembly.zarr", mode="r")


def command_arrays(group):
    arm = np.asarray(group["raw/arm_commanded_joint_position"][:], np.float64)
    hand = np.asarray(group["raw/hand_commanded_joint_position"][:], np.float64)
    present = np.asarray(group["raw/hand_command_present"][:], bool)
    if arm.ndim != 2 or arm.shape[1] != 6 or hand.shape != (len(arm), 16) or present.shape != (len(arm),):
        raise ValueError("Unexpected command shape")
    raw = np.concatenate((arm, hand), axis=1)
    valid = np.isfinite(raw)
    valid[:, 6:] &= present[:, None]
    # Zero is storage padding only. A false mask never denotes a training label.
    return np.where(valid, raw, 0), valid


def source_segments(group):
    ends = np.asarray(group["meta/episode_ends"][:], np.int64)
    n = group["data/timestamps"].shape[0]
    if ends.ndim != 1 or not len(ends) or np.any(np.diff(np.r_[0, ends]) <= 0) or ends[-1] != n:
        raise ValueError("Invalid segment boundaries")
    lineage = list(group.attrs.get("lineage", []))
    by_range = {(x["output_start"], x["output_end"]): x for x in lineage}
    if len(by_range) != len(ends):
        raise ValueError("Lineage must identify each segment exactly once")
    result = []
    for segment_id, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends, strict=True)):
        key = (int(start), int(end))
        if key not in by_range:
            raise ValueError(f"Missing lineage for {key}")
        item = dict(by_range[key])
        if len(item["source_positions"]) != end - start:
            raise ValueError("Lineage position count differs from segment length")
        positions = np.asarray(item["source_positions"])
        if positions.dtype.kind not in "iu" or np.any(positions < 0) or np.any(np.diff(positions) <= 0):
            raise ValueError("Lineage positions must be nonnegative increasing integers")
        result.append({**item, "segment_id": segment_id, "length": int(end - start)})
    return result


def selected_segments(group, cohort="all"):
    if cohort not in ("paper", "all"):
        raise ValueError("cohort must be paper or all")
    return [x for x in source_segments(group) if cohort == "all" or x["source_label"] != "nouse"]


def audit_source(group):
    """Audit all numeric/source rows; RGB is checked by the conversion validator."""
    segments = source_segments(group)
    n = int(group["data/timestamps"].shape[0])
    schema = {}
    for section in ("data", "raw", "meta"):
        for name, array in group[section].arrays():
            schema[f"{section}/{name}"] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    expected = {
        "data/right_arm_joints": (n, 6),
        "data/right_hand_joints": (n, 16),
        "data/right_wrist_pose": (n, 6),
        "data/right_tactile_data_fingers": (n, 8, 15, 3),
    }
    numeric = {}
    for key, shape in expected.items():
        array = np.asarray(group[key][:])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Nonfinite or wrong-shaped observations: {key}")
        numeric[key] = {"min": float(array.min()), "max": float(array.max())}
    action, valid = command_arrays(group)
    ids = np.asarray(group["data/source_episode_id"][:]).astype(str)
    groups = np.asarray(group["data/source_group"][:]).astype(str)
    source_frames = np.asarray(group["data/source_frame_index"][:])
    timestamps = np.asarray(group["data/timestamps"][:], np.float64)
    hashes = np.asarray(group["data/source_record_sha256"][:]).astype(str)
    differences = []
    per_segment = []
    for segment in segments:
        start, end = segment["output_start"], segment["output_end"]
        sl = slice(start, end)
        if set(ids[sl]) != {segment["source_episode_id"]} or set(groups[sl]) != {segment["source_group"]}:
            raise ValueError("Frame provenance disagrees with lineage")
        # Lineage positions index the source record list, whereas frame indices
        # are original filenames/counters (often one-based). Preserve both.
        if np.any(np.diff(source_frames[sl]) <= 0):
            raise ValueError("Source frame indices must increase within a segment")
        dt = np.diff(timestamps[sl])
        if not np.isfinite(timestamps[sl]).all() or np.any(dt <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing within a segment")
        differences.extend(dt.tolist())
        complete = valid[sl].all(axis=1)
        per_segment.append(
            {
                **segment,
                "complete_action_frames": int(complete.sum()),
                "all_actions_valid": bool(complete.all()),
                "paper_cohort": segment["source_label"] != "nouse",
            }
        )
    dt = np.asarray(differences)
    cohorts = {}
    for cohort in ("all", "paper"):
        selected = [x for x in per_segment if cohort == "all" or x["paper_cohort"]]
        cohorts[cohort] = {
            "segments": len(selected),
            "source_trajectories": len({x["source_episode_id"] for x in selected}),
            "frames": sum(x["length"] for x in selected),
            "complete_action_frames": sum(x["complete_action_frames"] for x in selected),
        }
    group_summary = {}
    for name in sorted({s["source_group"] for s in per_segment}):
        items = [s for s in per_segment if s["source_group"] == name]
        group_summary[name] = {
            "segments": len(items),
            "source_trajectories": len({s["source_episode_id"] for s in items}),
            "frames": sum(s["length"] for s in items),
            "complete_action_frames": sum(s["complete_action_frames"] for s in items),
            "source_labels": sorted({s["source_label"] for s in items}),
        }
    return {
        "status": "pass",
        "created_at": utc_now(),
        "schema": schema,
        "cohorts": cohorts,
        "source_group_segments": dict(Counter(x["source_group"] for x in segments)),
        "source_groups": group_summary,
        "numeric_ranges": numeric,
        "unique_record_hashes": len(set(hashes)),
        "duplicate_record_rows": len(hashes) - len(set(hashes)),
        "timing": {
            "nominal_fps": 5,
            "dt_min": float(dt.min()) if len(dt) else None,
            "dt_median": float(np.median(dt)) if len(dt) else None,
            "dt_max": float(dt.max()) if len(dt) else None,
            "max_nominal_step_error": float(np.max(np.abs(dt - 0.2))) if len(dt) else None,
        },
        "action": {
            "source": ["raw/arm_commanded_joint_position", "raw/hand_commanded_joint_position"],
            "shape": list(action.shape),
            "valid_elements": int(valid.sum()),
            "complete_frames": int(valid.all(axis=1).sum()),
            "hand_presence_mask": "raw/hand_command_present",
            "valid_min": float(action[valid].min()) if valid.any() else None,
            "valid_max": float(action[valid].max()) if valid.any() else None,
            "semantics": "Recorded commanded joint positions, source order/sign/units unchanged; not observed state.",
            "units": "Source-native; archive does not explicitly annotate raw command units.",
            "alignment": "Same source record; independent command/sensor timestamps and latency unprovided.",
        },
        "segments": per_segment,
        "limitations": [
            "No verified physical-object/session identity or per-episode success labels.",
            "Record-hash uniqueness does not establish absence of similar trajectories.",
            "No train/validation/test split or normalization fitted at this preparation stage.",
        ],
    }


def audit_assembly(data_root=DEFAULT_ROOT, report_dir=DEFAULT_REPORT):
    checksum = verify_archive(data_root)
    with open_assembly(archive_path(data_root)) as group:
        result = audit_source(group)
        result["source_attributes"] = dict(group.attrs)
    result["source"] = {"repo_id": REPO_ID, "revision": REVISION, "sha256": checksum["sha256"]}
    write_json(root_path(report_dir) / "source_audit.json", result)
    return {k: v for k, v in result.items() if k not in ("source_attributes", "segments", "schema")}
