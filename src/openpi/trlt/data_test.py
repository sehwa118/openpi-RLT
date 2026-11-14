"""Host-only checks of selection boundaries, split, train stats and time semantics."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from openpi.trlt.config import SENSORS
from openpi.trlt.config import TemporalConfig
from openpi.trlt.data import _seal
from openpi.trlt.data import _write
from openpi.trlt.data import build_tactile_window
from openpi.trlt.data import compute_train_normalization
from openpi.trlt.data import make_episode_splits
from openpi.trlt.data import plan_download
from openpi.trlt.data import read_episode


def _manifest():
    rows = []
    for obj, count in (
        ("a", 28),
        ("b", 23),
        ("c", 20),
        ("d", 18),
        ("e", 16),
        ("f", 14),
        ("g", 13),
        ("h", 12),
        ("i", 12),
        ("j", 10),
        ("k", 7),
    ):
        for _ in range(count):
            rows.append({"episode_index": len(rows), "object": obj, "length": 10, "motor_primitive": "squeeze"})
    return _seal({"revision": "fixed_sha", "episodes": rows})


def test_split_exact_stratified_and_deterministic():
    manifest = _manifest()
    split = make_episode_splits(manifest)
    assert split == make_episode_splits(manifest)
    assert list(map(len, split["splits"].values())) == [121, 26, 26]
    ids = [eid for part in split["splits"].values() for eid in part]
    assert len(ids) == len(set(ids)) == 173
    assert all(all(n >= 1 for n in quota.values()) for quota in split["quotas"].values())
    assert make_episode_splits(manifest, split_seed=43)["splits"] != split["splits"]


def test_history_is_causal_and_padding_and_missing_sensor_stay_invalid():
    cfg = replace(TemporalConfig(), tactile_history_frames=3)
    sensor = SENSORS[0]
    values = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
    times = {sensor: np.arange(5) / 30}
    data = {sensor: values}
    past = build_tactile_window(data, times, 1 / 30, cfg, resolution=1)
    assert past["tactile_valid_mask"][:, 0].tolist() == [False, True, True]
    assert not past["tactile_valid_mask"][:, 1:].any()
    changed = values.copy()
    changed[2:] = 999
    replay = build_tactile_window({sensor: changed}, times, 1 / 30, cfg, resolution=1)
    np.testing.assert_array_equal(past["tactile"], replay["tactile"])
    # Black/no-contact frames are valid observations.
    assert past["tactile_valid_mask"][1, 0]
    assert past["tactile"][1, 0].sum() == 0


def test_asynchronous_history_uses_latest_previous_or_invalid():
    cfg = replace(TemporalConfig(), tactile_history_frames=1, max_observation_lag_seconds=0.02)
    sensor = SENSORS[0]
    values = np.array([1, 2], np.float32).reshape(2, 1, 1, 1)
    result = build_tactile_window({sensor: values}, {sensor: np.array([0.09, 0.11])}, 0.10, cfg, resolution=1)
    assert result["tactile"][0, 0, 0, 0, 0] == 1
    result = build_tactile_window({sensor: values}, {sensor: np.array([0.01, 0.11])}, 0.10, cfg, resolution=1)
    assert not result["tactile_valid_mask"].any()


def test_normalization_reads_unique_train_frames_only_and_keeps_action58(tmp_path):
    rows = []
    for eid in range(3):
        row = {
            "episode_index": eid,
            "object": "a",
            "length": 3,
            "motor_primitive": "squeeze",
            "data_path": "data/shared.parquet",
            "dataset_from_index": eid * 3,
            "dataset_to_index": eid * 3 + 3,
        }
        rows.append(row)
    frames = []
    for eid, offset in enumerate((0, 10, 10000)):
        for t in range(3):
            frames.append(  # noqa: PERF401 -- small readable fixture
                {
                    "episode_index": eid,
                    "frame_index": t,
                    "timestamp": t / 30,
                    "index": eid * 3 + t,
                    "observation.state": [float(offset + t)] * 58,
                    "action": [float(offset + t + 5)] * 58,
                    "observation.tactile_force": [777.0] * 60,
                }
            )
    path = tmp_path / "source/data/shared.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(frames).write_parquet(path)
    subset = _seal({"revision": "fixed", "episodes": rows})
    _write(tmp_path / "subset_manifest.json", subset)
    split = _seal({"subset_hash": subset["hash"], "splits": {"train": [0], "validation": [1], "test": [2]}})
    _write(tmp_path / "splits.json", split)
    norm = compute_train_normalization(tmp_path)
    assert norm["unique_train_frames"] == 3
    np.testing.assert_allclose(norm["state"]["mean"], 1)
    np.testing.assert_allclose(norm["action"]["mean"], 6)
    assert len(norm["action"]["mean"]) == 58
    ep = read_episode(tmp_path, rows[0])
    np.testing.assert_array_equal(ep["action"][0], np.full(58, 5))
    np.testing.assert_array_equal(ep["observation.tactile_force"][0], np.full(60, 777))
    mean, std = np.array(norm["action"]["mean"]), np.array(norm["action"]["std"])
    np.testing.assert_allclose((ep["action"] - mean) / std * std + mean, ep["action"])


def test_plan_reuses_only_complete_exact_files(tmp_path):
    path = tmp_path / "source/data/a.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    manifest = _seal(
        {
            "revision": "fixed",
            "episode_count": 1,
            "frame_count": 3,
            "object_counts": {"a": 1},
            "files": [{"path": "data/a.parquet", "size": 3}, {"path": "videos/camera/b.mp4", "size": 100}],
        }
    )
    result = plan_download(tmp_path, manifest)
    assert result["existing_bytes"] == 3
    assert result["new_bytes"] == 100
    assert result["video_file_count"] == 1


def test_missing_local_file_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="no automatic download"):
        read_episode(tmp_path, {"data_path": "missing.parquet"})


def test_exact_selection_uses_primitive_and_independent_stream_pointers(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import huggingface_hub

    from openpi.trlt.config import CAMERAS
    from openpi.trlt.data import build_subset_manifest

    metadata = ["meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]
    source = tmp_path / "source"
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    info = {
        "fps": 30,
        "features": {"observation.state": {"shape": [58]}, "action": {"shape": [58]}},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (source / metadata[0]).write_text(json.dumps(info))
    (source / metadata[1]).write_bytes(b"tasks")
    rows = []
    for eid, primitive in enumerate(("squeeze", "press")):
        row = {
            "episode_index": eid,
            "motor_primitive": primitive,
            "caption": "same caption for different primitives",
            "object": "a",
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": eid,
            "dataset_from_index": eid * 2,
            "dataset_to_index": eid * 2 + 2,
        }
        for si, stream in enumerate((*CAMERAS, *SENSORS)):
            row.update(
                {
                    f"videos/{stream}/chunk_index": 0,
                    f"videos/{stream}/file_index": si + eid * 100,
                    f"videos/{stream}/from_timestamp": 0.0,
                    f"videos/{stream}/to_timestamp": 2 / 30,
                }
            )
        rows.append(row)
    pl.DataFrame(rows).write_parquet(source / metadata[2])
    all_files = [*metadata, "data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]
    for eid in range(2):
        all_files.extend(
            info["video_path"].format(video_key=stream, chunk_index=0, file_index=si + eid * 100)
            for si, stream in enumerate((*CAMERAS, *SENSORS))
        )
    all_files.append("videos/observation.images.tactile_left_raw_thumb/chunk-000/file-000.mp4")
    calls = []

    class Api:
        def dataset_info(self, repo_id, **kwargs):
            return SimpleNamespace(
                sha="immutable_sha",
                siblings=[SimpleNamespace(rfilename=p, size=100, blob_id="blob") for p in all_files],
            )

    def download(repo_id, path, **kwargs):
        calls.append((path, kwargs["revision"]))
        return str(source / path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest = build_subset_manifest(tmp_path)
    assert manifest["episode_count"] == 1
    paths = {f["path"] for f in manifest["files"]}
    assert len(paths) == len(metadata) + 1 + 13
    assert all("raw_" not in p for p in paths)
    assert "data/chunk-000/file-001.parquet" not in paths
    assert manifest["episodes"][0]["streams"][SENSORS[-1]]["path"].endswith("file-012.mp4")
    assert set(calls) == {(p, "immutable_sha") for p in metadata}


def test_download_cap_fails_before_any_data_download(tmp_path, monkeypatch):
    import huggingface_hub

    from openpi.trlt.data import download_subset

    manifest = _seal(
        {
            "revision": "fixed",
            "episode_count": 1,
            "frame_count": 3,
            "object_counts": {"a": 1},
            "episodes": [],
            "files": [{"path": "data/a.parquet", "size": 100}],
        }
    )
    _write(tmp_path / "subset_manifest.json", manifest)

    def unexpected(*args, **kwargs):
        raise AssertionError("Downloader must not start before cap/space checks")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected)
    with pytest.raises(ValueError, match="exceeds max_download_gb"):
        download_subset(tmp_path, max_download_gb=0)


def test_native_luma_codes_are_preserved():
    import av

    from openpi.trlt.data import _luma

    expected = np.tile(np.arange(240, dtype=np.uint8), (240, 1))
    frame = av.VideoFrame.from_ndarray(expected, format="gray")
    np.testing.assert_array_equal(_luma(frame), expected)


def test_adapter_keeps_commanded_action_and_masks_tail(tmp_path, monkeypatch):
    from openpi.trlt.data import TrexDataset

    row = {
        "episode_index": 7,
        "object": "a",
        "caption": "squeeze",
        "length": 3,
        "data_path": "data/shared.parquet",
        "dataset_from_index": 0,
        "dataset_to_index": 3,
    }
    subset = _seal({"revision": "fixed", "episodes": [row]})
    _write(tmp_path / "subset_manifest.json", subset)
    split = _seal({"subset_hash": subset["hash"], "splits": {"train": [7], "validation": [], "test": []}})
    _write(tmp_path / "splits.json", split)
    norm = _seal(
        {
            "split_hash": split["hash"],
            "state": {"mean": [0] * 58, "std": [1] * 58},
            "action": {"mean": [0] * 58, "std": [1] * 58},
        }
    )
    path = tmp_path / "source/data/shared.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "episode_index": [7] * 3,
            "frame_index": [0, 1, 2],
            "index": [0, 1, 2],
            "timestamp": [0.0, 1 / 30, 2 / 30],
            "observation.state": [[0.0] * 58, [1.0] * 58, [2.0] * 58],
            "action": [[100.0] * 58, [101.0] * 58, [102.0] * 58],
        }
    ).write_parquet(path)
    cfg = replace(TemporalConfig(), query_stride_frames=1)
    dataset = TrexDataset(tmp_path, normalization=norm, temporal=cfg)
    monkeypatch.setattr(dataset, "_tactile_episode", lambda episode_id: ({}, {}))
    first = dataset[0]
    last = dataset[2]
    np.testing.assert_array_equal(first["action_target"][0], np.full(58, 100))
    np.testing.assert_array_equal(last["action_target"][0], np.full(58, 102))
    assert last["action_valid_mask"].tolist() == [True, False, False, False, False, False]
    assert last["action_target"][1:].sum() == 0
    assert first["anchor_frame_index"] == last["anchor_frame_index"] == 0
    assert last["proprioception"][0] == 2
    assert not first["tactile_valid_mask"].any()


def test_float32_timestamp_roundoff_does_not_shift_history_one_frame():
    cfg = replace(TemporalConfig(), tactile_history_frames=10)
    sensor = SENSORS[0]
    values = np.arange(200, dtype=np.float32).reshape(200, 1, 1, 1)
    query = 153
    result = build_tactile_window(
        {sensor: values}, {sensor: np.arange(200) / 30}, float(np.float32(query / 30)), cfg, resolution=1
    )
    np.testing.assert_array_equal(result["tactile"][:, 0, 0, 0, 0], np.arange(query - 9, query + 1))
    assert np.all(result["tactile_timestamps"][:, 0] <= np.float32(query / 30))


def test_metadata_edit_invalidates_its_own_hash(tmp_path):
    from openpi.trlt.data import _read

    value = _seal({"state": {"mean": [1.0], "std": [2.0]}})
    _write(tmp_path / "normalization.json", value)
    assert _read(tmp_path / "normalization.json") == value
    value["state"]["mean"][0] = 999.0
    _write(tmp_path / "normalization.json", value)
    with pytest.raises(ValueError, match="self hash mismatch"):
        _read(tmp_path / "normalization.json")


def test_visual_cache_rejects_changed_anchor_time_contract(tmp_path):
    from types import SimpleNamespace

    from openpi.trlt.config import CAMERAS
    from openpi.trlt.data import TrexDataset

    subset = _seal(
        {"revision": "fixed", "episodes": [{"episode_index": 7, "object": "a", "caption": "squeeze", "length": 3}]}
    )
    _write(tmp_path / "subset_manifest.json", subset)
    split = _seal({"subset_hash": subset["hash"], "splits": {"train": [7], "validation": [], "test": []}})
    _write(tmp_path / "splits.json", split)
    metadata = {
        "revision": "fixed",
        "episode_manifest_hash": subset["hash"],
        "refresh_frames": 15,
        "camera_order": CAMERAS,
        "anchor_sampling": {"fps": 30.0, "max_observation_lag_seconds": 0.05},
    }
    cache = SimpleNamespace(manifest={"metadata": metadata}, cache_hash="fixture")
    assert len(TrexDataset(tmp_path, visual_cache=cache)) == 1
    metadata["anchor_sampling"]["fps"] = 60.0
    with pytest.raises(ValueError, match="provenance mismatch"):
        TrexDataset(tmp_path, visual_cache=cache)
    metadata.pop("anchor_sampling")
    with pytest.raises(ValueError, match="provenance mismatch"):
        TrexDataset(tmp_path, visual_cache=cache)


def test_global_query_manifest_is_deterministic_and_episode_bounded(tmp_path):
    from openpi.trlt.data import build_query_manifest

    subset = _manifest()
    split = make_episode_splits(subset)
    manifest = build_query_manifest(subset, split, output_root=tmp_path)
    assert manifest == build_query_manifest(subset, split, output_root=tmp_path)
    assert (tmp_path / (manifest["hash"] + ".json")).exists()
    source = {e["episode_index"]: e for e in subset["episodes"]}
    for name, rows in manifest["queries"].items():
        for eid, query, anchor, valid in rows:
            assert eid in split["splits"][name]
            assert query % 3 == 0
            assert anchor == 15 * (query // 15)
            assert 0 <= anchor <= query < source[eid]["length"]
            assert valid == min(6, source[eid]["length"] - query)
    changed = build_query_manifest(subset, split, replace(TemporalConfig(), query_stride_frames=1))
    assert changed["hash"] != manifest["hash"]


def test_packet_pts_index_validates_interior_count_cadence_and_boundaries():
    from openpi.trlt.data import validate_packet_timestamps

    pts = np.arange(100, dtype=np.float64) / 30
    result = validate_packet_timestamps(pts, 10 / 30, 20 / 30, 10)
    assert result["packet_frame_count"] == 10
    assert result["maximum_cadence_error_seconds"] < 1e-12
    with pytest.raises(ValueError, match="frame count"):
        validate_packet_timestamps(np.delete(pts, 15), 10 / 30, 20 / 30, 10)
    shifted = pts.copy()
    shifted[15] += 0.001
    with pytest.raises(ValueError, match="cadence"):
        validate_packet_timestamps(shifted, 10 / 30, 20 / 30, 10)
    with pytest.raises(ValueError, match="unique"):
        validate_packet_timestamps(np.insert(pts, 15, pts[15]), 10 / 30, 20 / 30, 10)
