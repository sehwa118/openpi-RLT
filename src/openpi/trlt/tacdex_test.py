"""CPU source-contract and real LeRobot reader tests for Assembly preparation."""

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest
import zarr

from openpi.trlt.tacdex_lerobot import convert_group
from openpi.trlt.tacdex_lerobot import validate_conversion
from openpi.trlt.tacdex_source import audit_source
from openpi.trlt.tacdex_source import command_arrays
from openpi.trlt.tacdex_source import selected_segments


@pytest.fixture
def source(tmp_path):
    root = zarr.open_group(str(tmp_path / "source.zarr"), mode="w", zarr_format=2)
    root.create_group("data")
    root.create_group("raw")
    root.create_group("meta")
    n = 7
    random = np.random.default_rng(9)

    def put(key, value):
        value = np.asarray(value)
        root.create_dataset(key, shape=value.shape, dtype=value.dtype, data=value)

    put("data/right_arm_joints", random.normal(size=(n, 6)))
    put("data/right_hand_joints", random.normal(size=(n, 16)).astype(np.float32))
    put("data/right_hand_joints_idx", np.tile(np.arange(16, dtype=np.int32), (n, 1)))
    put("data/right_tactile_area_fingers", np.tile(np.arange(8, dtype=np.int32), (n, 1)))
    put("data/right_wrist_pose", random.normal(size=(n, 6)))
    put("data/right_tactile_data_fingers", random.normal(size=(n, 8, 15, 3)))
    put("data/timestamps", 1700000000 + np.arange(n) * 0.201)
    put("data/source_episode_id", np.array(["a"] * 4 + ["b"] * 3))
    put("data/source_group", np.array(["early"] * 4 + ["task"] * 3))
    put("data/sub_task_instruction", np.array(["Assemble the early object."] * 4 + ["Assemble the task object."] * 3))
    # The real archive distinguishes one-based source frame counters from
    # zero-based positions in the source record list.
    put("data/source_frame_index", [1, 2, 5, 6, 1, 2, 3])
    put("data/source_record_sha256", np.array([str(i).zfill(64) for i in range(n)], dtype="S64"))
    put("meta/episode_ends", [2, 4, 7])
    arm_command = random.normal(size=(n, 6))
    arm_command[2, 0] = np.nan
    put("raw/arm_commanded_joint_position", arm_command)
    hand_command = random.normal(size=(n, 16)).astype(np.float32)
    hand_command[1] = np.nan
    put("raw/hand_commanded_joint_position", hand_command)
    put("raw/hand_command_present", [True, False, True, True, True, True, True])
    for name in ("main", "secondary"):
        put(f"data/camera_{name}_rgb", random.integers(0, 256, size=(n, 16, 24, 3), dtype=np.uint8))
    lineage = []
    for start, end, positions, episode, group, label in [
        (0, 2, [0, 1], "a", "early", "nouse"),
        (2, 4, [4, 5], "a", "early", "nouse"),
        (4, 7, [0, 1, 2], "b", "task", "task"),
    ]:
        lineage.append(
            {
                "output_start": start,
                "output_end": end,
                "source_positions": positions,
                "source_episode_id": episode,
                "source_episode": f"demo/{episode}",
                "source_group": group,
                "source_label": label,
                "source_split": "unspecified",
            }
        )
    root.attrs["lineage"] = lineage
    return root


def test_missing_commands_are_masked_without_state_substitution(source):
    action, valid = command_arrays(source)
    assert valid.shape == (7, 22)
    assert valid[1, :6].all()
    assert not valid[1, 6:].any()
    assert not valid[2, 0]
    assert valid[2, 1:].all()
    assert np.isfinite(action).all()
    assert np.all(action[~valid] == 0)
    np.testing.assert_array_equal(action[0, :6], source["raw/arm_commanded_joint_position"][0])
    assert not np.array_equal(action[0, :6], source["data/right_arm_joints"][0])


def test_audit_retains_nouse_and_original_trajectory_identity(source):
    report = audit_source(source)
    assert report["cohorts"]["all"] == {
        "segments": 3,
        "source_trajectories": 2,
        "frames": 7,
        "complete_action_frames": 5,
    }
    assert report["cohorts"]["paper"]["frames"] == 3
    assert len(selected_segments(source, "all")) == 3
    assert (
        selected_segments(source, "all")[0]["source_episode_id"]
        == selected_segments(source, "all")[1]["source_episode_id"]
    )
    assert report["duplicate_record_rows"] == 0


def test_audit_rejects_bad_lineage_and_time(source):
    lineage = list(source.attrs["lineage"])
    lineage[0]["source_positions"] = [0]
    source.attrs["lineage"] = lineage
    with pytest.raises(ValueError, match="position count"):
        audit_source(source)
    lineage[0]["source_positions"] = [0, 1]
    source.attrs["lineage"] = lineage
    source["data/timestamps"][1] = source["data/timestamps"][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        audit_source(source)


def test_lossless_roundtrip_through_real_lerobot_and_tamper_detection(source, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = tmp_path / "converted"
    manifest = convert_group(source, output, cohort="all", threads=1)
    assert manifest["completed_episodes"] == 3
    assert manifest["completed_frames"] == 7
    assert manifest["hand_joint_oxt_indices"] == list(range(16))
    assert manifest["tactile_area_oxt_indices"] == list(range(8))
    report = validate_conversion(source, output)
    assert report["numeric_rows"] == 7
    assert report["rgb_frames_compared_exactly"] == 14
    assert report["complete_action_frames"] == 5
    assert convert_group(source, output, cohort="all")["identity_hash"] == manifest["identity_hash"]
    with pytest.raises(ValueError, match="provenance differs"):
        convert_group(source, output, cohort="paper")
    path = output / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["action"][0] += 100
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="Numeric/source mismatch"):
        validate_conversion(source, output)


def test_existing_unrelated_output_is_preserved(source, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.json"
    marker.write_text(json.dumps({"keep": True}))
    with pytest.raises(ValueError, match="without Assembly provenance"):
        convert_group(source, output)
    assert json.loads(marker.read_text()) == {"keep": True}


def test_resume_continues_after_committed_episode(source, tmp_path, monkeypatch):
    from openpi.trlt import tacdex_lerobot

    output = tmp_path / "interrupted"
    encode = tacdex_lerobot.encode_segment_videos

    def interrupt_second_episode(group, segment, meta, episode_index, **kwargs):
        if episode_index == 1:
            raise RuntimeError("simulated interruption")
        return encode(group, segment, meta, episode_index, **kwargs)

    monkeypatch.setattr(tacdex_lerobot, "encode_segment_videos", interrupt_second_episode)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        convert_group(source, output, threads=1)
    first_video = output / "videos/chunk-000/observation.images.main/episode_000000.mp4"
    first_mtime = first_video.stat().st_mtime_ns
    with pytest.raises(ValueError, match="use --resume"):
        convert_group(source, output)
    monkeypatch.setattr(tacdex_lerobot, "encode_segment_videos", encode)
    manifest = convert_group(source, output, threads=1, resume=True)
    assert manifest["completed_episodes"] == 3
    assert first_video.stat().st_mtime_ns == first_mtime
    assert validate_conversion(source, output)["numeric_rows"] == 7
