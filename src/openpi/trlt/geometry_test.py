"""CPU-only geometry contracts: FK locality, causal alignment and cache provenance."""

import json

import numpy as np
import pytest

from openpi.trlt.config import digest
from openpi.trlt.data import _seal
from openpi.trlt.data import _write
from openpi.trlt.geometry import DEFAULT_SPEC
from openpi.trlt.geometry import GeometryCache
from openpi.trlt.geometry import HandGeometry
from openpi.trlt.geometry import prepare_geometry
from openpi.trlt.runner import OBSERVATION_FIELDS
from openpi.trlt.runner import _check_provenance
from openpi.trlt.runner import collate


def test_fk_joint_and_hand_locality():
    model = HandGeometry()
    state = np.zeros((3, 58))
    baseline = model.positions(state)
    assert baseline.shape == (3, 10, 3)
    assert np.max(np.abs(baseline)) < 0.3
    state[:, :7] = 100
    state[:, 29:36] = -100
    np.testing.assert_array_equal(model.positions(state), baseline)
    state[1, 14] = 0.5  # left index PIP, source left-hand offset 7 + index 7
    changed = model.positions(state)
    assert not np.allclose(changed[1, 1], baseline[1, 1])
    np.testing.assert_array_equal(changed[1, [0, 2, 3, 4, 5, 6, 7, 8, 9]], baseline[1, [0, 2, 3, 4, 5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(changed[[0, 2]], baseline[[0, 2]])
    state[2, 57] = 0.5
    changed = model.positions(state)
    assert not np.allclose(changed[2, 9], baseline[2, 9])
    np.testing.assert_array_equal(changed[2, :9], baseline[2, :9])


def test_fk_single_hinge_analytic():
    """Last index hinge moves pad about DIP while preserving distance to hinge."""
    from openpi.trlt.geometry import quaternion_matrix

    model = HandGeometry()
    sensor = model.spec["sensors"][1]
    rotation = np.eye(3)
    origin = np.zeros(3)
    for step in sensor["chain"]:
        origin += rotation @ step["pos"]
        rotation = rotation @ quaternion_matrix(step["quat_wxyz"])
    q = np.zeros((2, 58))
    q[1, 15] = np.pi / 2  # left index DIP
    p = model.positions(q)[:, 1]
    offset = np.array(sensor["pad_offset_m"])
    expected = origin + rotation @ np.array([-offset[1], offset[0], offset[2]])
    np.testing.assert_allclose(p[1], expected, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(p - origin, axis=-1), np.linalg.norm(offset), atol=1e-8)


def _fixture(tmp_path, monkeypatch):
    import openpi.trlt.data as data

    row = {"episode_index": 1, "length": 3}
    _write(tmp_path / "subset_manifest.json", _seal({"revision": "fixed", "episodes": [row]}))
    _write(tmp_path / "splits.json", _seal({"splits": {"train": [1], "validation": [], "test": []}}))
    state = np.zeros((3, 58))
    state[:, 14] = [0, 0.5, 1]
    ep = {"timestamp": np.array([0, 1, 2], np.float64) / 30, "observation.state": state}
    monkeypatch.setattr(data, "read_episode", lambda *_: ep)
    prepare_geometry(tmp_path)
    return GeometryCache(tmp_path), ep


def test_cache_alignment_no_future_and_padding(tmp_path, monkeypatch):
    cache, ep = _fixture(tmp_path, monkeypatch)
    stamps = np.broadcast_to(np.array([-0.01, 0.02, 0.04, 0.2])[:, None], (4, 10))
    valid = np.ones((4, 10), bool)
    valid[2, 2] = False
    result = cache.window(1, stamps, valid, 0.025)
    assert not result["geometry_valid_mask"][[0, 3]].any()
    assert not result["geometry_valid_mask"][2, 2]
    np.testing.assert_array_equal(result["tactile_positions"][1], cache.model.positions(ep["observation.state"][0]))
    np.testing.assert_array_equal(
        result["tactile_positions"][2, 1], cache.model.positions(ep["observation.state"][1])[1]
    )
    assert (result["tactile_positions"][~result["geometry_valid_mask"]] == 0).all()
    first = cache.directory.joinpath("episode_000001.npz").stat().st_mtime_ns
    prepare_geometry(tmp_path)
    assert first == cache.directory.joinpath("episode_000001.npz").stat().st_mtime_ns
    with pytest.raises(ValueError, match="selected split"):
        prepare_geometry(tmp_path, episode_ids=[2])


def test_cache_corruption_and_spec_changes_rejected(tmp_path, monkeypatch):
    cache, _ = _fixture(tmp_path, monkeypatch)
    path = cache.directory / "episode_000001.npz"
    with np.load(path) as data:
        values = {k: data[k].copy() for k in data.files}
    values["positions"][0, 0, 0] += 1
    np.savez(path, **values)
    with pytest.raises(ValueError, match="provenance/content"):
        cache.episode(1)
    from openpi.trlt.config import root_path

    spec = json.loads(root_path(DEFAULT_SPEC).read_text())
    spec["sensors"][0]["pad_offset_m"][0] += 0.01
    target = tmp_path / "spec.json"
    target.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="spec hash"):
        HandGeometry(target)
    spec["hash"] = digest({k: v for k, v in spec.items() if k != "hash"})
    target.write_text(json.dumps(spec))
    assert HandGeometry(target).identity != cache.model.identity


def test_geometry_provenance_and_collation():
    for a, b in (
        ({}, {"geometry_hash": "x"}),
        ({"geometry_hash": "x"}, {}),
        ({"geometry_hash": "x"}, {"geometry_hash": "y"}),
    ):
        with pytest.raises(ValueError, match="geometry_hash"):
            _check_provenance(a, b, "test")
    _check_provenance({"geometry_hash": "x"}, {"geometry_hash": "x"}, "test")
    sample = {
        k: np.zeros((1,))
        for k in OBSERVATION_FIELDS
        if k not in ("spatial_valid_mask", "tactile_positions", "geometry_valid_mask")
    }
    assert "tactile_positions" not in collate([sample], OBSERVATION_FIELDS)
    sample.update(tactile_positions=np.zeros((2, 10, 3)), geometry_valid_mask=np.ones((2, 10), bool))
    assert collate([sample], OBSERVATION_FIELDS)["tactile_positions"].shape == (1, 2, 10, 3)


def test_cached_context_geometry_tracks_selected_observation_and_reset():
    from openpi.trlt.cache import CachedContext
    from openpi.trlt.config import TemporalConfig

    context = CachedContext(TemporalConfig(tactile_history_frames=2))
    context.update_visual(
        {"visual_features": np.ones((6, 12)), "visual_valid_mask": np.ones(6, bool)}, episode_id=1, timestamp=0.0
    )
    image = np.zeros((10, 2, 2, 1))
    valid = np.ones(10, bool)
    context.append_tactile(image, valid, episode_id=1, timestamp=0.0, positions=np.ones((10, 3)) * 0.1)
    context.append_tactile(image, valid, episode_id=1, timestamp=1 / 30, positions=np.ones((10, 3)) * 0.2)
    batch = context.encode(lambda b: b, current_state=np.zeros(58), query_time=1 / 30)
    np.testing.assert_allclose(batch["tactile_positions"][0, :, 0, 0], [0.1, 0.2])
    assert batch["geometry_valid_mask"].all()
    context.append_tactile(image, valid, episode_id=1, timestamp=2 / 30)
    batch = context.encode(lambda b: b, current_state=np.zeros(58), query_time=2 / 30)
    assert batch["geometry_valid_mask"][0, 0].all()
    assert not batch["geometry_valid_mask"][0, 1].any()
    context.reset(2)
    assert not context.geometry_history
    assert not context.geometry_enabled
