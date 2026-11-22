"""CPU tests for the 3DTacDex PaXini -> LingBot-Touch magnetic adapter and single-hand encoding."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.lingbot_touch_flax import LingBotTouchConfig
from openpi.models.lingbot_touch_flax import encode_single_hand
from openpi.models.lingbot_touch_flax import init_random_params
from openpi.models.lingbot_touch_flax import load_flax_bundle
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_lingbot import PAXINI_TIP_TAXELS_MM
from openpi.trlt.tacdex_lingbot import TACDEX_TACTILE_AREAS
from openpi.trlt.tacdex_lingbot import PaxiniLingBotConfig
from openpi.trlt.tacdex_lingbot import canonical_geometry
from openpi.trlt.tacdex_lingbot import causal_windows
from openpi.trlt.tacdex_lingbot import grouped_split
from openpi.trlt.tacdex_lingbot import load_oxt_task
from openpi.trlt.tacdex_lingbot import prepare_signal

SMALL = LingBotTouchConfig(embed_dim=40, depth=3, num_heads=2, output_layers=(0, 2))


@pytest.mark.parametrize("pads", ["tip", "tip_pulp"])
def test_geometry_shapes_coverage_and_symmetry(pads):
    config = PaxiniLingBotConfig(pads=pads)
    pos, coverage = canonical_geometry(config)
    m = 15 if pads == "tip" else 30
    assert pos.shape == (5, m, 3)
    assert pos.dtype == np.float32
    assert coverage.shape == (5, m)
    assert coverage[:4].all()
    assert not coverage[4].any()
    assert np.all(pos[4] == 0)
    np.testing.assert_allclose(pos[0], pos[3])  # identical physical pad on every finger
    tip = pos[1, :15]
    np.testing.assert_allclose(tip[:5, 0], -tip[10:, 0])  # lateral columns mirror
    assert tip[9, 1] == tip[:, 1].max()  # distal dome taxel is most distal (+y)
    expected = (PAXINI_TIP_TAXELS_MM - np.array(config.origin_mm)) / config.radius_mm
    np.testing.assert_allclose(tip, expected.astype(np.float32), rtol=1e-6)
    if pads == "tip_pulp":
        assert pos[1, 15:, 1].max() < pos[1, :15, 1].min()  # pulp proximal of tip


def test_prepare_signal_routing_scaling_and_clipping():
    config = PaxiniLingBotConfig(scales_xyz=(10.0, 20.0, 30.0))
    raw = np.zeros((2, 8, 15, 3))
    raw[0, 0, 3] = [5.0, -40.0, 15.0]  # thumb tip taxel 3
    raw[0, 1, 3] = [9.0, 9.0, 9.0]  # thumb pulp: ignored in tip mode
    raw[0, 6, 14] = [0.0, 0.0, 90.0]  # ring tip, Z clipped high
    raw[1, 2, 0] = [0.0, 0.0, -3.0]  # index tip, negative Z clipped to 0
    signal, stats = prepare_signal(raw, config)
    assert signal.shape == (2, 5, 15, 3)
    assert signal.dtype == np.float32
    np.testing.assert_allclose(signal[0, 0, 3], [0.5, -1.0, 0.5])
    np.testing.assert_allclose(signal[0, 3, 14], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(signal[1, 1, 0], [0.0, 0.0, 0.0])
    assert np.count_nonzero(signal) == 4
    assert not signal[:, 4].any()
    assert signal[..., :2].min() >= -1
    assert signal[..., 2].min() >= 0
    assert signal[..., 2].max() <= 1
    assert stats.frac_z_clipped_high > 0
    assert stats.frac_z_negative_clipped > 0
    assert stats.frac_xy_clipped > 0

    both, _ = prepare_signal(raw, PaxiniLingBotConfig(pads="tip_pulp", scales_xyz=(10.0, 20.0, 30.0)))
    assert both.shape == (2, 5, 30, 3)
    np.testing.assert_allclose(both[0, 0, :15], signal[0, 0])
    np.testing.assert_allclose(both[0, 0, 15 + 3], [0.9, 0.45, 0.3], rtol=1e-6)

    flipped, _ = prepare_signal(raw, PaxiniLingBotConfig(scales_xyz=(10.0, 20.0, 30.0), axis_signs=(-1, 1, 1)))
    np.testing.assert_allclose(flipped[0, 0, 3], [-0.5, -1.0, 0.5])
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_signal(np.full((1, 8, 15, 3), np.nan), config)


def test_causal_windows_respect_segments():
    anchors, windows, padded, segments = causal_windows([3, 7], frames=4)
    np.testing.assert_array_equal(anchors, np.arange(7))
    np.testing.assert_array_equal(windows[0], [0, 0, 0, 0])
    np.testing.assert_array_equal(windows[2], [0, 0, 1, 2])
    np.testing.assert_array_equal(windows[3], [3, 3, 3, 3])  # never reaches back into segment 0
    np.testing.assert_array_equal(windows[6], [3, 4, 5, 6])
    np.testing.assert_array_equal(padded, [3, 2, 1, 3, 2, 1, 0])
    np.testing.assert_array_equal(segments, [0, 0, 0, 1, 1, 1, 1])
    assert np.all(windows <= anchors[:, None])


def test_grouped_split_is_disjoint_and_deterministic():
    groups = np.repeat(np.array([f"g{i}" for i in range(10)]), 5)
    a, b = grouped_split(groups, seed=3), grouped_split(groups, seed=3)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert (a["train"].astype(int) + a["validation"] + a["test"] == 1).all()
    assert not set(groups[a["train"]]) & set(groups[a["test"]])


def test_single_hand_encoding_shapes_and_padding_independence():
    adapter = PaxiniLingBotConfig(frames=4)
    pos, coverage = canonical_geometry(adapter)
    params = init_random_params(SMALL, seed=0)
    rng = np.random.default_rng(0)
    raw = rng.integers(-20, 40, size=(3, 4, 8, 15, 3)).astype(np.float64)
    signal, _ = prepare_signal(raw, adapter)
    latent, time_id, finger_id, tok_pos = encode_single_hand(
        params, SMALL, jnp.asarray(signal), jnp.asarray(pos), coverage, modality="magnetic"
    )
    assert latent.shape == (3, 2 * 60, SMALL.target_dim)
    np.testing.assert_array_equal(np.asarray(time_id), np.repeat([0, 1], 60))
    np.testing.assert_array_equal(np.asarray(finger_id), np.tile(np.repeat(np.arange(4), 15), 2))
    np.testing.assert_allclose(np.asarray(tok_pos)[0, :15], pos[0])
    assert np.isfinite(np.asarray(latent)).all()

    # Uncovered pinky slots must not influence any token.
    noisy = signal.copy()
    noisy[:, :, 4] = 0.7
    latent2, *_ = encode_single_hand(params, SMALL, jnp.asarray(noisy), jnp.asarray(pos), coverage, modality="magnetic")
    np.testing.assert_array_equal(np.asarray(latent), np.asarray(latent2))

    # Real contact changes tokens (full attention mixes all fingers).
    touched = signal.copy()
    touched[:, :, 1, 7, 2] = 1.0
    latent3, *_ = encode_single_hand(
        params, SMALL, jnp.asarray(touched), jnp.asarray(pos), coverage, modality="magnetic"
    )
    assert not np.allclose(np.asarray(latent3), np.asarray(latent))


def _write_fake_oxt(path: Path, ends=(4, 9, 15)):
    zarr = pytest.importorskip("zarr")
    frames = ends[-1]
    rng = np.random.default_rng(1)
    store_cls = getattr(zarr.storage, "ZipStore", None) or zarr.ZipStore
    store = store_cls(str(path), mode="w")
    root = zarr.open_group(store=store, path="fake.zarr", mode="w")
    tactile = rng.integers(0, 5, size=(frames, 8, 15, 3)).astype(np.float64)
    root.create_dataset("data/right_tactile_data_fingers", data=tactile)
    root.create_dataset(
        "data/right_tactile_area_fingers", data=np.tile(np.array(TACDEX_TACTILE_AREAS, np.int32), (frames, 1))
    )
    root.create_dataset("data/right_hand_joints", data=rng.normal(size=(frames, 16)).astype(np.float32))
    root.create_dataset("data/right_arm_joints", data=rng.normal(size=(frames, 6)))
    root.create_dataset("data/timestamps", data=np.arange(frames) * 0.2)
    root.create_dataset("meta/episode_ends", data=np.asarray(ends, np.int64))
    store.close()
    return tactile


def test_load_oxt_task_from_zip(tmp_path):
    tactile = _write_fake_oxt(tmp_path / "fake.zarr.zip")
    task = load_oxt_task(tmp_path / "fake.zarr.zip")
    assert task.group == "fake.zarr"
    assert task.frames == 15
    np.testing.assert_array_equal(task.tactile, tactile)
    np.testing.assert_array_equal(task.segment_ids(), np.repeat([0, 1, 2], [4, 5, 6]))
    assert task.hand_command is None
    assert len(set(task.trajectory_id.tolist())) == 3  # falls back to segment ids


def test_cli_inspect_extract_probe_random_control(tmp_path, monkeypatch):
    import importlib.util

    _write_fake_oxt(tmp_path / "fake.zarr.zip", ends=tuple(range(20, 20 * 31, 20)))
    spec = importlib.util.spec_from_file_location("lingbot_tacdex", root_path("scripts/lingbot_tacdex.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr("openpi.models.lingbot_touch_flax.LingBotTouchConfig.tiny", classmethod(lambda cls: SMALL))
    zarr_path = str(tmp_path / "fake.zarr.zip")
    cli.main(["inspect", "--zarr", zarr_path, "--output", str(tmp_path / "inspect.json")])
    assert json.loads((tmp_path / "inspect.json").read_text())["anchors"] == 600
    for seed in (0, 1):
        cli.main(
            ["extract", "--zarr", zarr_path, "--random-init-seed", str(seed), "--tag", f"r{seed}",
             "--output-dir", str(tmp_path / f"r{seed}"), "--batch-size", "128", "--save-last-tokens"]
        )  # fmt: skip
    data = np.load(tmp_path / "r0" / "features.npz")
    assert data["finger_mean"].shape == (600, 4, SMALL.target_dim)
    assert data["last_tubelet_tokens"].shape == (600, 4, 15, SMALL.target_dim)
    cli.main(["probe", "--features", str(tmp_path / "r0"), str(tmp_path / "r1"), "--output-dir", str(tmp_path / "rep")])
    report = json.loads((tmp_path / "rep" / "probe.json").read_text())
    assert "tip_normal_force_sum" in report["results"]
    assert report["results"]["tip_normal_force_sum"]["raw_last_frame"]["test_pooled_r2"] > 0.99


BUNDLE = root_path("checkpoints/lingbot_touch/tiny_flax_f23969ccd4b6bba2163e69cc05a577455a075fe9")


@pytest.mark.skipif(not (BUNDLE / "params.msgpack").exists(), reason="converted Tiny bundle not present")
def test_real_tiny_bundle_encodes_paxini_window():
    config, params, manifest = load_flax_bundle(BUNDLE)
    assert manifest["parameter_count"] == 5455104
    adapter = PaxiniLingBotConfig()
    pos, coverage = canonical_geometry(adapter)
    raw = np.zeros((2, 2, 8, 15, 3))
    raw[1, :, 2, 7] = [2.0, -3.0, 20.0]
    signal, _ = prepare_signal(raw, adapter)
    latent, *_ = encode_single_hand(
        params, config, jnp.asarray(signal), jnp.asarray(pos), coverage, modality="magnetic"
    )
    latent = np.asarray(latent)
    assert latent.shape == (2, 60, 768)
    assert np.isfinite(latent).all()
    assert not np.allclose(latent[0], latent[1])
