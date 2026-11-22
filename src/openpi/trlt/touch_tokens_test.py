"""CPU tests: touch-token caches, latent join, touch probe training/evaluation and H0 fairness."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.trlt_actor import PartAttentionActionProbe
from openpi.models.trlt_actor import build_action_probe
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import probe_config_dict
from openpi.trlt.config import probe_variant_name
from openpi.trlt.config import root_path
from openpi.trlt.runner import LatentDataset
from openpi.trlt.runner import evaluate_action_probe
from openpi.trlt.runner import file_hash
from openpi.trlt.runner import train_action_probe
from openpi.trlt.tacdex_lingbot import PaxiniLingBotConfig
from openpi.trlt.touch_tokens import TouchLatentDataset
from openpi.trlt.touch_tokens import build_touch_cache
from openpi.trlt.touch_tokens import query_windows

TEMPORAL = TemporalConfig(
    fps=5,
    visual_refresh_frames=3,
    query_stride_frames=1,
    tactile_history_frames=3,
    tactile_history_stride_frames=1,
    action_horizon_frames=3,
    max_observation_lag_seconds=0.3,
)
PROV = {
    "dataset_kind": "3dtacdex_assembly",
    "split_hash": "s",
    "query_manifest_hash": "q",
    "episode_manifest_hash": "e",
    "temporal": {"fps": 5},
}


def _contiguous(ep, start, end, temporal):
    from openpi.trlt.tacdex_data import contiguous

    return contiguous(ep, start, end, temporal)


def _episodes(seed=0):
    rng = np.random.default_rng(seed)
    eps = {}
    for eid, length in ((0, 9), (1, 7)):
        frames = np.arange(length) + 100 * eid
        if eid == 1:
            frames[3:] += 1  # a frame gap: rows >= 3 are not contiguous with rows < 3
        tactile = rng.integers(0, 6, size=(length, 8, 15, 3)).astype(np.float64)
        valid = np.ones((length, 8), bool)
        valid[5, 2] = False  # index tip invalid at row 5 of episode 0
        eps[eid] = {
            "observation.tactile": tactile,
            "observation.tactile_valid_mask": valid,
            "source.frame_index": frames,
            "source.elapsed_seconds": np.arange(length) * 0.2 + 10 * eid,
            "action": np.zeros((length, 22)),
        }
    return eps


def _queries(eps):
    return [(eid, q) for eid, ep in eps.items() for q in range(len(ep["action"]))]


def test_query_windows_use_dataset_contiguity():
    eps = _episodes()
    rows, rep = query_windows(eps[0], 0, 4, _contiguous, TEMPORAL)
    assert rows == [0, 0, 0, 0]
    assert rep == 3
    rows, rep = query_windows(eps[0], 6, 4, _contiguous, TEMPORAL)
    assert rows == [3, 4, 5, 6]
    assert rep == 0
    rows, rep = query_windows(eps[1], 4, 4, _contiguous, TEMPORAL)  # gap between rows 2 and 3
    assert rows == [3, 3, 3, 4]
    assert rep == 2


@pytest.mark.parametrize(("source", "frames"), [("raw", 2), ("random", 2), ("random", 4)])
def test_build_cache_shapes_validity_and_identity(tmp_path, source, frames):
    eps = _episodes()
    adapter = PaxiniLingBotConfig(frames=frames)
    kwargs = {"random_init_seed": 0} if source == "random" else {}
    manifest = build_touch_cache(
        queries=_queries(eps),
        episode=eps.__getitem__,
        contiguous=_contiguous,
        temporal=TEMPORAL,
        provenance=dict(PROV, split="train"),
        output_dir=tmp_path / "c",
        source=source,
        adapter=adapter,
        batch_size=5,
        **kwargs,
    )
    tokens = np.load(tmp_path / "c" / "tokens.npy")
    valid = np.load(tmp_path / "c" / "valid.npy")
    k = 15 if source == "raw" else 15 * frames // 2
    c = frames * 3 if source == "raw" else 768
    assert tokens.shape == (16, 4, k, c)
    assert tokens.dtype == np.float16
    assert np.isfinite(tokens.astype(np.float32)).all()
    assert valid.shape == (16, 4, k)
    assert not valid[5, 1].any()  # index tip invalid at the query row -> all its tokens masked
    assert valid[5, [0, 2, 3]].all()
    assert manifest["count"] == 16
    if source == "raw":
        # Query row 6 has the invalid row-5 index-tip frame in its history: that frame is zeroed.
        assert not tokens[6, 1, :, :3].any()
    if source == "raw":
        # Token (finger, taxel) = [x,y,z] per frame of the normalized window; finger 1 = index tip (pad 2).
        expected = np.clip(eps[0]["observation.tactile"][[6, 7], 2] / 30.0, -1, 1)  # [T, 15, 3]
        np.testing.assert_allclose(
            tokens[7, 1].astype(np.float32), expected.transpose(1, 0, 2).reshape(15, 6), atol=1e-3
        )
    with pytest.raises(FileExistsError):
        build_touch_cache(
            queries=_queries(eps), episode=eps.__getitem__, contiguous=_contiguous, temporal=TEMPORAL,
            provenance=dict(PROV, split="train"), output_dir=tmp_path / "c", source=source, adapter=adapter,
            **kwargs,
        )  # fmt: skip


def _write_latents(path, eps, split, rng):
    rows = _queries(eps)
    n = len(rows)
    values = {
        "z": rng.normal(size=(n, 3, 8)).astype(np.float32),
        "proprioception": rng.normal(size=(n, 22)).astype(np.float32),
        "action_target": rng.normal(size=(n, 3, 22)).astype(np.float32),
        "action_valid_mask": np.ones((n, 3, 22), bool),
        "visual_age_seconds": np.zeros(n, np.float32),
        "episode_id": np.asarray([e for e, _ in rows], np.int32),
        "segment_id": np.asarray([e for e, _ in rows], np.int32),
        "anchor_frame_index": np.asarray([eps[e]["source.frame_index"][q] for e, q in rows], np.int64),
        "query_frame_index": np.asarray([eps[e]["source.frame_index"][q] for e, q in rows], np.int64),
        "query_timestamp": np.asarray([eps[e]["source.elapsed_seconds"][q] for e, q in rows], np.float32),
        "object": np.asarray(["g"] * n),
    }
    # Shuffle row order: the join must be by key, not by position.
    order = rng.permutation(n)
    values = {k: v[order] for k, v in values.items()}
    path.mkdir(parents=True)
    np.savez_compressed(path / "chunk_000000.npz", **values)
    manifest = {
        "version": 1,
        "provenance": dict(PROV, split=split, frozen=True, model="trlt_self_only"),
        "fingerprint": "f",
        "shards": [{"file": "chunk_000000.npz", "count": n, "sha256": file_hash(path / "chunk_000000.npz")}],
        "count": n,
        "normalization": None,
    }
    manifest["cache_hash"] = digest(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))
    return LatentDataset(path)


def _touch_config(kv="layernorm", tag="lingbot_random"):
    return ProbeConfig(
        hidden_dim=32, depth=1, action_dim=22, horizon=3, batch_size=8, max_epochs=1, patience=None,
        architecture="touch_part_attention", attention_mask_mode="hierarchical", num_heads=4,
        variant_tag=tag, touch_num_heads=4, touch_kv_mode=kv,
    )  # fmt: skip


def _cache(tmp_path, eps, split, source="random"):
    out = tmp_path / f"touch_{source}_{split}"
    build_touch_cache(
        queries=_queries(eps),
        episode=eps.__getitem__,
        contiguous=_contiguous,
        temporal=TEMPORAL,
        provenance=dict(PROV, split=split),
        output_dir=out,
        source=source,
        adapter=PaxiniLingBotConfig(),
        random_init_seed=0 if source == "random" else None,
    )
    return out


def test_touch_latent_join_reorders_fingers_and_checks_provenance(tmp_path):
    eps = _episodes()
    latents = _write_latents(tmp_path / "lat_train", eps, "train", np.random.default_rng(0))
    cache = _cache(tmp_path, eps, "train", source="raw")
    joined = TouchLatentDataset(latents, cache)
    assert joined.finger_order == ("index", "middle", "ring", "thumb")
    tokens = np.load(cache / "tokens.npy")
    for i in range(len(joined)):
        row = joined[i]
        key = (int(row["segment_id"]), int(row["query_frame_index"]))
        j = int(joined.rows[i])
        assert key == (
            int(np.load(cache / "keys.npz")["segment_id"][j]),
            int(np.load(cache / "keys.npz")["query_frame_index"][j]),
        )
        np.testing.assert_array_equal(row["touch_tokens"], tokens[j][[1, 2, 3, 0]].astype(np.float32))
        assert row["touch_valid_mask"].shape == (4, 15)
        assert row["touch_pos_xyz"].shape == (4, 15, 3)
    assert joined.provenance["touch_identity"]
    val_latents = _write_latents(tmp_path / "lat_val", eps, "validation", np.random.default_rng(1))
    with pytest.raises(ValueError, match="split"):
        TouchLatentDataset(val_latents, cache)


def test_touch_probe_initially_equals_part_attention():
    """Zero-initialized touch gates: H1/H2/H3 start as exactly the H0 function (fair comparison)."""
    base = ProbeConfig(
        hidden_dim=32, depth=2, action_dim=22, horizon=3, architecture="part_attention",
        attention_mask_mode="hierarchical", num_heads=4,
    )  # fmt: skip
    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.normal(size=(5, 3, 8)), jnp.float32)
    prop = jnp.asarray(rng.normal(size=(5, 22)), jnp.float32)
    age = jnp.zeros((5,), jnp.float32)
    touch = jnp.asarray(rng.normal(size=(5, 4, 15, 768)), jnp.float32)
    valid = jnp.ones((5, 4, 15), bool)
    pos = jnp.asarray(rng.normal(size=(4, 15, 3)), jnp.float32)
    key = jax.random.PRNGKey(0)
    h0 = PartAttentionActionProbe(base)
    p0 = h0.init(key, z, prop, age)["params"]
    for kv in ("layernorm", "raw_mlp"):
        h1 = build_action_probe(replace(base, architecture="touch_part_attention", touch_kv_mode=kv))
        p1 = h1.init(key, z, prop, age, touch, valid, touch_pos_xyz=pos)["params"]
        shared = {k: v for k, v in p1.items() if not k.startswith("touch_ca_")}
        assert set(shared) == set(p0)
        out0 = h0.apply({"params": p0}, z, prop, age)
        out1 = h1.apply({"params": {**p1, **p0}}, z, prop, age, touch, valid, touch_pos_xyz=pos)
        np.testing.assert_allclose(np.asarray(out1), np.asarray(out0), rtol=1e-6, atol=1e-6)


def test_probe_config_identity_unchanged_for_existing_configs():
    for name in ("probe_mlp", "probe_part_attn_hierarchical", "probe_resmlp"):
        raw = json.loads(root_path(f"configs/trlt/3dtacdex_assembly/attention_actor/{name}.json").read_text())
        value = probe_config_dict(ProbeConfig(**raw["probe"]))
        assert not any(key.startswith("touch_") for key in value)
    raw = json.loads(root_path("configs/trlt/3dtacdex_assembly/lingbot_touch/probe_touch_raw_taxel.json").read_text())
    cfg = ProbeConfig(**raw["probe"])
    assert probe_variant_name(cfg) == "touch_part_attention_hierarchical_raw_taxel"
    assert probe_config_dict(cfg)["touch_kv_mode"] == "raw_mlp"
    reference = json.loads(
        root_path("configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_hierarchical.json").read_text()
    )
    differing = {k for k in raw["probe"] if raw["probe"].get(k) != reference["probe"].get(k)}
    assert differing == {"architecture", "variant_tag", "touch_num_heads", "touch_use_pos_xyz", "touch_kv_mode"}
    for key in ("model", "temporal", "representation", "protocol_dir", "tactile_preprocessing"):
        assert raw[key] == reference[key]


@pytest.mark.parametrize(("source", "kv"), [("random", "layernorm"), ("raw", "raw_mlp")])
def test_touch_probe_trains_and_evaluates_on_cpu(tmp_path, source, kv):
    eps = _episodes()
    train = TouchLatentDataset(
        _write_latents(tmp_path / "lat_train", eps, "train", np.random.default_rng(0)),
        _cache(tmp_path, eps, "train", source),
    )
    val = TouchLatentDataset(
        _write_latents(tmp_path / "lat_val", eps, "validation", np.random.default_rng(1)),
        _cache(tmp_path, eps, "validation", source),
    )
    config = _touch_config(kv=kv, tag=f"t_{source}")
    result = train_action_probe(
        train, val, config, tmp_path / "ckpt", seed=0, max_steps=2, log_dir=tmp_path / "logs",
        progress=False, num_workers=0,
    )  # fmt: skip
    assert result["manifest"]["actual_train_steps"] == 2
    assert result["manifest"]["provenance"]["touch_identity"] == train.provenance["touch_identity"]
    norm = {"action": {"mean": [0.0] * 22, "std": [1.0] * 22}}
    norm["hash"] = digest(norm)
    test = TouchLatentDataset(
        _write_latents(tmp_path / "lat_test", eps, "test", np.random.default_rng(2)),
        _cache(tmp_path, eps, "test", source),
    )
    report = evaluate_action_probe(test, tmp_path / "ckpt" / "best", norm, tmp_path / "eval")
    assert report["model"] == f"trlt_self_only__touch_part_attention_hierarchical_t_{source}"
    assert np.isfinite(report["metrics"]["primary_normalized_episode_macro_mse"])
    # A cache from a different touch source must be rejected at evaluation.
    other = "raw" if source == "random" else "random"
    mismatched = TouchLatentDataset(test.latents, _cache(tmp_path, eps, "test", other))
    with pytest.raises(ValueError, match="touch_identity"):
        evaluate_action_probe(mismatched, tmp_path / "ckpt" / "best", norm, tmp_path / "eval2")
