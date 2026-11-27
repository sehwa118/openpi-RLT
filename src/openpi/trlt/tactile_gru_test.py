"""CPU contract tests for the Assembly tactile MLP-GRU comparison (B0/M1/M5/M5s/G5).

See docs/trlt/assembly_tactile_gru_plan.md. Run with CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu.
"""

from dataclasses import replace
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.tactile_readout import _raw_gru_tokens
from openpi.models.trlt_actor import PartAttentionActionProbe
from openpi.models.trlt_actor import build_action_probe
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import digest
from openpi.trlt.config import probe_config_dict
from openpi.trlt.config import probe_variant_name
from openpi.trlt.config import root_path
from openpi.trlt.runner import evaluate_action_probe
from openpi.trlt.runner import probe_loss
from openpi.trlt.runner import train_action_probe
from openpi.trlt.tacdex_lingbot import PaxiniLingBotConfig
from openpi.trlt.touch_tokens import TouchLatentDataset
from openpi.trlt.touch_tokens import build_touch_cache
from openpi.trlt.touch_tokens_test import PROV
from openpi.trlt.touch_tokens_test import TEMPORAL
from openpi.trlt.touch_tokens_test import _contiguous
from openpi.trlt.touch_tokens_test import _episodes
from openpi.trlt.touch_tokens_test import _queries
from openpi.trlt.touch_tokens_test import _write_latents

CFG = "configs/trlt/3dtacdex_assembly"
CONDITIONS = {  # name -> (config file, raw frames)
    "M1": ("tactile_gru/probe_tgru_m1.json", 1),
    "M5": ("tactile_gru/probe_tgru_m5.json", 5),
    "M5s": ("tactile_gru/probe_tgru_m5s.json", 5),
    "G5": ("tactile_gru/probe_tgru_g5.json", 5),
}
LONG_CONDITIONS = {  # long-history follow-up (docs/trlt/assembly_tactile_gru_long_history_plan.md)
    "M10": ("tactile_gru/probe_tgru_m10.json", 10),
    "M10s": ("tactile_gru/probe_tgru_m10s.json", 10),
    "G10": ("tactile_gru/probe_tgru_g10.json", 10),
    "M20": ("tactile_gru/probe_tgru_m20.json", 20),
    "M20s": ("tactile_gru/probe_tgru_m20s.json", 20),
    "G20": ("tactile_gru/probe_tgru_g20.json", 20),
}
ALL_CONDITIONS = {**CONDITIONS, **LONG_CONDITIONS}
B0_CONFIG = "attention_actor/probe_part_attn_hierarchical_drop01.json"
H3_CONFIG = "lingbot_touch/probe_touch_raw_taxel.json"
B0_JIWOO = Path(
    "/workspace/users/jiwoo/trlt_attention_actor_3dtacdex_self2_rawclip30_d64_mseed/logs/seed0/"
    "probe_part_attention_hierarchical_drop01/manifest.json"
)


def _raw(name):
    return json.loads(root_path(f"{CFG}/{name}").read_text())


def _probe(name):
    return ProbeConfig(**_raw(name)["probe"])


def _inputs(rng, batch, frames, *, hidden_z=(1, 512)):
    out = [
        jnp.asarray(rng.normal(size=(batch, *hidden_z)), jnp.float32),
        jnp.asarray(rng.normal(size=(batch, 22)), jnp.float32),
        jnp.asarray(rng.uniform(0, 0.6, size=(batch,)), jnp.float32),
    ]
    if frames:
        out += [
            jnp.asarray(rng.uniform(-1, 1, size=(batch, 4, 15, 3 * frames)), jnp.float32),
            jnp.ones((batch, 4, 15), bool),
        ]
    return out


POS = jnp.asarray(np.random.default_rng(9).uniform(-1, 1, size=(4, 15, 3)), jnp.float32)


def _init(cfg, args, key=0):
    model = build_action_probe(cfg)
    kwargs = {"touch_pos_xyz": POS} if len(args) > 3 else {}
    params = model.init(jax.random.PRNGKey(key), *args, train=False, **kwargs)["params"]
    return model, params, kwargs


def _open_gates(params, value=0.5):
    """Make the zero-initialized touch AdaLN gates nonzero so the touch branch is active."""
    params = flax.core.unfreeze(params) if hasattr(params, "unfreeze") else jax.tree.map(lambda x: x, params)
    for name in params:
        if name.startswith("touch_ca_"):
            bias = params[name]["touch_adaln"]["bias"]
            params[name]["touch_adaln"]["bias"] = jnp.full_like(bias, value)
    return params


# --------------------------------------------------------------------------- configs / identity


def test_new_configs_differ_from_b0_only_in_touch_fields():
    b0 = _raw(B0_CONFIG)
    for name, (path, frames) in ALL_CONDITIONS.items():
        raw = _raw(path)
        for key in ("model", "temporal", "representation", "protocol_dir", "tactile_preprocessing", "split_seed"):
            assert raw[key] == b0[key], (name, key)
        differing = {k for k in set(raw["probe"]) | set(b0["probe"]) if raw["probe"].get(k) != b0["probe"].get(k)}
        assert differing <= {
            "architecture", "variant_tag", "touch_num_heads", "touch_use_pos_xyz", "touch_kv_mode",
            "touch_raw_frames", "touch_raw_hidden_dim", "touch_gru_hidden_dim",
        }, (name, differing)  # fmt: skip
        cfg = ProbeConfig(**raw["probe"])
        assert (cfg.hidden_dim, cfg.depth, cfg.num_heads, cfg.dropout_rate) == (128, 2, 4, 0.1)
        assert (cfg.attention_mask_mode, cfg.horizon, cfg.action_dim) == ("hierarchical", 3, 22)
        assert cfg.touch_raw_frames == frames
    names = {probe_variant_name(_probe(path)) for path, _ in ALL_CONDITIONS.values()}
    assert len(names) == len(ALL_CONDITIONS)
    assert all(n.startswith("touch_part_attention_hierarchical_tgru_") for n in names)


def test_existing_identities_unchanged():
    """B0 and H3 serialize exactly as before: the new optional fields are omitted when unset."""
    b0 = probe_config_dict(_probe(B0_CONFIG))
    assert not any(k.startswith("touch_") for k in b0)
    if B0_JIWOO.exists():  # reused B0 checkpoints: their stored train_config must still match
        assert json.loads(B0_JIWOO.read_text())["train_config"] == b0
    h3 = probe_config_dict(_probe(H3_CONFIG))
    assert not {"touch_raw_frames", "touch_raw_hidden_dim", "touch_gru_hidden_dim"} & set(h3)
    assert h3["touch_kv_mode"] == "raw_mlp"
    assert h3["dropout_rate"] == 0.0
    assert probe_variant_name(_probe(H3_CONFIG)) == "touch_part_attention_hierarchical_raw_taxel"
    # H3 parameter tree (names and shapes) is unchanged: Dense(C->d) then Dense(d->d).
    rng = np.random.default_rng(0)
    _, params, _ = _init(_probe(H3_CONFIG), _inputs(rng, 2, 2))
    enc = params["touch_ca_0"]
    assert enc["touch_raw_fc1"]["kernel"].shape == (6, 128)
    assert enc["touch_proj"]["kernel"].shape == (128, 128)
    assert "touch_gru" not in enc


@pytest.mark.parametrize(
    "kwargs",
    [
        {"touch_kv_mode": "raw_gru"},  # GRU width required
        {"touch_kv_mode": "raw_mlp", "touch_gru_hidden_dim": 32},
        {"touch_kv_mode": "layernorm", "touch_raw_hidden_dim": 16},
        {"touch_kv_mode": "layernorm", "touch_raw_frames": 5},
        {"touch_kv_mode": "raw_gru", "touch_gru_hidden_dim": 0},
        {"touch_kv_mode": "raw_mlp", "touch_raw_frames": True},
    ],
)
def test_invalid_touch_encoder_configs_rejected(kwargs):
    base = dict(_raw(CONDITIONS["M5"][0])["probe"])
    for key in ("touch_raw_frames", "touch_raw_hidden_dim", "touch_gru_hidden_dim"):
        base.pop(key, None)
    with pytest.raises(ValueError, match="touch_"):
        ProbeConfig(**{**base, **kwargs})
    with pytest.raises(ValueError, match="touch_part_attention"):
        ProbeConfig(**{**_raw(B0_CONFIG)["probe"], "touch_raw_frames": 5})


def test_parameter_counts():
    rng = np.random.default_rng(0)
    counts = {}
    for name, (path, frames) in {"B0": (B0_CONFIG, 0), **ALL_CONDITIONS}.items():
        _, params, _ = _init(_probe(path), _inputs(rng, 2, frames))
        touch = {k: v for k, v in params.items() if k.startswith("touch_ca_")}
        enc = sum(
            x.size
            for layer in touch.values()
            for k, v in layer.items()
            if k in ("touch_raw_fc1", "touch_proj", "touch_gru")
            for x in jax.tree.leaves(v)
        )
        counts[name] = (
            sum(x.size for x in jax.tree.leaves(params)),
            sum(x.size for x in jax.tree.leaves(touch)),
            enc,
        )
    assert counts["B0"] == (721602, 0, 0)  # equals the reused B0 manifests' parameter_count
    # Shared cross-attention parts are identical; only the encoder differs.
    for name in ALL_CONDITIONS:
        assert counts[name][1] - counts[name][2] == counts["M5"][1] - counts["M5"][2]
    # Capacity-matched MLP controls are within 1% of the GRU encoder (GRU size does not depend on T).
    for t in (5, 10, 20):
        assert counts[f"G{t}"] == counts["G5"]
        assert abs(counts[f"M{t}s"][2] - counts[f"G{t}"][2]) / counts[f"G{t}"][2] < 0.01
        assert counts[f"M{t}"][2] / counts[f"G{t}"][2] > 2
    assert {n: counts[n] for n in LONG_CONDITIONS} == {
        "M10": (994754, 273152, 40960),
        "M10s": (969314, 247712, 15520),
        "G10": (969218, 247616, 15424),
        "M20": (1002434, 280832, 48640),
        "M20s": (969170, 247568, 15376),
        "G20": (969218, 247616, 15424),
    }


# --------------------------------------------------------------------------- GRU semantics


def test_raw_gru_is_shared_time_ordered_and_window_local():
    import flax.linen as nn

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return _raw_gru_tokens(x, 8, 16)

    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.uniform(-1, 1, size=(3, 4, 15, 15)), jnp.float32)
    m = M()
    p = m.init(jax.random.PRNGKey(0), x)
    y = m.apply(p, x)
    assert y.shape == (3, 4, 15, 16)
    # Single shared GRU (no per-taxel / per-finger parameters).
    assert set(p["params"]) == {"touch_gru", "touch_proj"}
    # Permuting taxels/fingers permutes outputs: weights are shared across tokens.
    perm = rng.permutation(15)
    np.testing.assert_allclose(m.apply(p, x[:, ::-1][:, :, perm]), y[:, ::-1][:, :, perm], atol=1e-6)
    # Each window is processed from a zero state: one token's output depends only on its own window.
    x2 = x.at[0, 0, 0].set(0.3)
    y2 = m.apply(p, x2)
    changed = np.abs(np.asarray(y2 - y)).sum(-1) > 1e-7
    assert changed[0, 0, 0]
    assert changed.sum() == 1
    # Order matters (sequential processing, oldest->newest), unlike a set function.
    reversed_time = x.reshape(3, 4, 15, 5, 3)[:, :, :, ::-1].reshape(3, 4, 15, 15)
    assert not np.allclose(m.apply(p, reversed_time), y, atol=1e-4)
    # Matches an explicit zero-initialized GRU recurrence.
    cell = nn.GRUCell(8)
    cp = {"params": p["params"]["touch_gru"]["cell"]}
    steps = x.reshape(-1, 5, 3)
    h = jnp.zeros((steps.shape[0], 8))
    for t in range(5):
        h, _ = cell.apply(cp, h, steps[:, t])
    manual = h @ p["params"]["touch_proj"]["kernel"] + p["params"]["touch_proj"]["bias"]
    np.testing.assert_allclose(manual.reshape(y.shape), y, atol=1e-5)


def test_raw_frames_guard_rejects_cache_config_mismatch():
    rng = np.random.default_rng(0)
    for name in ("M5", "G5", "M5s", "M10", "G20"):
        cfg = _probe(ALL_CONDITIONS[name][0])
        with pytest.raises(ValueError, match="raw_frames"):
            _init(cfg, _inputs(rng, 2, 1))  # 1-frame tokens into a multi-frame condition
    with pytest.raises(ValueError, match="raw_frames"):
        _init(_probe(ALL_CONDITIONS["G10"][0]), _inputs(rng, 2, 20))  # 20-frame cache into a 10-frame model
    with pytest.raises(ValueError, match="raw_frames"):
        _init(_probe(CONDITIONS["M1"][0]), _inputs(rng, 2, 5))


def test_touch_conditions_start_as_b0_function():
    """Zero-initialized touch gates: every new condition starts as exactly the B0 function."""
    rng = np.random.default_rng(2)
    b0cfg = _probe(B0_CONFIG)
    args0 = _inputs(rng, 4, 0)
    p0 = PartAttentionActionProbe(b0cfg).init(jax.random.PRNGKey(0), *args0)["params"]
    out0 = PartAttentionActionProbe(b0cfg).apply({"params": p0}, *args0)
    for name, (path, frames) in ALL_CONDITIONS.items():
        touch = list(_inputs(np.random.default_rng(3), 4, frames)[3:])
        model, p1, kw = _init(_probe(path), [*args0, *touch])
        shared = {k: v for k, v in p1.items() if not k.startswith("touch_ca_")}
        assert set(shared) == set(p0), name
        out1 = model.apply({"params": {**p1, **p0}}, *args0, *touch, train=False, **kw)
        np.testing.assert_allclose(np.asarray(out1), np.asarray(out0), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", list(ALL_CONDITIONS))
def test_gradients_reach_trainable_encoder_but_not_frozen_inputs(name):
    path, frames = ALL_CONDITIONS[name]
    cfg = _probe(path)
    rng = np.random.default_rng(4)
    args = _inputs(rng, 8, frames)
    target = jnp.asarray(rng.normal(size=(8, 3, 22)), jnp.float32)
    mask = jnp.ones((8, 3), bool)
    model, params, kw = _init(cfg, args)

    def loss(p, z, touch):
        a = [z, args[1], args[2], touch, args[4]]
        pred = model.apply({"params": p}, *a, train=True, rngs={"dropout": jax.random.PRNGKey(1)}, **kw)
        return probe_loss(pred, target, mask)

    # At the exact init the touch gate is zero, so encoder gradients are zero by construction ...
    g0 = jax.grad(loss)(params, args[0], args[3])
    assert np.any(np.abs(np.asarray(g0["touch_ca_0"]["touch_adaln"]["kernel"])) > 0)
    # ... and once the gate is open the trainable MLP/GRU receives gradient through stop_gradient(input).
    opened = _open_gates(params)
    gp, gz, gt = jax.grad(loss, argnums=(0, 1, 2))(opened, args[0], args[3])
    enc_key = "touch_gru" if cfg.touch_kv_mode == "raw_gru" else "touch_raw_fc1"
    for layer in ("touch_ca_0", "touch_ca_1"):
        leaves = jax.tree.leaves(gp[layer][enc_key])
        assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
        assert sum(float(jnp.abs(x).sum()) for x in leaves) > 0, (name, layer)
        assert float(jnp.abs(gp[layer]["touch_proj"]["kernel"]).sum()) > 0
    # Frozen Stage A latent z and the frozen tactile cache receive no gradient.
    assert not np.any(np.asarray(gz))
    assert not np.any(np.asarray(gt))


@pytest.mark.parametrize("name", list(ALL_CONDITIONS))
def test_optimizer_updates_encoder_and_actor(name):
    path, frames = ALL_CONDITIONS[name]
    cfg = _probe(path)
    rng = np.random.default_rng(5)
    args = _inputs(rng, 16, frames)
    target = jnp.asarray(rng.normal(size=(16, 3, 22)), jnp.float32)
    mask = jnp.ones((16, 3), bool)
    model, params, kw = _init(cfg, args)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(5e-3, weight_decay=1e-4))
    opt = tx.init(params)
    start = params

    @jax.jit
    def step(p, o, key):
        def f(q):
            pred = model.apply({"params": q}, *args, train=True, rngs={"dropout": key}, **kw)
            return probe_loss(pred, target, mask)

        value, g = jax.value_and_grad(f)(p)
        upd, o = tx.update(g, o, p)
        return optax.apply_updates(p, upd), o, value

    losses = []
    for i in range(30):
        params, opt, value = step(params, opt, jax.random.PRNGKey(i))
        losses.append(float(value))
    assert losses[-1] < losses[0]
    enc_key = "touch_gru" if cfg.touch_kv_mode == "raw_gru" else "touch_raw_fc1"
    for group in ("block_0", "part_in_arm", "touch_ca_0", "touch_ca_1"):
        before, after = jax.tree.leaves(start[group]), jax.tree.leaves(params[group])
        assert any(not np.allclose(a, b) for a, b in zip(before, after, strict=True)), (name, group)
    for layer in ("touch_ca_0", "touch_ca_1"):
        before, after = jax.tree.leaves(start[layer][enc_key]), jax.tree.leaves(params[layer][enc_key])
        assert any(not np.allclose(a, b) for a, b in zip(before, after, strict=True)), (name, layer)


@pytest.mark.parametrize("name", list(ALL_CONDITIONS))
def test_eval_is_deterministic_and_dropout_only_in_training(name):
    path, frames = ALL_CONDITIONS[name]
    rng = np.random.default_rng(6)
    args = _inputs(rng, 4, frames)
    model, params, kw = _init(_probe(path), args)
    params = _open_gates(params)
    a = model.apply({"params": params}, *args, train=False, **kw)
    b = model.apply({"params": params}, *args, train=False, **kw)
    assert a.shape == (4, 3, 22)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    t1 = model.apply({"params": params}, *args, train=True, rngs={"dropout": jax.random.PRNGKey(1)}, **kw)
    t2 = model.apply({"params": params}, *args, train=True, rngs={"dropout": jax.random.PRNGKey(2)}, **kw)
    assert not np.allclose(t1, t2)


def test_invalid_fingers_are_masked_out():
    path, frames = CONDITIONS["G5"]
    rng = np.random.default_rng(7)
    args = _inputs(rng, 2, frames)
    model, params, kw = _init(_probe(path), args)
    params = _open_gates(params)
    valid = args[4].at[:, 1].set(False)  # index finger fully invalid
    base = model.apply({"params": params}, *args[:4], valid, train=False, **kw)
    noisy = args[3].at[:, 1].set(rng.uniform(-1, 1, size=(2, 15, 15)))  # changing its tokens has no effect
    other = model.apply({"params": params}, *args[:3], noisy, valid, train=False, **kw)
    np.testing.assert_allclose(np.asarray(base), np.asarray(other), atol=1e-6)


# --------------------------------------------------------------------------- raw caches (1 / 5 frames)


def _build(tmp_path, eps, frames, split="train", name=None):
    out = tmp_path / (name or f"raw_t{frames}_{split}")
    manifest = build_touch_cache(
        queries=_queries(eps), episode=eps.__getitem__, contiguous=_contiguous, temporal=TEMPORAL,
        provenance=dict(PROV, split=split), output_dir=out, source="raw",
        adapter=PaxiniLingBotConfig(frames=frames), batch_size=5,
    )  # fmt: skip
    return out, manifest


def test_raw_caches_1_and_5_frames_are_causal_and_aligned(tmp_path):
    eps = _episodes()
    caches = {f: _build(tmp_path, eps, f) for f in (1, 2, 5)}  # odd frames: no LingBot tubelet rule on raw
    ids = {f: m["touch_identity"] for f, (_, m) in caches.items()}
    assert len(set(ids.values())) == 3  # 1/2/5-frame caches never share an identity
    keys = {}
    for f, (out, _) in caches.items():
        tokens = np.load(out / "tokens.npy").astype(np.float32)
        assert tokens.shape == (16, 4, 15, 3 * f)
        with np.load(out / "keys.npz") as k:
            keys[f] = (k["segment_id"].tolist(), k["query_frame_index"].tolist())
        # Newest slot is the query frame itself (equal to the 1-frame cache), never a future frame.
        np.testing.assert_array_equal(tokens[..., -3:], np.load(caches[1][0] / "tokens.npy").astype(np.float32))
        np.testing.assert_array_equal(np.load(out / "valid.npy"), np.load(caches[1][0] / "valid.npy"))
    assert keys[1] == keys[2] == keys[5]  # identical query sets/order across history lengths
    t5 = np.load(caches[5][0] / "tokens.npy").astype(np.float32)
    signal = np.clip(eps[0]["observation.tactile"] / 30.0, -1, 1)
    z = np.clip(eps[0]["observation.tactile"][..., 2] / 30.0, 0, 1)
    signal[..., 2] = z
    # Query row 8 of episode 0 (all rows contiguous): rows 4..8 oldest->newest; finger order thumb,index,..
    # (finger 0 = pad 0 thumb tip, finger 2 = pad 4 middle tip).
    for finger, pad in ((0, 0), (2, 4), (3, 6)):
        expected = signal[[4, 5, 6, 7, 8], pad].transpose(1, 0, 2).reshape(15, 15)
        np.testing.assert_allclose(t5[8, finger], expected, atol=2e-3)
    # Early query (row 1): missing history replicates the segment's first row, not a padding value.
    expected = signal[[0, 0, 0, 0, 1], 0].transpose(1, 0, 2).reshape(15, 15)
    np.testing.assert_allclose(t5[1, 0], expected, atol=2e-3)
    # Segment boundary: episode 1 has a frame gap before row 3; query row 4 never reads rows < 3.
    row = 9 + 4  # episode 1 queries follow the 9 rows of episode 0
    ep1 = np.clip(eps[1]["observation.tactile"] / 30.0, -1, 1)
    ep1[..., 2] = np.clip(eps[1]["observation.tactile"][..., 2] / 30.0, 0, 1)
    expected = ep1[[3, 3, 3, 3, 4], 0].transpose(1, 0, 2).reshape(15, 15)
    np.testing.assert_allclose(t5[row, 0], expected, atol=2e-3)
    assert caches[5][1]["stats"]["replicated_history_queries"] > caches[2][1]["stats"]["replicated_history_queries"]


def test_raw_caches_10_and_20_frames_are_causal_and_replicate_segment_start(tmp_path):
    eps = _episodes()
    caches = {f: _build(tmp_path, eps, f) for f in (1, 10, 20)}
    assert len({m["touch_identity"] for _, m in caches.values()}) == 3
    t1 = np.load(caches[1][0] / "tokens.npy").astype(np.float32)
    signal = np.clip(eps[0]["observation.tactile"] / 30.0, -1, 1)
    signal[..., 2] = np.clip(eps[0]["observation.tactile"][..., 2] / 30.0, 0, 1)
    for f in (10, 20):
        out, _ = caches[f]
        tokens = np.load(out / "tokens.npy").astype(np.float32)
        assert tokens.shape == (16, 4, 15, 3 * f)
        np.testing.assert_array_equal(tokens[..., -3:], t1)
        with np.load(out / "keys.npz") as k, np.load(caches[1][0] / "keys.npz") as k1:
            assert k["segment_id"].tolist() == k1["segment_id"].tolist()
            assert k["query_frame_index"].tolist() == k1["query_frame_index"].tolist()
        # Query row 8 of episode 0: rows 0..8, with the missing front replicated from row 0 (oldest->newest).
        rows = [0] * (f - 9) + list(range(9))
        expected = signal[rows, 0].transpose(1, 0, 2).reshape(15, 3 * f)
        np.testing.assert_allclose(tokens[8, 0], expected, atol=2e-3)
    assert caches[20][1]["stats"]["replicated_history_queries"] >= caches[10][1]["stats"]["replicated_history_queries"]


def test_g5_trains_and_evaluates_end_to_end_on_cpu(tmp_path):
    eps = _episodes()
    cfg = replace(_probe(CONDITIONS["G5"][0]), hidden_dim=32, batch_size=8, max_epochs=1, patience=None)
    split = {}
    for i, name in enumerate(("train", "validation", "test")):
        cache, _ = _build(tmp_path, eps, 5, name)
        split[name] = TouchLatentDataset(
            _write_latents(tmp_path / f"lat_{name}", eps, name, np.random.default_rng(i)), cache
        )
    result = train_action_probe(
        split["train"], split["validation"], cfg, tmp_path / "ckpt", seed=0, max_steps=2,
        log_dir=tmp_path / "logs", progress=False, num_workers=0,
    )  # fmt: skip
    manifest = result["manifest"]
    assert manifest["actual_train_steps"] == 2
    assert manifest["train_config"]["touch_kv_mode"] == "raw_gru"
    assert manifest["train_config"]["touch_gru_hidden_dim"] == 32
    assert manifest["provenance"]["touch_identity"] == split["train"].provenance["touch_identity"]
    norm = {"action": {"mean": [0.0] * 22, "std": [1.0] * 22}}
    norm["hash"] = digest(norm)
    report = evaluate_action_probe(split["test"], tmp_path / "ckpt" / "best", norm, tmp_path / "eval")
    assert report["model"].endswith("touch_part_attention_hierarchical_tgru_g5")
    assert np.isfinite(report["metrics"]["primary_normalized_episode_macro_mse"])
    again = evaluate_action_probe(split["test"], tmp_path / "ckpt" / "best", norm, tmp_path / "eval_again")
    assert again["metrics"] == report["metrics"]
    # A 1-frame cache cannot be evaluated with a 5-frame checkpoint (identity mismatch).
    one, _ = _build(tmp_path, eps, 1, "test", name="raw_t1_test_other")
    with pytest.raises(ValueError, match="touch_identity"):
        evaluate_action_probe(TouchLatentDataset(split["test"].latents, one), tmp_path / "ckpt" / "best", norm,
                              tmp_path / "eval_bad")  # fmt: skip
