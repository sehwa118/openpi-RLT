"""CPU unit and parity tests for LingBot-Touch Flax encoder, TactilePointCache, and Stage B readout."""

import importlib.util
import json

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from torch import nn
from torch.nn import functional as fn

from openpi.models.lingbot_touch_flax import LingBotTouchConfig
from openpi.models.lingbot_touch_flax import LingBotTouchEncoder
from openpi.models.lingbot_touch_flax import convert_torch_state_dict_to_flax
from openpi.models.lingbot_touch_flax import encode_bimanual_visuotactile
from openpi.models.tactile_readout import GlobalTouchAttentiveReadout
from openpi.models.tactile_readout import PerFingerTouchCrossAttention
from openpi.models.tactile_readout import TouchCrossAttentionConfig
from openpi.models.tactile_readout import TouchPartAttentionActionProbe
from openpi.models.trlt_actor import part_specs_for_dim
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import root_path
from openpi.trlt.runner import probe_loss
from openpi.trlt.tactile_points import LINGBOT_MAP_SHA256
from openpi.trlt.tactile_points import TactilePointCache
from openpi.trlt.tactile_points import load_lingbot_visuotactile_map
from openpi.trlt.tactile_points import make_synthetic_sampling_map
from openpi.trlt.tactile_points import prepare_raw_to_lingbot_signal
from openpi.trlt.tactile_points import sample_deformation_240


class _TorchPointTokenizer(nn.Module):
    def __init__(self, embed_dim: int, tubelet_size: int):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.signal_mlp = nn.Sequential(
            nn.Linear(tubelet_size * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tubelets: torch.Tensor) -> torch.Tensor:
        return self.signal_mlp(tubelets)


class _TorchRoPE(nn.Module):
    def __init__(self, head_dim: int, config: LingBotTouchConfig):
        super().__init__()
        self.head_dim = head_dim
        self.axis_dim = 2 * ((head_dim // 2) // 5)
        self.config = config

    def forward(self, q, k, time_id, finger_id, pos_xyz):
        scale_fp32 = float(torch.tensor(self.config.spatial_coordinate_scale, dtype=torch.float32).item())
        specs = (
            (self.config.time_theta, time_id, 1.0),
            (self.config.finger_theta, finger_id, 1.0),
            (self.config.spatial_theta, pos_xyz[..., 0], scale_fp32),
            (self.config.spatial_theta, pos_xyz[..., 1], scale_fp32),
            (self.config.spatial_theta, pos_xyz[..., 2], scale_fp32),
        )
        q_chunks, k_chunks = [], []
        start = 0
        for theta, pos, scale in specs:
            end = start + self.axis_dim
            inv_freq = torch.pow(
                torch.tensor(theta, dtype=torch.float32),
                -torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim,
            )
            phase = pos.float()[:, None, :, None] * float(scale) * inv_freq[None, None, None, :]
            cos, sin = phase.cos(), phase.sin()
            for tensor, out_list in ((q, q_chunks), (k, k_chunks)):
                vals = tensor[..., start:end].float()
                even, odd = vals[..., 0::2], vals[..., 1::2]
                out_list.append(torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2))
            start = end
        if start < self.head_dim:
            q_chunks.append(q[..., start:].float())
            k_chunks.append(k[..., start:].float())
        return torch.cat(q_chunks, dim=-1).to(q.dtype), torch.cat(k_chunks, dim=-1).to(k.dtype)


class _TorchBlock(nn.Module):
    def __init__(self, config: LingBotTouchConfig):
        super().__init__()
        dim = config.embed_dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, 3 * dim)
        self.attn.proj = nn.Linear(dim, dim)
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.rope = _TorchRoPE(self.head_dim, config)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * config.mlp_ratio)), nn.GELU(), nn.Linear(int(dim * config.mlp_ratio), dim))

    def forward(self, x, time_id, finger_id, pos_xyz):
        b, n, c = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.rope(q, k, time_id, finger_id, pos_xyz)
        out = fn.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x + self.attn.proj(out.transpose(1, 2).reshape(b, n, c))
        return x + self.mlp(self.norm2(x))


class _TorchTargetEncoderReplica(nn.Module):
    """Test double reproducing Robbyant/lingbot-touch TargetEncoder module hierarchy & tensor math."""

    def __init__(self, config: LingBotTouchConfig):
        super().__init__()
        self.config = config
        self.tokenizer = nn.Module()
        self.tokenizer.tokenizers = nn.ModuleDict(
            {
                "magnetic": _TorchPointTokenizer(config.embed_dim, config.tubelet_size),
                "piezoresistive": _TorchPointTokenizer(config.embed_dim, config.tubelet_size),
                "visuotactile": _TorchPointTokenizer(config.embed_dim, config.tubelet_size),
            }
        )
        if config.use_shared_projector:
            self.shared_projector = nn.Module()
            self.shared_projector.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
            self.shared_projector.fc1 = nn.Linear(config.embed_dim, config.embed_dim)
            self.shared_projector.fc2 = nn.Linear(config.embed_dim, config.embed_dim)
        else:
            self.shared_projector = None
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([_TorchBlock(config) for _ in range(config.depth)])
        self.transformer.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.transformer.norms_block = nn.ModuleList(
            [nn.LayerNorm(config.embed_dim, eps=1e-6) for _ in config.output_layers]
        )

    def forward(self, signal, pos_xyz, coverage_mask, *, modality: str):
        b, frames, fingers, taxels, channels = signal.shape
        keep = coverage_mask.reshape(-1).nonzero().flatten()
        spatial = keep.numel()
        compact_sig = signal.reshape(b, frames, fingers * taxels, channels).index_select(2, keep)
        compact_pos = pos_xyz.unsqueeze(0).expand(b, -1, -1, -1).reshape(b, fingers * taxels, 3).index_select(1, keep)
        finger = torch.arange(fingers, dtype=torch.long).view(fingers, 1).expand(fingers, taxels).reshape(-1).index_select(0, keep)
        times = frames // self.config.tubelet_size
        tubelets = (
            compact_sig.reshape(b, times, self.config.tubelet_size, spatial, channels)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(b, times, spatial, self.config.tubelet_size * channels)
        )
        x = self.tokenizer.tokenizers[modality](tubelets).reshape(b, times * spatial, self.config.embed_dim)
        if self.shared_projector is not None:
            h = fn.gelu(self.shared_projector.fc1(self.shared_projector.norm(x)))
            x = x + self.shared_projector.fc2(h)
        time_id = torch.arange(times, dtype=torch.long).repeat_interleave(spatial).unsqueeze(0).expand(b, -1)
        finger_id = finger.repeat(times).unsqueeze(0).expand(b, -1)
        pos_b = compact_pos[:, None, :, :].expand(b, times, spatial, 3).reshape(b, times * spatial, 3)
        outputs = []
        for idx, block in enumerate(self.transformer.blocks):
            x = block(x, time_id, finger_id, pos_b)
            if idx in self.config.output_layers:
                norm = self.transformer.norms_block[self.config.output_layers.index(idx)]
                outputs.append(norm(x))
        return torch.cat(outputs, dim=-1), time_id[0], finger_id[0], pos_b


@pytest.mark.parametrize("use_shared_projector", [False, True])
@pytest.mark.parametrize("modality", ["visuotactile", "magnetic", "piezoresistive"])
def test_lingbot_touch_flax_matches_pytorch_reference(use_shared_projector, modality):
    cfg = LingBotTouchConfig(
        embed_dim=36,
        depth=3,
        num_heads=3,
        tubelet_size=2,
        mlp_ratio=4.0,
        output_layers=(0, 2),
        use_shared_projector=use_shared_projector,
    )
    torch.manual_seed(11)
    pt_model = _TorchTargetEncoderReplica(cfg).eval()
    flax_model = LingBotTouchEncoder(cfg)

    rng = np.random.default_rng(17)
    signal_np = rng.uniform(0.0, 1.0, size=(2, 6, 5, 5, 3)).astype(np.float32)
    if modality != "magnetic":
        signal_np[..., :2] = 0.0
    pos_np = rng.normal(0.0, 0.7, size=(5, 5, 3)).astype(np.float32)
    cov_np = np.ones((5, 5), dtype=bool)
    cov_np[1, 3:] = False
    cov_np[4, 4:] = False

    # Verify Flax init pytree structure matches converted PyTorch parameters identically.
    flax_init = flax_model.init(
        jax.random.PRNGKey(0),
        jnp.asarray(signal_np),
        jnp.asarray(pos_np),
        cov_np,
        modality=modality,
    )["params"]
    flax_params = convert_torch_state_dict_to_flax(pt_model.state_dict(), cfg)
    assert jax.tree_util.tree_structure(flax_params) == jax.tree_util.tree_structure(flax_init)
    for a, b in zip(jax.tree_util.tree_leaves(flax_params), jax.tree_util.tree_leaves(flax_init), strict=True):
        assert a.shape == b.shape
        assert a.dtype == b.dtype

    with torch.no_grad():
        pt_latent, pt_time, pt_finger, pt_pos = pt_model(
            torch.from_numpy(signal_np), torch.from_numpy(pos_np), torch.from_numpy(cov_np), modality=modality
        )
    flax_out = flax_model.apply(
        {"params": flax_params}, jnp.asarray(signal_np), jnp.asarray(pos_np), cov_np, modality=modality
    )

    np.testing.assert_array_equal(np.asarray(flax_out.address.time_id), pt_time.numpy())
    np.testing.assert_array_equal(np.asarray(flax_out.address.finger_id), pt_finger.numpy())
    np.testing.assert_allclose(np.asarray(flax_out.address.pos_xyz), pt_pos.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(flax_out.latent), pt_latent.numpy(), atol=1e-4, rtol=1e-4)


def test_lingbot_touch_tiny_config_and_parameter_count():
    tiny_json = {
        "address_rope": {
            "finger_theta": 10000.0,
            "mode": "axial_time_finger_xyz_qk_v1",
            "spatial_coordinate_scale": 1.5707963267948966,
            "spatial_theta": 100.0,
            "time_theta": 10000.0,
        },
        "depth": 12,
        "embed_dim": 192,
        "group_by_source": {
            "magnetic": "magnetic",
            "visuotactile": "visuotactile",
            "piezoresistive": "piezoresistive",
        },
        "mlp_ratio": 4.0,
        "num_heads": 3,
        "output_layers": [2, 5, 8, 11],
        "tubelet_size": 2,
        "use_shared_projector": False,
    }
    cfg = LingBotTouchConfig.from_dict(tiny_json)
    assert cfg == LingBotTouchConfig.tiny()
    assert cfg.target_dim == 768

    encoder = LingBotTouchEncoder(cfg)
    sig = jnp.ones((2, 10, 5, 4, 3), dtype=jnp.float32) * 0.25
    pos = jnp.zeros((5, 4, 3), dtype=jnp.float32)
    cov = np.ones((5, 4), dtype=bool)
    params = encoder.init(jax.random.PRNGKey(0), sig, pos, cov, modality="visuotactile")["params"]
    param_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
    assert param_count == 5_455_104

    out = encoder.apply({"params": params}, sig, pos, cov, modality="visuotactile")
    assert out.latent.shape == (2, 5 * 5 * 4, 768)
    assert bool(jnp.all(jnp.isfinite(out.latent)))
    grad_sig = jax.grad(lambda s: jnp.sum(encoder.apply({"params": params}, s, pos, cov).latent ** 2))(sig)
    assert bool(jnp.all(grad_sig == 0))

    with pytest.raises(ValueError, match="mismatch"):
        convert_torch_state_dict_to_flax({}, cfg)


def test_convert_bundle_roundtrip_and_parity(tmp_path):
    spec = importlib.util.spec_from_file_location("convert_lingbot_touch", root_path("scripts/convert_lingbot_touch.py"))
    assert spec is not None
    assert spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    cfg = LingBotTouchConfig(embed_dim=36, depth=2, num_heads=3, output_layers=(0, 1))
    torch.manual_seed(3)
    pt_model = _TorchTargetEncoderReplica(cfg).eval()
    bundle_dir = tmp_path / "bundle"
    out_dir = tmp_path / "flax_out"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(json.dumps(cfg.to_dict()))
    torch.save(pt_model.state_dict(), bundle_dir / "encoder.pt")

    manifest = converter.convert_bundle(bundle_dir, out_dir, verify=True, atol=1e-4)
    assert manifest["verified"] is True
    assert manifest["max_abs_diff"] < 1e-4
    template = convert_torch_state_dict_to_flax(pt_model.state_dict(), cfg)
    restored = serialization.from_bytes(template, (out_dir / "params.msgpack").read_bytes())
    assert manifest["parameter_count"] == sum(leaf.size for leaf in jax.tree_util.tree_leaves(restored))


def test_tactile_point_cache_sampling_flip_causal_window_and_integrity(tmp_path):
    smap = make_synthetic_sampling_map(32)
    rng = np.random.default_rng(5)
    frames = rng.integers(0, 256, size=(4, 5, 240, 240), dtype=np.uint8)
    # Create strong horizontal asymmetry so flip_horizontal has a clear measurable effect.
    frames[..., :120] = 0
    frames[..., 120:] = 200
    pts_no_flip = sample_deformation_240(frames, smap, flip_horizontal=False)
    pts_flip = sample_deformation_240(frames, smap, flip_horizontal=True)
    assert pts_no_flip.shape == (4, 5, 32)
    assert float(np.max(np.abs(pts_no_flip - pts_flip))) > 50.0

    sig = prepare_raw_to_lingbot_signal(pts_no_flip, smap.coverage_mask)
    assert sig.shape == (4, 5, 32, 3)
    assert bool(np.all(sig[..., :2] == 0.0))
    assert float(sig[..., 2].min()) >= 0.0
    assert float(sig[..., 2].max()) <= 1.0

    metadata = {"dtype": "float16", "num_points": 32, "map_sha256": smap.map_sha256}
    cache = TactilePointCache.create(tmp_path / "tactile_cache", metadata)
    ep_len = 12
    ep_pts = np.zeros((ep_len, 10, 32), dtype=np.float32)
    for step in range(ep_len):
        ep_pts[step] = step * 10.0
    ep_times = np.broadcast_to((np.arange(ep_len, dtype=np.float64) / 30.0)[:, None], (ep_len, 10)).copy()
    ep_valid = np.ones((ep_len, 10), dtype=bool)
    cache.put(7, ep_pts, ep_times, ep_valid)

    # At query_time = 2 / 30s, slots 7..9 match t=0, 1/30, 2/30 and earlier slots (target < 0) stay padded False.
    win = cache.window(7, query_time=2.0 / 30.0, temporal=TemporalConfig(), tubelet_size=2)
    assert win["left_raw"].shape == (10, 5, 32)
    assert win["right_raw"].shape == (10, 5, 32)
    assert win["tactile_valid_mask"].shape == (10, 10)
    assert win["tubelet_valid_mask"].shape == (5, 10)
    assert not bool(win["tactile_valid_mask"][:7].any())
    assert bool(win["tactile_valid_mask"][7:].all())
    assert not bool(win["tubelet_valid_mask"][:3].any())
    assert bool(win["tubelet_valid_mask"][3:].all())
    np.testing.assert_allclose(win["left_raw"][9], 20.0, atol=1e-2)

    # Tampering manifest triggers integrity rejection.
    manifest = json.loads((tmp_path / "tactile_cache" / "manifest.json").read_text())
    manifest["metadata"]["num_points"] = 96
    (tmp_path / "tactile_cache" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        TactilePointCache(tmp_path / "tactile_cache")

    # Missing or bad checksum on LingBot asset dir raises ValueError.
    fake_repo = tmp_path / "lingbot" / "data_interfaces" / "data"
    fake_repo.mkdir(parents=True)
    (fake_repo / "pretraining_sampling_maps_v1.json").write_text("{}")
    (fake_repo / "multires_references_v1.json").write_text("{}")
    assert LINGBOT_MAP_SHA256 != "bad"
    with pytest.raises(ValueError, match="checksum"):
        load_lingbot_visuotactile_map(tmp_path / "lingbot", num_points=32)


@pytest.mark.parametrize(("action_dim", "num_fingers"), [(58, 10), (22, 4)])
@pytest.mark.parametrize("mask_mode", ["full", "kinematic", "hierarchical"])
def test_touch_part_attention_shapes_stop_gradient_and_all_masked_safety(action_dim, num_fingers, mask_mode):
    cfg = ProbeConfig(
        architecture="part_attention",
        attention_mask_mode=mask_mode,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=action_dim,
        horizon=6,
        dropout_rate=0.1,
    )
    touch_cfg = TouchCrossAttentionConfig(num_heads=4, dropout_rate=0.1, use_pos_xyz=True, num_global_queries=4)
    model = TouchPartAttentionActionProbe(cfg, touch_cfg)

    rng = np.random.default_rng(23)
    z = jnp.asarray(rng.normal(size=(2, 1, 16)), dtype=jnp.float32)
    prop = jnp.asarray(rng.normal(size=(2, action_dim)), dtype=jnp.float32)
    age = jnp.array([0.01, 0.04], dtype=jnp.float32)
    touch = jnp.asarray(rng.normal(size=(2, num_fingers, 8, 24)), dtype=jnp.float32)
    pos = jnp.asarray(rng.normal(size=(num_fingers, 8, 3)), dtype=jnp.float32)
    touch_mask = jnp.ones((2, num_fingers, 8), dtype=bool).at[0, 0, :].set(False)

    variables = model.init(
        jax.random.PRNGKey(0), z, prop, age, touch, touch_mask, touch_pos_xyz=pos, train=False
    )

    def loss_fn(params, z_in, prop_in, touch_in):
        pred = model.apply(
            params,
            z_in,
            prop_in,
            age,
            touch_in,
            touch_mask,
            touch_pos_xyz=pos,
            train=True,
            rngs={"dropout": jax.random.PRNGKey(9)},
        )
        return jnp.sum(pred**2), pred

    (_, pred), (grad_params, grad_z, grad_prop, grad_touch) = jax.value_and_grad(
        loss_fn, argnums=(0, 1, 2, 3), has_aux=True
    )(variables, z, prop, touch)
    assert pred.shape == (2, 6, action_dim)
    assert bool(jnp.all(jnp.isfinite(pred)))
    assert bool(jnp.all(grad_z == 0))
    assert bool(jnp.all(grad_touch == 0))
    assert float(jnp.max(jnp.abs(grad_prop))) > 0.0
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grad_params))

    # All-masked sample in GlobalTouchAttentiveReadout outputs finite residual and zero token sensitivity.
    readout = GlobalTouchAttentiveReadout(embed_dim=16, num_queries=4, num_heads=4, use_pos_xyz=True)
    flat_touch = touch.reshape((2, num_fingers * 8, 24))
    flat_mask = jnp.ones((2, num_fingers * 8), dtype=bool).at[0, :].set(False)
    ro_vars = readout.init(jax.random.PRNGKey(1), flat_touch, flat_mask)
    ro_base = readout.apply(ro_vars, flat_touch, flat_mask)
    ro_pert = readout.apply(ro_vars, flat_touch.at[0].add(100.0), flat_mask)
    np.testing.assert_array_equal(np.asarray(ro_base[0]), np.asarray(ro_pert[0]))
    assert bool(jnp.all(jnp.isfinite(ro_base)))


def test_touch_cross_attention_adaln_zero_non_deadlock_and_finger_isolation():
    cfg = ProbeConfig(
        architecture="part_attention",
        attention_mask_mode="hierarchical",
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=58,
        horizon=2,
    )
    model = TouchPartAttentionActionProbe(cfg, TouchCrossAttentionConfig(num_heads=4, use_pos_xyz=True))
    rng = np.random.default_rng(31)
    z = jnp.asarray(rng.normal(size=(2, 1, 12)), dtype=jnp.float32)
    prop = jnp.asarray(rng.normal(size=(2, 58)), dtype=jnp.float32)
    age = jnp.array([0.01, 0.02], dtype=jnp.float32)
    touch = jnp.asarray(rng.normal(size=(2, 10, 6, 24)), dtype=jnp.float32)
    pos = jnp.asarray(rng.normal(size=(10, 6, 3)), dtype=jnp.float32)
    touch_mask = jnp.ones((2, 10, 6), dtype=bool)
    target = jnp.asarray(rng.normal(size=(2, 2, 58)), dtype=jnp.float32)
    valid = jnp.ones((2, 2), dtype=bool)

    params = model.init(
        jax.random.PRNGKey(4), z, prop, age, touch, touch_mask, touch_pos_xyz=pos, train=False
    )["params"]
    tx = optax.adamw(learning_rate=5e-2)
    opt_state = tx.init(params)

    def loss_fn(p):
        pred = model.apply(
            {"params": p}, z, prop, age, touch, touch_mask, touch_pos_xyz=pos, train=False
        )
        return probe_loss(pred, target, valid)

    _, g1 = jax.value_and_grad(loss_fn)(params)
    assert float(jnp.max(jnp.abs(g1["touch_ca_0"]["touch_adaln"]["kernel"]))) > 0.0
    assert float(jnp.max(jnp.abs(g1["touch_ca_0"]["touch_proj"]["kernel"]))) == 0.0
    assert float(jnp.max(jnp.abs(g1["touch_ca_0"]["touch_out"]["kernel"]))) == 0.0

    for step in range(3):
        _, g = jax.value_and_grad(loss_fn)(params)
        if step == 1:
            for key_name in ("touch_proj", "pos_proj", "query", "key", "value", "touch_out"):
                assert float(jnp.max(jnp.abs(g["touch_ca_0"][key_name]["kernel"]))) > 0.0
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

    parts = {part.name: part for part in part_specs_for_dim(58)}
    base_out = model.apply({"params": params}, z, prop, age, touch, touch_mask, touch_pos_xyz=pos, train=False)
    # Use a non-uniform perturbation across channels/tokens so LayerNorm does not cancel it out.
    touch_delta = jnp.asarray(rng.normal(scale=3.0, size=(2, 6, 24)), dtype=jnp.float32)
    pert_out = model.apply(
        {"params": params}, z, prop, age, touch.at[:, 0].add(touch_delta), touch_mask, touch_pos_xyz=pos, train=False
    )
    delta = np.asarray(jnp.abs(pert_out - base_out))
    assert delta[:, :, parts["left_thumb"].start : parts["left_thumb"].end].max() > 1e-4
    for isolated in ("left_arm", "right_arm", "right_thumb"):
        assert delta[:, :, parts[isolated].start : parts[isolated].end].max() == 0.0

    # Standalone PerFingerTouchCrossAttention isolates left_thumb touch strictly from left_index & arms.
    ca = PerFingerTouchCrossAttention(dim=16, parts=part_specs_for_dim(58), num_heads=4, use_pos_xyz=False)
    x_in = jnp.ones((2, 12, 16), dtype=jnp.float32)
    cond_in = jnp.ones((2, 16), dtype=jnp.float32)
    ca_params = dict(ca.init(jax.random.PRNGKey(5), x_in, cond_in, touch, touch_mask)["params"])
    ca_params["touch_adaln"] = {
        "kernel": jnp.ones_like(ca_params["touch_adaln"]["kernel"]) * 0.25,
        "bias": jnp.ones_like(ca_params["touch_adaln"]["bias"]) * 0.25,
    }
    ca_base = ca.apply({"params": ca_params}, x_in, cond_in, touch, touch_mask)
    ca_pert = ca.apply({"params": ca_params}, x_in, cond_in, touch.at[:, 0].add(touch_delta), touch_mask)
    ca_diff = np.asarray(jnp.abs(ca_pert - ca_base))
    assert ca_diff[:, 1].max() > 1e-4  # left_thumb is part index 1
    assert ca_diff[:, 0].max() == 0.0  # left_arm
    assert ca_diff[:, 2].max() == 0.0  # left_index


def test_end_to_end_point_cache_to_frozen_encoder_to_touch_actor(tmp_path):
    smap = make_synthetic_sampling_map(8)
    cache = TactilePointCache.create(tmp_path / "cache", {"dtype": "float16", "num_points": 8})
    rng = np.random.default_rng(42)
    ep_frames = rng.integers(0, 200, size=(6, 10, 240, 240), dtype=np.uint8)
    sampled_left = sample_deformation_240(ep_frames[:, :5], smap)
    sampled_right = sample_deformation_240(ep_frames[:, 5:], smap, flip_horizontal=True)
    ep_pts = np.concatenate((sampled_left, sampled_right), axis=1)
    ep_times = np.broadcast_to((np.arange(6, dtype=np.float64) / 30.0)[:, None], (6, 10)).copy()
    cache.put(0, ep_pts, ep_times, np.ones((6, 10), dtype=bool))

    wins = [cache.window(0, query_time=t / 30.0, temporal=TemporalConfig(), tubelet_size=2) for t in (4, 5)]
    left_sig = jnp.asarray(prepare_raw_to_lingbot_signal(np.stack([w["left_raw"] for w in wins]), smap.coverage_mask))
    right_sig = jnp.asarray(prepare_raw_to_lingbot_signal(np.stack([w["right_raw"] for w in wins]), smap.coverage_mask))
    pos_xyz = jnp.asarray(smap.pos_xyz)
    tubelet_valid = jnp.asarray(np.stack([w["tubelet_valid_mask"] for w in wins]))  # [2, 5, 10]
    touch_valid = jnp.repeat(tubelet_valid.transpose((0, 2, 1)), 8, axis=-1)  # [2, 10, 5*8]

    enc_cfg = LingBotTouchConfig(embed_dim=36, depth=2, num_heads=3, tubelet_size=2, output_layers=(0, 1))
    enc = LingBotTouchEncoder(enc_cfg)
    enc_params = enc.init(jax.random.PRNGKey(0), left_sig, pos_xyz, smap.coverage_mask, modality="visuotactile")["params"]
    touch_tokens, touch_pos = encode_bimanual_visuotactile(
        enc_params, enc_cfg, left_sig, right_sig, pos_xyz, smap.coverage_mask
    )
    assert touch_tokens.shape == (2, 10, 5 * 8, 72)
    assert touch_pos.shape == (10, 5 * 8, 3)

    actor_cfg = ProbeConfig(
        architecture="part_attention",
        attention_mask_mode="kinematic",
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=58,
        horizon=6,
    )
    actor = TouchPartAttentionActionProbe(actor_cfg)
    z = jnp.ones((2, 1, 16), dtype=jnp.float32) * 0.1
    prop = jnp.zeros((2, 58), dtype=jnp.float32)
    age = jnp.array([0.01, 0.02], dtype=jnp.float32)
    target = jnp.ones((2, 6, 58), dtype=jnp.float32) * 0.3
    valid = jnp.ones((2, 6), dtype=bool)

    actor_params = actor.init(
        jax.random.PRNGKey(1), z, prop, age, touch_tokens, touch_valid, touch_pos_xyz=touch_pos, train=False
    )["params"]
    tx = optax.adamw(learning_rate=1e-2)
    opt_state = tx.init(actor_params)

    def train_loss(p):
        pred = actor.apply(
            {"params": p}, z, prop, age, touch_tokens, touch_valid, touch_pos_xyz=touch_pos, train=False
        )
        return probe_loss(pred, target, valid)

    losses = []
    for _ in range(2):
        loss_val, grads = jax.value_and_grad(train_loss)(actor_params)
        losses.append(float(loss_val))
        updates, opt_state = tx.update(grads, opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, updates)
    assert np.all(np.isfinite(losses))
    assert losses[1] != losses[0]
