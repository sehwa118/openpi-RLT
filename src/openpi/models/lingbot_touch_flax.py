"""Bit-accurate Flax port of Robbyant/lingbot-touch frozen TargetEncoder (HaT-Tiny / HaT-Small)."""

from collections.abc import Mapping
from dataclasses import dataclass
import functools
import json
import math
from pathlib import Path
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

MODALITIES = ("magnetic", "piezoresistive", "visuotactile")
ROPE_MODE = "axial_time_finger_xyz_qk_v1"
ROPE_FIELDS = ("time_theta", "finger_theta", "spatial_theta", "spatial_coordinate_scale")


def _positive_fp32(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    converted = np.float32(parsed)
    if not math.isfinite(parsed) or parsed <= 0 or not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be positive finite FP32")
    return parsed


@dataclass(frozen=True)
class LingBotTouchConfig:
    """Configuration matching Robbyant/lingbot-touch TargetEncoder config.json."""

    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    tubelet_size: int = 2
    mlp_ratio: float = 4.0
    output_layers: tuple[int, ...] = (2, 5, 8, 11)
    group_by_source: tuple[tuple[str, str], ...] = (
        ("magnetic", "magnetic"),
        ("piezoresistive", "piezoresistive"),
        ("visuotactile", "visuotactile"),
    )
    time_theta: float = 10000.0
    finger_theta: float = 10000.0
    spatial_theta: float = 100.0
    spatial_coordinate_scale: float = math.pi / 2
    use_shared_projector: bool = False

    def __post_init__(self):
        for name in ("embed_dim", "depth", "num_heads", "tubelet_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embed_dim % self.num_heads or self.embed_dim // self.num_heads < 10:
            raise ValueError("embed_dim must divide num_heads with head dimension >= 10")
        ratio = _positive_fp32(self.mlp_ratio, "mlp_ratio")
        if int(self.embed_dim * ratio) < 1:
            raise ValueError("mlp_ratio produces an empty hidden layer")
        object.__setattr__(self, "mlp_ratio", ratio)

        layers = tuple(self.output_layers)
        if not layers or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= self.depth for i in layers):
            raise ValueError("output_layers must contain valid zero-based block indices")
        if list(layers) != sorted(set(layers)):
            raise ValueError("output_layers must be unique and increasing")
        object.__setattr__(self, "output_layers", layers)

        routes = dict(self.group_by_source)
        if routes != {name: name for name in MODALITIES}:
            raise ValueError(
                "group_by_source must map magnetic, piezoresistive, and visuotactile to identically named groups"
            )
        object.__setattr__(self, "group_by_source", tuple((k, routes[k]) for k in MODALITIES))

        for field in ROPE_FIELDS:
            object.__setattr__(self, field, _positive_fp32(getattr(self, field), f"address_rope.{field}"))
        if not isinstance(self.use_shared_projector, bool):
            raise ValueError("use_shared_projector must be boolean")

    @property
    def target_dim(self) -> int:
        return self.embed_dim * len(self.output_layers)

    @property
    def route_map(self) -> dict[str, str]:
        return dict(self.group_by_source)

    @classmethod
    def tiny(cls) -> "LingBotTouchConfig":
        return cls()

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "LingBotTouchConfig":
        expected = {
            "embed_dim",
            "depth",
            "num_heads",
            "tubelet_size",
            "mlp_ratio",
            "output_layers",
            "group_by_source",
            "address_rope",
            "use_shared_projector",
        }
        if not isinstance(mapping, Mapping) or set(mapping) != expected:
            raise ValueError(f"config requires exactly {sorted(expected)}")
        rope = mapping["address_rope"]
        if not isinstance(rope, Mapping) or set(rope) != {"mode", *ROPE_FIELDS} or rope["mode"] != ROPE_MODE:
            raise ValueError("address_rope must use the exact axial_time_finger_xyz_qk_v1 schema")
        routes = mapping["group_by_source"]
        if not isinstance(routes, Mapping) or dict(routes) != {name: name for name in MODALITIES}:
            raise ValueError(
                "group_by_source must map magnetic, piezoresistive, and visuotactile to identically named groups"
            )
        layers = mapping["output_layers"]
        if not isinstance(layers, list | tuple):
            raise ValueError("output_layers must be a sequence")
        return cls(
            embed_dim=mapping["embed_dim"],
            depth=mapping["depth"],
            num_heads=mapping["num_heads"],
            tubelet_size=mapping["tubelet_size"],
            mlp_ratio=mapping["mlp_ratio"],
            output_layers=tuple(layers),
            group_by_source=tuple((k, str(routes[k])) for k in MODALITIES),
            time_theta=rope["time_theta"],
            finger_theta=rope["finger_theta"],
            spatial_theta=rope["spatial_theta"],
            spatial_coordinate_scale=rope["spatial_coordinate_scale"],
            use_shared_projector=mapping["use_shared_projector"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "LingBotTouchConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "tubelet_size": self.tubelet_size,
            "mlp_ratio": self.mlp_ratio,
            "output_layers": list(self.output_layers),
            "group_by_source": self.route_map,
            "address_rope": {
                "mode": ROPE_MODE,
                "time_theta": self.time_theta,
                "finger_theta": self.finger_theta,
                "spatial_theta": self.spatial_theta,
                "spatial_coordinate_scale": self.spatial_coordinate_scale,
            },
            "use_shared_projector": self.use_shared_projector,
        }


@dataclass(frozen=True)
class TokenAddress:
    """5D spatiotemporal token coordinates shared across blocks."""

    time_id: jax.Array
    finger_id: jax.Array
    pos_xyz: jax.Array


@dataclass(frozen=True)
class LingBotTouchOutput:
    """Output container mirroring Robbyant/lingbot-touch EncoderOutput."""

    latent: jax.Array
    address: TokenAddress


def _rotate_pairs(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    values_f = values.astype(jnp.float32)
    even = values_f[..., 0::2]
    odd = values_f[..., 1::2]
    return jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(values_f.shape)


def apply_time_finger_xyz_rope(
    q: jax.Array, k: jax.Array, address: TokenAddress, config: LingBotTouchConfig
) -> tuple[jax.Array, jax.Array]:
    """Apply exact 5D axial RoPE (time, finger, x, y, z) matching tactile_jepa_inference/rope.py."""
    batch, _, tokens, head_dim = q.shape
    axis_dim = 2 * ((head_dim // 2) // 5)
    time_b = jnp.broadcast_to(address.time_id[None, :], (batch, tokens))
    finger_b = jnp.broadcast_to(address.finger_id[None, :], (batch, tokens))
    scale_fp32 = float(np.float32(config.spatial_coordinate_scale))

    axes = (
        (config.time_theta, time_b, 1.0),
        (config.finger_theta, finger_b, 1.0),
        (config.spatial_theta, address.pos_xyz[..., 0], scale_fp32),
        (config.spatial_theta, address.pos_xyz[..., 1], scale_fp32),
        (config.spatial_theta, address.pos_xyz[..., 2], scale_fp32),
    )
    q_chunks: list[jax.Array] = []
    k_chunks: list[jax.Array] = []
    start = 0
    exponent = -jnp.arange(0, axis_dim, 2, dtype=jnp.float32) / axis_dim
    for theta, pos, scale in axes:
        end = start + axis_dim
        inv_freq = jnp.power(jnp.asarray(theta, dtype=jnp.float32), exponent)
        phase = pos.astype(jnp.float32)[:, None, :, None] * float(scale) * inv_freq[None, None, None, :]
        cos = jnp.cos(phase)
        sin = jnp.sin(phase)
        q_chunks.append(_rotate_pairs(q[..., start:end], cos, sin))
        k_chunks.append(_rotate_pairs(k[..., start:end], cos, sin))
        start = end
    if start < head_dim:
        q_chunks.append(q[..., start:].astype(jnp.float32))
        k_chunks.append(k[..., start:].astype(jnp.float32))
    return jnp.concatenate(q_chunks, axis=-1).astype(q.dtype), jnp.concatenate(k_chunks, axis=-1).astype(k.dtype)


class PointTokenizer(nn.Module):
    """Per-modality spatiotemporal tubelet MLP tokenizer."""

    embed_dim: int
    tubelet_size: int

    def setup(self):
        self.fc1 = nn.Dense(self.embed_dim)
        self.fc2 = nn.Dense(self.embed_dim)

    def __call__(self, tubelets: jax.Array) -> jax.Array:
        hidden = jax.nn.gelu(self.fc1(tubelets), approximate=False)
        return self.fc2(hidden)


class ModalityRoutedTokenizer(nn.Module):
    """Route inputs to the matching modality tokenizer and initialize all three routes."""

    config: LingBotTouchConfig

    def setup(self):
        self.magnetic = PointTokenizer(self.config.embed_dim, self.config.tubelet_size)
        self.piezoresistive = PointTokenizer(self.config.embed_dim, self.config.tubelet_size)
        self.visuotactile = PointTokenizer(self.config.embed_dim, self.config.tubelet_size)

    def __call__(self, signal: jax.Array, pos_xyz: jax.Array, coverage_mask: Any, *, modality: str):
        if modality not in self.config.route_map:
            raise ValueError(f"modality must be one of {tuple(self.config.route_map)}")
        if isinstance(coverage_mask, jax.core.Tracer):
            raise ValueError("coverage_mask must be static/concrete when tracing LingBotTouchEncoder")
        mask_np = np.asarray(coverage_mask, dtype=bool)
        batch, frames, fingers, taxels, channels = signal.shape
        if mask_np.shape != (fingers, taxels) or not mask_np.any():
            raise ValueError("coverage_mask must be nonempty boolean [5, M]")

        keep = np.flatnonzero(mask_np.reshape(-1))
        spatial = int(keep.size)
        keep_idx = jnp.asarray(keep, dtype=jnp.int32)

        flat_signal = signal.reshape((batch, frames, fingers * taxels, channels))
        compact_signal = jnp.take(flat_signal, keep_idx, axis=2)

        pos = jnp.asarray(pos_xyz, dtype=jnp.float32)
        if pos.ndim == 3:
            pos = jnp.broadcast_to(pos[None], (batch, fingers, taxels, 3))
        compact_pos = jnp.take(pos.reshape((batch, fingers * taxels, 3)), keep_idx, axis=1)

        finger_grid = np.broadcast_to(np.arange(fingers, dtype=np.int32)[:, None], (fingers, taxels)).reshape(-1)
        finger_keep = jnp.asarray(finger_grid[keep], dtype=jnp.int32)

        times = frames // self.config.tubelet_size
        tubelets = (
            compact_signal.reshape((batch, times, self.config.tubelet_size, spatial, channels))
            .transpose((0, 1, 3, 2, 4))
            .reshape((batch, times, spatial, self.config.tubelet_size * channels))
        )

        if self.is_initializing():
            dummy = tubelets[:1, :1, :1]
            for route in MODALITIES:
                _ = getattr(self, route)(dummy)

        tokens = getattr(self, self.config.route_map[modality])(tubelets)
        address = TokenAddress(
            time_id=jnp.repeat(jnp.arange(times, dtype=jnp.int32), spatial),
            finger_id=jnp.tile(finger_keep, (times,)),
            pos_xyz=jnp.broadcast_to(compact_pos[:, None, :, :], (batch, times, spatial, 3)).reshape(
                (batch, times * spatial, 3)
            ),
        )
        return tokens.reshape((batch, times * spatial, -1)), address


class SharedPointwiseProjector(nn.Module):
    """Optional residual MLP applied before transformer blocks."""

    dim: int

    def setup(self):
        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.fc1 = nn.Dense(self.dim)
        self.fc2 = nn.Dense(self.dim)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        hidden = jax.nn.gelu(self.fc1(self.norm(tokens)), approximate=False)
        return tokens + self.fc2(hidden)


class AddressAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection and 5D axial RoPE."""

    config: LingBotTouchConfig

    def setup(self):
        dim = self.config.embed_dim
        self.qkv = nn.Dense(3 * dim)
        self.proj = nn.Dense(dim)

    def __call__(self, x: jax.Array, address: TokenAddress) -> jax.Array:
        batch, tokens, channels = x.shape
        num_heads = self.config.num_heads
        head_dim = channels // num_heads
        qkv = self.qkv(x).reshape((batch, tokens, 3, num_heads, head_dim)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_time_finger_xyz_rope(q, k, address, self.config)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        return self.proj(out.transpose((0, 2, 1, 3)).reshape((batch, tokens, channels)))


class AddressBlock(nn.Module):
    """Pre-LN Transformer block with exact GELU MLP."""

    config: LingBotTouchConfig

    def setup(self):
        dim = self.config.embed_dim
        mlp_dim = int(dim * self.config.mlp_ratio)
        self.norm1 = nn.LayerNorm(epsilon=1e-6)
        self.attn = AddressAttention(self.config)
        self.norm2 = nn.LayerNorm(epsilon=1e-6)
        self.mlp_fc1 = nn.Dense(mlp_dim)
        self.mlp_fc2 = nn.Dense(dim)

    def __call__(self, x: jax.Array, address: TokenAddress) -> jax.Array:
        x = x + self.attn(self.norm1(x), address)
        hidden = jax.nn.gelu(self.mlp_fc1(self.norm2(x)), approximate=False)
        return x + self.mlp_fc2(hidden)


class AddressTransformer(nn.Module):
    """Multi-layer encoder concatenating selected normalized block outputs."""

    config: LingBotTouchConfig

    def setup(self):
        self.blocks = [AddressBlock(self.config) for _ in range(self.config.depth)]
        self.norms_block = [nn.LayerNorm(epsilon=1e-6) for _ in self.config.output_layers]

    def __call__(self, x: jax.Array, address: TokenAddress) -> jax.Array:
        outputs = []
        out_index = 0
        out_set = set(self.config.output_layers)
        for index, block in enumerate(self.blocks):
            x = block(x, address)
            if index in out_set:
                outputs.append(self.norms_block[out_index](x))
                out_index += 1
        return jnp.concatenate(outputs, axis=-1)


class LingBotTouchEncoder(nn.Module):
    """Flax equivalent of Robbyant/lingbot-touch frozen TargetEncoder."""

    config: LingBotTouchConfig

    def setup(self):
        self.tokenizer = ModalityRoutedTokenizer(self.config)
        self.shared_projector = SharedPointwiseProjector(self.config.embed_dim) if self.config.use_shared_projector else None
        self.transformer = AddressTransformer(self.config)

    def __call__(
        self,
        signal: jax.Array,
        pos_xyz: jax.Array,
        coverage_mask: Any,
        *,
        modality: str = "visuotactile",
        stop_gradient: bool = True,
    ) -> LingBotTouchOutput:
        if signal.ndim != 5 or signal.shape[2] != 5 or signal.shape[-1] != 3:
            raise ValueError("signal must have shape [B, T, 5, M, 3]")
        batch, frames, _, taxels, _ = signal.shape
        if batch < 1 or frames < 1 or taxels < 1 or frames % self.config.tubelet_size:
            raise ValueError("B, T and M must be positive and T divisible by tubelet_size")
        if tuple(pos_xyz.shape) not in ((5, taxels, 3), (batch, 5, taxels, 3)):
            raise ValueError("pos_xyz must have shape [5, M, 3] or [B, 5, M, 3]")

        tokens, address = self.tokenizer(signal, pos_xyz, coverage_mask, modality=modality)
        if self.shared_projector is not None:
            tokens = self.shared_projector(tokens)
        latent = self.transformer(tokens, address)
        if stop_gradient:
            latent = jax.lax.stop_gradient(latent)
        return LingBotTouchOutput(latent=latent, address=address)


def _to_numpy(value: Any) -> np.ndarray:
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if not np.isfinite(arr).all():
        raise ValueError("Encoder checkpoint contains nonfinite weights")
    return arr.astype(np.float32).copy()


def _linear_leaf(state: dict[str, np.ndarray], prefix: str, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    w = state[f"{prefix}.weight"]
    b = state[f"{prefix}.bias"]
    if w.shape != (out_dim, in_dim) or b.shape != (out_dim,):
        raise ValueError(f"Unexpected linear shape at {prefix}: {w.shape}, {b.shape}")
    return {"kernel": jnp.asarray(w.T), "bias": jnp.asarray(b)}


def _layernorm_leaf(state: dict[str, np.ndarray], prefix: str, dim: int) -> dict[str, jax.Array]:
    w = state[f"{prefix}.weight"]
    b = state[f"{prefix}.bias"]
    if w.shape != (dim,) or b.shape != (dim,):
        raise ValueError(f"Unexpected LayerNorm shape at {prefix}: {w.shape}, {b.shape}")
    return {"scale": jnp.asarray(w), "bias": jnp.asarray(b)}


def _expected_torch_keys(config: LingBotTouchConfig) -> set[str]:
    keys: set[str] = set()
    for mod in MODALITIES:
        keys.update(
            {
                f"tokenizer.tokenizers.{mod}.signal_mlp.0.weight",
                f"tokenizer.tokenizers.{mod}.signal_mlp.0.bias",
                f"tokenizer.tokenizers.{mod}.signal_mlp.2.weight",
                f"tokenizer.tokenizers.{mod}.signal_mlp.2.bias",
            }
        )
    if config.use_shared_projector:
        keys.update(
            {
                "shared_projector.norm.weight",
                "shared_projector.norm.bias",
                "shared_projector.fc1.weight",
                "shared_projector.fc1.bias",
                "shared_projector.fc2.weight",
                "shared_projector.fc2.bias",
            }
        )
    for i in range(config.depth):
        keys.update(
            {
                f"transformer.blocks.{i}.norm1.weight",
                f"transformer.blocks.{i}.norm1.bias",
                f"transformer.blocks.{i}.attn.qkv.weight",
                f"transformer.blocks.{i}.attn.qkv.bias",
                f"transformer.blocks.{i}.attn.proj.weight",
                f"transformer.blocks.{i}.attn.proj.bias",
                f"transformer.blocks.{i}.norm2.weight",
                f"transformer.blocks.{i}.norm2.bias",
                f"transformer.blocks.{i}.mlp.0.weight",
                f"transformer.blocks.{i}.mlp.0.bias",
                f"transformer.blocks.{i}.mlp.2.weight",
                f"transformer.blocks.{i}.mlp.2.bias",
            }
        )
    # Present on Robbyant's PyTorch AddressTransformer even though output_layers norms produce final outputs.
    keys.update({"transformer.norm.weight", "transformer.norm.bias"})
    for j in range(len(config.output_layers)):
        keys.update({f"transformer.norms_block.{j}.weight", f"transformer.norms_block.{j}.bias"})
    return keys


def convert_torch_state_dict_to_flax(state_dict: Mapping[str, Any], config: LingBotTouchConfig) -> dict[str, Any]:
    """Validate and convert PyTorch LingBot-Touch TargetEncoder state_dict to Flax params."""
    expected = _expected_torch_keys(config)
    if set(state_dict) != expected:
        missing = sorted(expected - set(state_dict))
        extra = sorted(set(state_dict) - expected)
        raise ValueError(f"State dict key mismatch (missing={missing[:5]}, extra={extra[:5]})")

    state = {key: _to_numpy(val) for key, val in state_dict.items()}
    dim = config.embed_dim
    mlp_dim = int(dim * config.mlp_ratio)
    in_dim = config.tubelet_size * 3

    # Verify unused PyTorch norm tensor shapes as an integrity guard.
    _layernorm_leaf(state, "transformer.norm", dim)

    tokenizer_params = {
        mod: {
            "fc1": _linear_leaf(state, f"tokenizer.tokenizers.{mod}.signal_mlp.0", in_dim, dim),
            "fc2": _linear_leaf(state, f"tokenizer.tokenizers.{mod}.signal_mlp.2", dim, dim),
        }
        for mod in MODALITIES
    }

    transformer_params: dict[str, Any] = {}
    for i in range(config.depth):
        prefix = f"transformer.blocks.{i}"
        transformer_params[f"blocks_{i}"] = {
            "norm1": _layernorm_leaf(state, f"{prefix}.norm1", dim),
            "attn": {
                "qkv": _linear_leaf(state, f"{prefix}.attn.qkv", dim, 3 * dim),
                "proj": _linear_leaf(state, f"{prefix}.attn.proj", dim, dim),
            },
            "norm2": _layernorm_leaf(state, f"{prefix}.norm2", dim),
            "mlp_fc1": _linear_leaf(state, f"{prefix}.mlp.0", dim, mlp_dim),
            "mlp_fc2": _linear_leaf(state, f"{prefix}.mlp.2", mlp_dim, dim),
        }
    for j in range(len(config.output_layers)):
        transformer_params[f"norms_block_{j}"] = _layernorm_leaf(state, f"transformer.norms_block.{j}", dim)

    flax_params: dict[str, Any] = {"tokenizer": tokenizer_params, "transformer": transformer_params}
    if config.use_shared_projector:
        flax_params["shared_projector"] = {
            "norm": _layernorm_leaf(state, "shared_projector.norm", dim),
            "fc1": _linear_leaf(state, "shared_projector.fc1", dim, dim),
            "fc2": _linear_leaf(state, "shared_projector.fc2", dim, dim),
        }
    return flax_params


@functools.partial(jax.jit, static_argnames=("config", "coverage_tuple"))
def _encode_bimanual_jit(
    params: Any,
    config: LingBotTouchConfig,
    left_signal: jax.Array,
    right_signal: jax.Array,
    pos_xyz: jax.Array,
    coverage_tuple: tuple[tuple[bool, ...], ...],
) -> tuple[jax.Array, jax.Array]:
    encoder = LingBotTouchEncoder(config)
    left_out = encoder.apply({"params": params}, left_signal, pos_xyz, coverage_tuple, modality="visuotactile")
    right_out = encoder.apply({"params": params}, right_signal, pos_xyz, coverage_tuple, modality="visuotactile")

    batch, frames, _, points, _ = left_signal.shape
    times = frames // config.tubelet_size
    c_dim = config.target_dim

    def _split_fingers(latent: jax.Array) -> jax.Array:
        # [B, times * 5 * P, C] -> [B, 5, times * P, C]
        return latent.reshape((batch, times, 5, points, c_dim)).transpose((0, 2, 1, 3, 4)).reshape(
            (batch, 5, times * points, c_dim)
        )

    touch_tokens = jnp.concatenate((_split_fingers(left_out.latent), _split_fingers(right_out.latent)), axis=1)
    if pos_xyz.ndim == 3:
        per_hand_pos = jnp.broadcast_to(pos_xyz[:, None, :, :], (5, times, points, 3)).reshape((5, times * points, 3))
        touch_pos = jnp.concatenate((per_hand_pos, per_hand_pos), axis=0)
    else:
        per_hand_pos = jnp.broadcast_to(pos_xyz[:, :, None, :, :], (batch, 5, times, points, 3)).reshape(
            (batch, 5, times * points, 3)
        )
        touch_pos = jnp.concatenate((per_hand_pos, per_hand_pos), axis=1)
    return touch_tokens, touch_pos


def encode_bimanual_visuotactile(
    params: Any,
    config: LingBotTouchConfig,
    left_signal: jax.Array,
    right_signal: jax.Array,
    pos_xyz: jax.Array,
    coverage_mask: Any,
) -> tuple[jax.Array, jax.Array]:
    """Encode left/right hands independently under canonical-right geometry and reorder per finger.

    Requires full five-finger valid coverage [5, P] (all True), matching Sharpa pretraining samples.
    Returns (touch_tokens [B, 10, T_tub*P, C], touch_pos_xyz [10, T_tub*P, 3] or [B, 10, T_tub*P, 3]).
    """
    mask_np = np.asarray(coverage_mask, dtype=bool)
    if mask_np.ndim != 2 or mask_np.shape[0] != 5 or not mask_np.all():
        raise ValueError("encode_bimanual_visuotactile requires full [5, P] True coverage_mask")
    coverage_tuple = tuple(tuple(bool(cell) for cell in row) for row in mask_np)
    return _encode_bimanual_jit(params, config, left_signal, right_signal, pos_xyz, coverage_tuple)
