"""Stage B action probe / actor architectures conditioned on frozen Stage A latents."""

from dataclasses import dataclass
from typing import Literal

from flax import linen as nn
import jax
import jax.numpy as jnp

from openpi.trlt.config import ProbeConfig


@dataclass(frozen=True)
class PartSpec:
    """Contiguous slice of the state/action vector belonging to one body part."""

    name: str
    start: int
    end: int
    side: Literal["left", "right", "single"]
    role: Literal["arm", "finger"]
    parent: str | None = None

    @property
    def dim(self) -> int:
        return self.end - self.start


# T-Rex 58D layout: left arm (0..6), 5 left fingers (7..28), right arm (29..35), 5 right fingers (36..57).
TREX_PARTS: tuple[PartSpec, ...] = (
    PartSpec("left_arm", 0, 7, "left", "arm", None),
    PartSpec("left_thumb", 7, 12, "left", "finger", "left_arm"),
    PartSpec("left_index", 12, 16, "left", "finger", "left_arm"),
    PartSpec("left_middle", 16, 20, "left", "finger", "left_arm"),
    PartSpec("left_ring", 20, 24, "left", "finger", "left_arm"),
    PartSpec("left_pinky", 24, 29, "left", "finger", "left_arm"),
    PartSpec("right_arm", 29, 36, "right", "arm", None),
    PartSpec("right_thumb", 36, 41, "right", "finger", "right_arm"),
    PartSpec("right_index", 41, 45, "right", "finger", "right_arm"),
    PartSpec("right_middle", 45, 49, "right", "finger", "right_arm"),
    PartSpec("right_ring", 49, 53, "right", "finger", "right_arm"),
    PartSpec("right_pinky", 53, 58, "right", "finger", "right_arm"),
)

# OMY+LEAP 22D layout: 6D arm (0..5), 4x4D LEAP fingers (6..21).
OMY_LEAP_PARTS: tuple[PartSpec, ...] = (
    PartSpec("arm", 0, 6, "single", "arm", None),
    PartSpec("index", 6, 10, "single", "finger", "arm"),
    PartSpec("middle", 10, 14, "single", "finger", "arm"),
    PartSpec("ring", 14, 18, "single", "finger", "arm"),
    PartSpec("thumb", 18, 22, "single", "finger", "arm"),
)


def part_specs_for_dim(action_dim: int) -> tuple[PartSpec, ...]:
    """Return canonical body-part slice specifications for supported embodiments."""
    if action_dim == 58:
        return TREX_PARTS
    if action_dim == 22:
        return OMY_LEAP_PARTS
    raise ValueError(f"No body-part decomposition defined for action_dim={action_dim}")


def part_attention_mask(parts: tuple[PartSpec, ...], mode: str) -> jax.Array:
    """Return [N, N] boolean mask where entry (i, j) allows query part i to attend to key part j."""
    count = len(parts)
    if mode == "full":
        return jnp.ones((count, count), dtype=bool)
    if mode == "kinematic":
        matrix = [
            [
                i == j
                or pi.parent == pj.name
                or pj.parent == pi.name
                or (pi.role == "arm" and pj.role == "arm")
                for j, pj in enumerate(parts)
            ]
            for i, pi in enumerate(parts)
        ]
        return jnp.asarray(matrix, dtype=bool)
    if mode == "hierarchical":
        matrix = [
            [(pj.role == "arm") if pi.role == "arm" else (pj.side == pi.side) for pj in parts]
            for pi in parts
        ]
        return jnp.asarray(matrix, dtype=bool)
    raise ValueError(f"Unknown attention_mask_mode: {mode}")


class ActionProbe(nn.Module):
    """Legacy Stage B MLP action probe preserving exact Linen parameter names."""

    config: ProbeConfig

    @nn.compact
    def __call__(self, z, proprioception, visual_age_seconds, *, train=False):
        z = jax.lax.stop_gradient(z)
        age = visual_age_seconds.reshape((z.shape[0], 1))
        x = jnp.concatenate((z.reshape((z.shape[0], -1)), proprioception, age), axis=-1)
        for index in range(self.config.depth):
            x = nn.Dense(self.config.hidden_dim, name=f"hidden_{index}")(x)
            x = nn.gelu(x)
            x = nn.Dropout(self.config.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.config.horizon * self.config.action_dim, name="output")(x)
        return x.reshape((x.shape[0], self.config.horizon, self.config.action_dim))


class ResMLPActionProbe(nn.Module):
    """Pre-LN Residual MLP baseline parameter-matched to the part-attention actor."""

    config: ProbeConfig

    @nn.compact
    def __call__(self, z, proprioception, visual_age_seconds, *, train=False):
        z = jax.lax.stop_gradient(z)
        age = visual_age_seconds.reshape((z.shape[0], 1))
        x = jnp.concatenate((z.reshape((z.shape[0], -1)), proprioception, age), axis=-1)
        x = nn.Dense(self.config.hidden_dim, name="input_proj")(x)
        for index in range(self.config.depth):
            h = nn.LayerNorm(name=f"block_{index}_norm")(x)
            h = nn.Dense(self.config.hidden_dim, name=f"block_{index}_fc1")(h)
            h = nn.gelu(h)
            h = nn.Dropout(self.config.dropout_rate)(h, deterministic=not train)
            h = nn.Dense(self.config.hidden_dim, name=f"block_{index}_fc2")(h)
            h = nn.Dropout(self.config.dropout_rate)(h, deterministic=not train)
            x = x + h
        x = nn.LayerNorm(name="final_norm")(x)
        x = nn.Dense(self.config.horizon * self.config.action_dim, name="output")(x)
        return x.reshape((x.shape[0], self.config.horizon, self.config.action_dim))


class AdaLNTransformerBlock(nn.Module):
    """Pre-LN Transformer block with DiT-style zero-initialized AdaLN conditioning."""

    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, cond, mask, *, train=False):
        mod = nn.Dense(
            6 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="adaln_modulation",
        )(nn.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod[:, None, :], 6, axis=-1)

        h = nn.LayerNorm(use_scale=False, use_bias=False, name="norm1")(x)
        h = h * (1.0 + scale_msa) + shift_msa
        head_dim = self.dim // self.num_heads
        q = nn.DenseGeneral((self.num_heads, head_dim), name="query")(h)
        k = nn.DenseGeneral((self.num_heads, head_dim), name="key")(h)
        v = nn.DenseGeneral((self.num_heads, head_dim), name="value")(h)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim**-0.5)
        attn_mask = jnp.broadcast_to(mask[None, None, :, :], logits.shape)
        weights = jax.nn.softmax(jnp.where(attn_mask, logits, jnp.finfo(logits.dtype).min), axis=-1)
        weights = jnp.where(attn_mask, weights, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-20)
        weights = nn.Dropout(self.dropout_rate)(weights, deterministic=not train)
        attn_out = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(h.shape)
        attn_out = nn.Dense(self.dim, name="attn_out")(attn_out)
        attn_out = nn.Dropout(self.dropout_rate)(attn_out, deterministic=not train)
        x = x + gate_msa * attn_out

        h = nn.LayerNorm(use_scale=False, use_bias=False, name="norm2")(x)
        h = h * (1.0 + scale_mlp) + shift_mlp
        mlp_dim = int(self.dim * self.mlp_ratio)
        h = nn.Dense(mlp_dim, name="ffn_fc1")(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not train)
        h = nn.Dense(self.dim, name="ffn_fc2")(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not train)
        return x + gate_mlp * h


class PartAttentionActionProbe(nn.Module):
    """Body-part Transformer action actor modulated by frozen Stage A visuotactile latents."""

    config: ProbeConfig

    @nn.compact
    def __call__(self, z, proprioception, visual_age_seconds, *, train=False):
        cfg = self.config
        parts = part_specs_for_dim(cfg.action_dim)
        if proprioception.shape[-1] != cfg.action_dim:
            raise ValueError(f"Expected proprioception dim {cfg.action_dim}, got {proprioception.shape[-1]}")

        z = jax.lax.stop_gradient(z)
        batch_size = z.shape[0]
        age = visual_age_seconds.reshape((batch_size, 1))
        # Conditioning depends strictly on frozen visuotactile z and visual age, preserving part mask isolation.
        cond_in = jnp.concatenate((z.reshape((batch_size, -1)), age), axis=-1)
        cond = nn.Dense(cfg.hidden_dim, name="cond_fc1")(cond_in)
        cond = nn.silu(cond)
        cond = nn.Dense(cfg.hidden_dim, name="cond_fc2")(cond)

        tokens = [
            nn.Dense(cfg.hidden_dim, name=f"part_in_{part.name}")(proprioception[:, part.start : part.end])
            for part in parts
        ]
        x = jnp.stack(tokens, axis=1)
        part_embed = self.param("part_embed", nn.initializers.normal(stddev=0.02), (len(parts), cfg.hidden_dim))
        x = x + part_embed[None, :, :]

        mask = part_attention_mask(parts, cfg.attention_mask_mode)
        for index in range(cfg.depth):
            x = AdaLNTransformerBlock(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout_rate=cfg.dropout_rate,
                name=f"block_{index}",
            )(x, cond, mask, train=train)

        final_mod = nn.Dense(
            2 * cfg.hidden_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="final_adaln",
        )(nn.silu(cond))
        shift_out, scale_out = jnp.split(final_mod[:, None, :], 2, axis=-1)
        x = nn.LayerNorm(use_scale=False, use_bias=False, name="final_norm")(x)
        x = x * (1.0 + scale_out) + shift_out

        outputs = [
            nn.Dense(cfg.horizon * part.dim, name=f"part_out_{part.name}")(x[:, index, :]).reshape(
                (batch_size, cfg.horizon, part.dim)
            )
            for index, part in enumerate(parts)
        ]
        return jnp.concatenate(outputs, axis=-1)


def build_action_probe(config: ProbeConfig) -> nn.Module:
    """Build Stage B probe/actor module from ProbeConfig."""
    if config.architecture == "mlp":
        return ActionProbe(config)
    if config.architecture == "resmlp":
        return ResMLPActionProbe(config)
    if config.architecture == "part_attention":
        return PartAttentionActionProbe(config)
    if config.architecture == "touch_part_attention":
        from openpi.models.tactile_readout import TouchCrossAttentionConfig
        from openpi.models.tactile_readout import TouchPartAttentionActionProbe

        return TouchPartAttentionActionProbe(
            config,
            TouchCrossAttentionConfig(
                num_heads=config.touch_num_heads,
                use_pos_xyz=config.touch_use_pos_xyz,
                kv_mode=config.touch_kv_mode,
                raw_hidden_dim=config.touch_raw_hidden_dim,
                gru_hidden_dim=config.touch_gru_hidden_dim,
                raw_frames=config.touch_raw_frames,
            ),
        )
    raise ValueError(f"Unknown probe architecture: {config.architecture}")
