"""Attentive tactile readouts and per-finger LingBot-Touch cross-attention for Stage B."""

from dataclasses import dataclass
import math

from flax import linen as nn
import jax
import jax.numpy as jnp

from openpi.models.trlt_actor import AdaLNTransformerBlock
from openpi.models.trlt_actor import PartSpec
from openpi.models.trlt_actor import part_attention_mask
from openpi.models.trlt_actor import part_specs_for_dim
from openpi.trlt.config import ProbeConfig


@dataclass(frozen=True)
class TouchCrossAttentionConfig:
    """Hyperparameters for per-finger and global attentive tactile readouts."""

    num_heads: int = 4
    dropout_rate: float = 0.0
    use_pos_xyz: bool = True
    num_global_queries: int = 8

    def __post_init__(self):
        if self.num_heads <= 0 or self.num_global_queries <= 0:
            raise ValueError("num_heads and num_global_queries must be positive")
        if not math.isfinite(self.dropout_rate) or not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("dropout_rate must be finite and in [0, 1)")


class PerFingerTouchCrossAttention(nn.Module):
    """Route frozen finger tactile tokens into matching finger body-part tokens via AdaLN-Zero."""

    dim: int
    parts: tuple[PartSpec, ...]
    num_heads: int = 4
    dropout_rate: float = 0.0
    use_pos_xyz: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        cond: jax.Array,
        touch_tokens: jax.Array,
        touch_valid_mask: jax.Array,
        *,
        touch_pos_xyz: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        finger_indices = [i for i, part in enumerate(self.parts) if part.role == "finger"]
        if not finger_indices:
            raise ValueError("parts must contain at least one finger part")
        if touch_tokens.ndim != 4 or touch_tokens.shape[1] != len(finger_indices):
            raise ValueError(f"Expected touch_tokens [B, {len(finger_indices)}, K, C], got {touch_tokens.shape}")
        if touch_valid_mask.shape != touch_tokens.shape[:3]:
            raise ValueError("touch_valid_mask must match leading [B, F, K] shape of touch_tokens")

        batch = x.shape[0]
        num_fingers = len(finger_indices)
        mod = nn.Dense(
            3 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="touch_adaln",
        )(nn.silu(cond))
        shift_ca, scale_ca, gate_ca = jnp.split(mod[:, None, :], 3, axis=-1)

        finger_tokens = x[:, finger_indices, :]
        q_norm = nn.LayerNorm(use_scale=False, use_bias=False, name="norm_q")(finger_tokens)
        q_in = q_norm * (1.0 + scale_ca) + shift_ca

        touch = jax.lax.stop_gradient(touch_tokens)
        kv = nn.Dense(self.dim, name="touch_proj")(nn.LayerNorm(name="norm_kv")(touch))
        if self.use_pos_xyz and touch_pos_xyz is not None:
            pos = jnp.asarray(jax.lax.stop_gradient(touch_pos_xyz), dtype=kv.dtype)
            if pos.ndim == 3:
                pos = jnp.broadcast_to(pos[None], (batch, num_fingers, touch_tokens.shape[2], 3))
            kv = kv + nn.Dense(self.dim, name="pos_proj")(pos)

        head_dim = self.dim // self.num_heads
        q = nn.DenseGeneral((self.num_heads, head_dim), name="query")(q_in)
        k = nn.DenseGeneral((self.num_heads, head_dim), name="key")(kv)
        v = nn.DenseGeneral((self.num_heads, head_dim), name="value")(kv)

        logits = jnp.einsum("bfhd,bfkhd->bfhk", q, k) * (head_dim**-0.5)
        valid_mask = touch_valid_mask.astype(bool)
        attn_mask = jnp.broadcast_to(valid_mask[:, :, None, :], logits.shape)
        weights = jax.nn.softmax(jnp.where(attn_mask, logits, jnp.finfo(logits.dtype).min), axis=-1)
        weights = jnp.where(attn_mask, weights, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-20)
        weights = nn.Dropout(self.dropout_rate)(weights, deterministic=not train)

        context = jnp.einsum("bfhk,bfkhd->bfhd", weights, v).reshape((batch, num_fingers, self.dim))
        out = nn.Dense(self.dim, name="touch_out")(context)
        out = nn.Dropout(self.dropout_rate)(out, deterministic=not train)
        has_valid = jnp.any(valid_mask, axis=-1)[..., None]
        out = jnp.where(has_valid, out, 0.0)

        full_update = jnp.zeros_like(x).at[:, finger_indices, :].set(gate_ca * out)
        return x + full_update


class GlobalTouchAttentiveReadout(nn.Module):
    """Learnable query cross-attention over LingBot-Touch tokens (recommended readout over pooling)."""

    embed_dim: int
    num_queries: int = 8
    num_heads: int = 4
    dropout_rate: float = 0.0
    use_pos_xyz: bool = True

    @nn.compact
    def __call__(
        self,
        touch_tokens: jax.Array,
        valid_mask: jax.Array,
        *,
        pos_xyz: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if touch_tokens.ndim != 3 or valid_mask.shape != touch_tokens.shape[:2]:
            raise ValueError("Expected touch_tokens [B, N, C] and valid_mask [B, N]")

        batch = touch_tokens.shape[0]
        queries = self.param("queries", nn.initializers.normal(0.02), (self.num_queries, self.embed_dim))
        q_base = jnp.broadcast_to(queries[None], (batch, self.num_queries, self.embed_dim))
        q_in = nn.LayerNorm(name="norm_q")(q_base)

        touch = jax.lax.stop_gradient(touch_tokens)
        kv = nn.Dense(self.embed_dim, name="touch_proj")(nn.LayerNorm(name="norm_kv")(touch))
        if self.use_pos_xyz and pos_xyz is not None:
            pos = jnp.asarray(jax.lax.stop_gradient(pos_xyz), dtype=kv.dtype)
            if pos.ndim == 2:
                pos = jnp.broadcast_to(pos[None], (batch, touch_tokens.shape[1], 3))
            kv = kv + nn.Dense(self.embed_dim, name="pos_proj")(pos)

        head_dim = self.embed_dim // self.num_heads
        q = nn.DenseGeneral((self.num_heads, head_dim), name="query")(q_in)
        k = nn.DenseGeneral((self.num_heads, head_dim), name="key")(kv)
        v = nn.DenseGeneral((self.num_heads, head_dim), name="value")(kv)

        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim**-0.5)
        valid = valid_mask.astype(bool)
        attn_mask = jnp.broadcast_to(valid[:, None, None, :], logits.shape)
        weights = jax.nn.softmax(jnp.where(attn_mask, logits, jnp.finfo(logits.dtype).min), axis=-1)
        weights = jnp.where(attn_mask, weights, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-20)
        weights = nn.Dropout(self.dropout_rate)(weights, deterministic=not train)

        context = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape((batch, self.num_queries, self.embed_dim))
        update = nn.Dense(self.embed_dim, name="out_proj")(context)
        update = nn.Dropout(self.dropout_rate)(update, deterministic=not train)
        has_valid = jnp.any(valid, axis=-1)[:, None, None]
        return q_base + jnp.where(has_valid, update, 0.0)


class TouchPartAttentionActionProbe(nn.Module):
    """Stage B body-part actor combining AdaLN-Zero z conditioning and per-finger touch cross-attention."""

    config: ProbeConfig
    touch_config: TouchCrossAttentionConfig = TouchCrossAttentionConfig()

    @nn.compact
    def __call__(
        self,
        z: jax.Array,
        proprioception: jax.Array,
        visual_age_seconds: jax.Array,
        touch_tokens: jax.Array,
        touch_valid_mask: jax.Array,
        *,
        touch_pos_xyz: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        cfg = self.config
        parts = part_specs_for_dim(cfg.action_dim)
        if proprioception.shape[-1] != cfg.action_dim:
            raise ValueError(f"Expected proprioception dim {cfg.action_dim}, got {proprioception.shape[-1]}")

        z = jax.lax.stop_gradient(z)
        touch_tokens = jax.lax.stop_gradient(touch_tokens)
        batch_size = z.shape[0]
        age = visual_age_seconds.reshape((batch_size, 1))
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
            touch_dropout = (
                self.touch_config.dropout_rate if self.touch_config.dropout_rate > 0 else cfg.dropout_rate
            )
            x = PerFingerTouchCrossAttention(
                dim=cfg.hidden_dim,
                parts=parts,
                num_heads=self.touch_config.num_heads,
                dropout_rate=touch_dropout,
                use_pos_xyz=self.touch_config.use_pos_xyz,
                name=f"touch_ca_{index}",
            )(x, cond, touch_tokens, touch_valid_mask, touch_pos_xyz=touch_pos_xyz, train=train)

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
