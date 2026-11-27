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

TOUCH_KV_MODES = ("layernorm", "raw_mlp", "raw_gru")


@dataclass(frozen=True)
class TouchCrossAttentionConfig:
    """Hyperparameters for per-finger and global attentive tactile readouts."""

    num_heads: int = 4
    dropout_rate: float = 0.0
    use_pos_xyz: bool = True
    num_global_queries: int = 8
    # "layernorm": LN -> Dense on (already normalized) frozen encoder tokens.
    # "raw_mlp": Dense -> GELU -> Dense without LN, for raw taxel vectors whose magnitude carries force.
    # "raw_gru": per-taxel GRU over the T time-ordered xyz steps of a raw token (C = 3T), zero initial
    #            state per window, last hidden -> Dense. Weights are shared across taxels and fingers.
    kv_mode: str = "layernorm"
    # raw_mlp only: width of the first Dense (None = model dim; keeps legacy H3 parameters unchanged).
    raw_hidden_dim: int | None = None
    # raw_gru only: GRU hidden width.
    gru_hidden_dim: int | None = None
    # raw_* only: expected history frames per raw token (C must equal 3 * raw_frames); None = unchecked.
    raw_frames: int | None = None

    def __post_init__(self):
        if self.num_heads <= 0 or self.num_global_queries <= 0:
            raise ValueError("num_heads and num_global_queries must be positive")
        if self.kv_mode not in TOUCH_KV_MODES:
            raise ValueError(f"kv_mode must be one of {TOUCH_KV_MODES}")
        if (self.gru_hidden_dim is not None) != (self.kv_mode == "raw_gru"):
            raise ValueError("gru_hidden_dim is required for, and only valid with, kv_mode=raw_gru")
        if self.raw_hidden_dim is not None and self.kv_mode != "raw_mlp":
            raise ValueError("raw_hidden_dim is only valid with kv_mode=raw_mlp")
        if self.raw_frames is not None and self.kv_mode not in ("raw_mlp", "raw_gru"):
            raise ValueError("raw_frames is only valid with raw kv modes")
        for name in ("raw_hidden_dim", "gru_hidden_dim", "raw_frames"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.dropout_rate) or not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("dropout_rate must be finite and in [0, 1)")


class _RawGRUEncoder(nn.Module):
    """Shared GRU over time-ordered xyz steps: [N, T, 3] -> last hidden [N, H]; zero state per window."""

    hidden_dim: int

    @nn.compact
    def __call__(self, steps: jax.Array) -> jax.Array:
        cell = nn.GRUCell(self.hidden_dim, name="cell")
        carry = jnp.zeros((steps.shape[0], self.hidden_dim), steps.dtype)
        for t in range(steps.shape[1]):  # oldest -> newest (raw cache layout); T is static and small
            carry, _ = cell(carry, steps[:, t])
        return carry


def _raw_gru_tokens(touch: jax.Array, hidden_dim: int, dim: int) -> jax.Array:
    """Raw tokens [B, F, K, 3T] (per-token xyz blocks ordered oldest->newest) -> kv [B, F, K, dim]."""
    channels = touch.shape[-1]
    if channels % 3:
        raise ValueError(f"raw_gru expects 3*T channels per token, got {channels}")
    steps = touch.astype(jnp.float32).reshape((-1, channels // 3, 3))
    last = _RawGRUEncoder(hidden_dim, name="touch_gru")(steps)
    return nn.Dense(dim, name="touch_proj")(last.reshape((*touch.shape[:-1], hidden_dim)))


class PerFingerTouchCrossAttention(nn.Module):
    """Route frozen finger tactile tokens into matching finger body-part tokens via AdaLN-Zero."""

    dim: int
    parts: tuple[PartSpec, ...]
    num_heads: int = 4
    dropout_rate: float = 0.0
    use_pos_xyz: bool = True
    kv_mode: str = "layernorm"
    raw_hidden_dim: int | None = None
    gru_hidden_dim: int | None = None
    raw_frames: int | None = None

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

        # Only the frozen cache input is stopped; the trainable encoders below receive gradients.
        touch = jax.lax.stop_gradient(touch_tokens)
        raw_mode = self.kv_mode in ("raw_mlp", "raw_gru")
        if raw_mode and self.raw_frames is not None and touch.shape[-1] != 3 * self.raw_frames:
            raise ValueError(f"raw tokens have {touch.shape[-1]} channels; expected 3 * raw_frames={self.raw_frames}")
        if self.kv_mode == "raw_mlp":
            width = self.raw_hidden_dim or self.dim
            hidden = nn.gelu(nn.Dense(width, name="touch_raw_fc1")(touch.astype(jnp.float32)), approximate=False)
            kv = nn.Dense(self.dim, name="touch_proj")(hidden)
        elif self.kv_mode == "raw_gru":
            kv = _raw_gru_tokens(touch, self.gru_hidden_dim, self.dim)
        elif self.kv_mode == "layernorm":
            kv = nn.Dense(self.dim, name="touch_proj")(nn.LayerNorm(name="norm_kv")(touch.astype(jnp.float32)))
        else:
            raise ValueError(f"kv_mode must be one of {TOUCH_KV_MODES}")
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
                kv_mode=self.touch_config.kv_mode,
                raw_hidden_dim=self.touch_config.raw_hidden_dim,
                gru_hidden_dim=self.touch_config.gru_hidden_dim,
                raw_frames=self.touch_config.raw_frames,
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
