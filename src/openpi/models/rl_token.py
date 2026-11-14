import dataclasses
from typing import Any

import chex
from einops import einops
from flax import linen as nn
import jax
import jax.numpy as jnp

from openpi.models.utils.fsq_tokenizer import CrossAttentionLayer
from openpi.models.utils.fsq_tokenizer import sinusoidal_pe_init


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 512  # Internal dimension of encoder-decoder (lightweight)
    input_dim: int = 2048  # VLA prefix embedding dimension (Gemma 2B hidden size)
    mlp_ratio: float = 4.0
    num_heads: int = 8
    dropout_rate: float = 0.0


class RLTokenEncoder(nn.Module):
    """Compresses VLA prefix embeddings [b, seq, input_dim] into RL tokens [b, num_rl_tokens, embed_dim]."""

    config: RLTokenConfig

    @nn.compact
    def __call__(
        self,
        prefix_embs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config

        # Invalid source values must not enter projections, including NaN padding.
        if mask is not None:
            prefix_embs = jnp.where(mask[..., None], prefix_embs, 0.0)

        # Project from VLA dim to internal dim if they differ
        if cfg.input_dim != cfg.embed_dim:
            prefix_embs = nn.Dense(cfg.embed_dim, name="input_proj")(prefix_embs)

        x = self.param("q_embed", sinusoidal_pe_init, (cfg.num_rl_tokens, cfg.embed_dim))
        x = jnp.broadcast_to(x, prefix_embs.shape[:-2] + x.shape[-2:])

        if mask is not None:
            chex.assert_equal_shape([prefix_embs[..., 0], mask])
            attn_mask = einops.repeat(mask, "... kv -> ... 1 q kv", q=cfg.num_rl_tokens)
        else:
            attn_mask = jnp.ones((*prefix_embs.shape[:-2], 1, cfg.num_rl_tokens, prefix_embs.shape[-2]))

        y = prefix_embs + self.param("y_pos_enc", sinusoidal_pe_init, prefix_embs.shape[-2:])
        if mask is not None:
            y = jnp.where(mask[..., None], y, 0.0)

        for _ in range(cfg.num_layers):
            x = CrossAttentionLayer(
                dropout_rate=cfg.dropout_rate,
                num_heads=cfg.num_heads,
                causal=False,
                mlp_ratio=cfg.mlp_ratio,
            )(x, y, train=train, mask_self=None, mask_cross=attn_mask)

        return x


class RLTokenDecoder(nn.Module):
    """Reconstructs prefix embeddings [b, seq, input_dim] from RL tokens [b, num_rl_tokens, embed_dim]."""

    config: RLTokenConfig

    @nn.compact
    def __call__(
        self,
        rl_tokens: jnp.ndarray,
        target_seq_len: int,
        *,
        train: bool = True,
        target_valid_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        cfg = self.config

        x = self.param("q_embed", sinusoidal_pe_init, (target_seq_len, cfg.embed_dim))
        x = jnp.broadcast_to(x, rl_tokens.shape[:-2] + x.shape[-2:])

        attn_mask = jnp.ones((*rl_tokens.shape[:-2], 1, target_seq_len, cfg.num_rl_tokens))

        y = rl_tokens + self.param("y_pos_enc", sinusoidal_pe_init, rl_tokens.shape[-2:])

        query_self_mask = None
        if target_valid_mask is not None:
            chex.assert_shape(target_valid_mask, (*rl_tokens.shape[:-2], target_seq_len))
            query_self_mask = target_valid_mask[..., None, :, None] & target_valid_mask[..., None, None, :]

        for _ in range(cfg.num_layers):
            x = CrossAttentionLayer(
                dropout_rate=cfg.dropout_rate,
                num_heads=cfg.num_heads,
                causal=False,
                mlp_ratio=cfg.mlp_ratio,
            )(x, y, train=train, mask_self=query_self_mask, mask_cross=attn_mask)

        # Project back to VLA dim if they differ
        if cfg.input_dim != cfg.embed_dim:
            x = nn.Dense(cfg.input_dim, name="output_proj")(x)

        return x


class RLTokenModel(nn.Module):
    """RL Token encoder-decoder: compresses VLA prefix embeddings into RL tokens via cross-attention."""

    config: RLTokenConfig

    def setup(self):
        self.encoder = RLTokenEncoder(config=self.config)
        self.decoder = RLTokenDecoder(config=self.config)

    def encode(
        self,
        prefix_embs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> jnp.ndarray:
        return self.encoder(prefix_embs, mask, train=train)

    def decode(
        self,
        rl_tokens: jnp.ndarray,
        target_seq_len: int,
        *,
        train: bool = True,
        target_valid_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self.decoder(rl_tokens, target_seq_len, train=train, target_valid_mask=target_valid_mask)

    def loss(
        self,
        prefix_embs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        rl_tokens = self.encode(prefix_embs, mask, train=train)

        target_seq_len = prefix_embs.shape[-2]
        reconstructed = self.decode(rl_tokens, target_seq_len, train=train, target_valid_mask=mask)

        target = jax.lax.stop_gradient(prefix_embs)
        if mask is not None:
            target = jnp.where(mask[..., None], target, 0.0)
        sq_error = jnp.square(reconstructed - target)

        if mask is not None:
            mask_expanded = mask[..., None].astype(sq_error.dtype)
            masked_sq_error = sq_error * mask_expanded
            num_valid = jnp.sum(mask_expanded) * prefix_embs.shape[-1]
            mse = jnp.sum(masked_sq_error) / jnp.maximum(num_valid, 1.0)
        else:
            mse = jnp.mean(sq_error)

        return mse, {"mse": mse}

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Dummy for .init"""
        return self.loss(*args, **kwargs)
