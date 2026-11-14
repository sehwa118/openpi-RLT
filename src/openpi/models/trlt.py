"""Pure tensor TRLT encoders and reconstruction objectives.

All fusion variants share a parameterization and differ only in a directed attention mask.
The original RLT encoder and visual decoder retain their Linen parameter names.
"""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from openpi.models.rl_token import RLTokenConfig
from openpi.models.rl_token import RLTokenDecoder
from openpi.models.rl_token import RLTokenEncoder
from openpi.trlt.config import ModelConfig

Batch = dict[str, Any]


def fusion_attention_mask(mode: str, stage: int, nv: int, nu: int) -> jax.Array:
    """Return [V,U,q] query/key connectivity, before padding masks."""
    n = nv + nu + 1
    visual = jnp.arange(n) < nv
    other = (jnp.arange(n) >= nv) & (jnp.arange(n) < n - 1)
    same = (visual[:, None] & visual[None, :]) | (other[:, None] & other[None, :])
    cross = (visual[:, None] & other[None, :]) | (other[:, None] & visual[None, :])
    if mode == "trlt_self_only":
        mask = same | cross
    elif mode == "trlt_cross_only":
        mask = cross
    elif mode == "trlt_alternating":
        mask = same if stage % 2 == 0 else cross
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
    return mask.at[-1, :].set(True)


class SafeAttention(nn.Module):
    """Dense attention with exactly zero update for an all-masked row."""

    dim: int
    heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, query, memory, mask, *, train=False):
        head_dim = self.dim // self.heads
        q = nn.DenseGeneral((self.heads, head_dim), name="query")(query)
        k = nn.DenseGeneral((self.heads, head_dim), name="key")(memory)
        v = nn.DenseGeneral((self.heads, head_dim), name="value")(memory)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * head_dim**-0.5
        mask = jnp.broadcast_to(mask[:, None], logits.shape)
        # Finite sentinel + explicit renormalization prevents softmax(all -inf).
        weights = jax.nn.softmax(jnp.where(mask, logits, jnp.finfo(logits.dtype).min), axis=-1)
        weights = jnp.where(mask, weights, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-20)
        weights = nn.Dropout(self.dropout)(weights, deterministic=not train)
        context = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(query.shape)
        update = nn.Dense(self.dim, name="output")(context)
        return jnp.where(jnp.any(mask, axis=(1, 3))[..., None], update, 0.0)


class FusionStage(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask, valid, *, train=False):
        cfg = self.config
        # One synchronous update from the same old normalized V/U/q state.
        old = nn.LayerNorm(name="attention_norm")(x)
        x = x + SafeAttention(cfg.embed_dim, cfg.num_heads, cfg.dropout_rate, name="attention")(
            old, old, mask, train=train
        )
        y = nn.LayerNorm(name="ffn_norm")(x)
        y = nn.Dense(int(cfg.embed_dim * cfg.mlp_ratio), name="ffn_in")(y)
        y = nn.gelu(y)
        y = nn.Dropout(cfg.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(cfg.embed_dim, name="ffn_out")(y)
        return jnp.where(valid[..., None], x + y, 0.0)


class Fusion(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, visual, other, visual_valid, other_valid, *, train=False, query_override=None):
        cfg = self.config
        q = self.param("q_embed", nn.initializers.normal(0.02), (1, cfg.embed_dim))
        q = jnp.broadcast_to(q, (visual.shape[0], 1, cfg.embed_dim))
        if query_override is not None:
            q = query_override
        valid = jnp.concatenate((visual_valid, other_valid, jnp.ones(q.shape[:2], dtype=bool)), axis=1)
        x = jnp.concatenate((visual, other, q), axis=1)
        x = jnp.where(valid[..., None], x, 0.0)
        for stage in range(cfg.num_stages):
            allowed = fusion_attention_mask(cfg.mode, stage, visual.shape[1], other.shape[1])
            mask = allowed[None] & valid[:, :, None] & valid[:, None, :]
            x = FusionStage(cfg, name=f"stage_{stage}")(x, mask, valid, train=train)
        return x[:, -1:], x[:, :-1]


class TactileCNN(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, tactile, pixel_valid):
        cfg = self.config
        b, length, sensors, h, w, channels = tactile.shape
        pixels = jnp.where(pixel_valid, tactile, 0.0).reshape((-1, h, w, channels))
        valid = jnp.any(pixel_valid, axis=-1).reshape((-1, h, w, 1))
        yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, h), jnp.linspace(-1, 1, w), indexing="ij")
        xy = jnp.broadcast_to(jnp.stack((xx, yy), axis=-1), (*pixels.shape[:-1], 2))
        x = jnp.concatenate((pixels, jnp.where(valid, xy, 0.0)), axis=-1)
        for index, width in enumerate(cfg.cnn_channels):
            x = nn.Conv(width, (3, 3), strides=(2, 2), padding="SAME", name=f"conv_{index}")(x)
            x = nn.LayerNorm(name=f"norm_{index}")(x)
            x = nn.gelu(x)
        # Flatten retains local contact position; global average pooling would erase it.
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(cfg.embed_dim, name="spatial_readout")(x)
        return x.reshape((b, length, sensors, cfg.embed_dim))


def tactile_masks(batch: Batch):
    tactile = batch["tactile"]
    frame = batch["tactile_valid_mask"].astype(bool)
    spatial = jnp.broadcast_to(batch.get("spatial_valid_mask", jnp.ones((), bool)), tactile.shape)
    pixel = spatial & frame[..., None, None, None]
    frame = frame & jnp.any(pixel, axis=(-3, -2, -1))
    return frame, pixel


def relative_times(batch: Batch):
    ts = batch["tactile_timestamps"]
    if ts.ndim == 2:
        ts = jnp.broadcast_to(ts[..., None], batch["tactile_valid_mask"].shape)
    valid, _ = tactile_masks(batch)
    return jnp.where(valid, ts - batch["query_timestamp"][:, None, None], 0.0)


class TRLTEncoder(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, batch: Batch, *, train=False):
        cfg = self.config
        visual_valid = batch["visual_valid_mask"].astype(bool)
        visual = jnp.where(visual_valid[..., None], batch["visual_features"], 0.0)
        nv = visual.shape[1]
        if nv % cfg.num_cameras:
            raise ValueError("Visual token count must be divisible by camera count")
        patches = nv // cfg.num_cameras
        visual = nn.Dense(cfg.embed_dim, name="visual_projection")(visual)
        camera = self.param("camera_embed", nn.initializers.normal(0.02), (cfg.num_cameras, cfg.embed_dim))
        patch = self.param("patch_embed", nn.initializers.normal(0.02), (patches, cfg.embed_dim))
        visual = visual + (camera[:, None] + patch[None]).reshape((1, nv, cfg.embed_dim))
        modality = self.param("modality_embed", nn.initializers.normal(0.02), (3, cfg.embed_dim))
        visual = visual + modality[0]
        frame_valid, pixel_valid = tactile_masks(batch)
        tactile = TactileCNN(cfg, name="tactile_cnn")(batch["tactile"], pixel_valid)
        length, sensors = tactile.shape[1:3]
        if sensors != cfg.num_sensors or length > cfg.history_frames:
            raise ValueError("Tactile sensor/history shape differs from model config")
        sensor = self.param("sensor_embed", nn.initializers.normal(0.02), (cfg.num_sensors, cfg.embed_dim))
        relative = relative_times(batch)[..., None]
        time = nn.Dense(cfg.embed_dim, name="time_in")(relative)
        time = nn.Dense(cfg.embed_dim, name="time_out")(nn.gelu(time))
        tactile = tactile + sensor[None, None] + modality[1] + time
        tactile = tactile.reshape((tactile.shape[0], length * sensors, cfg.embed_dim))
        age = (batch["query_timestamp"] - batch["visual_timestamp"])[..., None]
        if batch["proprioception"].shape[-1] != cfg.state_dim:
            raise ValueError("State dimension mismatch")
        state = jnp.concatenate((batch["proprioception"], age), axis=-1)
        state = nn.Dense(cfg.embed_dim, name="state_in")(state)
        state = nn.Dense(cfg.embed_dim, name="state_out")(nn.gelu(state)) + modality[2]
        other = jnp.concatenate((tactile, state[:, None]), axis=1)
        other_valid = jnp.concatenate(
            (frame_valid.reshape((state.shape[0], -1)), jnp.ones((state.shape[0], 1), bool)), axis=1
        )
        return Fusion(cfg, name="fusion")(visual, other, visual_valid, other_valid, train=train)[0]


class TactileDecoder(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, z, batch: Batch, *, train=False):
        cfg = self.config
        valid, _ = tactile_masks(batch)
        b, length, sensors = valid.shape
        sensor = self.param("sensor_query", nn.initializers.normal(0.02), (cfg.num_sensors, cfg.embed_dim))
        slot = self.param("history_query", nn.initializers.normal(0.02), (cfg.history_frames, cfg.embed_dim))
        time = nn.Dense(cfg.embed_dim, name="time_query")(relative_times(batch)[..., None])
        q = sensor[None, None] + slot[None, -length:, None] + time
        q = q.reshape((b, length * sensors, cfg.embed_dim))
        query_valid = valid.reshape((b, -1))
        q = jnp.where(query_valid[..., None], q, 0.0)
        for stage in range(cfg.decoder_layers):
            y = nn.LayerNorm(name=f"self_norm_{stage}")(q)
            mask = query_valid[:, :, None] & query_valid[:, None, :]
            q = q + SafeAttention(cfg.embed_dim, cfg.num_heads, cfg.dropout_rate, name=f"self_attention_{stage}")(
                y, y, mask, train=train
            )
            y = nn.LayerNorm(name=f"cross_norm_{stage}")(q)
            q = q + SafeAttention(cfg.embed_dim, cfg.num_heads, cfg.dropout_rate, name=f"cross_attention_{stage}")(
                y, z, query_valid[..., None], train=train
            )
            y = nn.LayerNorm(name=f"ffn_norm_{stage}")(q)
            y = nn.Dense(int(cfg.embed_dim * cfg.mlp_ratio), name=f"ffn_in_{stage}")(y)
            y = nn.gelu(y)
            y = nn.Dense(cfg.embed_dim, name=f"ffn_out_{stage}")(y)
            q = jnp.where(query_valid[..., None], q + y, 0.0)
        pixels = cfg.tactile_resolution**2 * cfg.tactile_channels
        output = nn.Dense(pixels, name="output")(q)
        return output.reshape(
            (b, length, sensors, cfg.tactile_resolution, cfg.tactile_resolution, cfg.tactile_channels)
        )


def visual_reconstruction_loss(prediction, target, valid):
    target = jax.lax.stop_gradient(target)
    error = jnp.where(valid[..., None], prediction - jnp.where(valid[..., None], target, 0.0), 0.0)
    return jnp.sum(error**2) / jnp.maximum(jnp.sum(valid) * target.shape[-1], 1)


def tactile_reconstruction_loss(prediction, batch: Batch):
    """Pixel mean per sensor/frame, then mean across valid sensor/frames."""
    _, pixel = tactile_masks(batch)
    target = jax.lax.stop_gradient(jnp.where(pixel, batch["tactile"], 0.0))
    error = jnp.where(pixel, prediction - target, 0.0) ** 2
    count = jnp.sum(pixel, axis=(-3, -2, -1))
    frame_valid = count > 0
    mse = jnp.sum(error, axis=(-3, -2, -1)) / jnp.maximum(count, 1)
    loss = jnp.sum(mse) / jnp.maximum(jnp.sum(frame_valid), 1)
    per_sensor = jnp.sum(mse, axis=(0, 1)) / jnp.maximum(jnp.sum(frame_valid, axis=(0, 1)), 1)
    per_history = jnp.sum(mse, axis=(0, 2)) / jnp.maximum(jnp.sum(frame_valid, axis=(0, 2)), 1)
    per_sample = jnp.sum(mse, axis=(1, 2)) / jnp.maximum(jnp.sum(frame_valid, axis=(1, 2)), 1)
    return loss, {
        "tactile_loss": loss,
        "tactile_count": jnp.sum(frame_valid),
        "tactile_total_count": jnp.asarray(frame_valid.size),
        "tactile_sensor_count": jnp.sum(frame_valid, axis=(0, 1)),
        "tactile_history_count": jnp.sum(frame_valid, axis=(0, 2)),
        "tactile_latest_count": jnp.sum(frame_valid[:, -1]),
        "tactile_past_count": jnp.sum(frame_valid[:, :-1]),
        "tactile_per_sensor": per_sensor,
        "tactile_per_history": per_history,
        "tactile_per_sample": per_sample,
        "tactile_latest": per_history[-1],
        "tactile_history": jnp.sum(mse[:, :-1]) / jnp.maximum(jnp.sum(frame_valid[:, :-1]), 1),
        "tactile_valid_fraction": jnp.mean(frame_valid),
    }


def _rlt_config(config: ModelConfig, layers: int):
    return RLTokenConfig(
        num_rl_tokens=1,
        num_layers=layers,
        embed_dim=config.embed_dim,
        input_dim=config.input_dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout_rate=config.dropout_rate,
    )


class RepresentationModel(nn.Module):
    config: ModelConfig

    def setup(self):
        cfg = self.config
        if cfg.mode == "rlt_original":
            self.encoder = RLTokenEncoder(_rlt_config(cfg, cfg.rlt_encoder_layers))
        else:
            self.encoder = TRLTEncoder(cfg)
            self.tactile_decoder = TactileDecoder(cfg)
        self.decoder = RLTokenDecoder(_rlt_config(cfg, cfg.decoder_layers))

    def encode(self, batch: Batch, *, train=False):
        if self.config.mode == "rlt_original":
            valid = batch["visual_valid_mask"].astype(bool)
            visual = jnp.where(valid[..., None], batch["visual_features"], 0.0)
            return self.encoder(visual, valid, train=train)
        return self.encoder(batch, train=train)

    def reconstruct(self, batch: Batch, *, train=False):
        z = self.encode(batch, train=train)
        output = {
            "z": z,
            "visual": self.decoder(
                z, batch["visual_features"].shape[1], target_valid_mask=batch["visual_valid_mask"], train=train
            ),
        }
        if self.config.mode != "rlt_original":
            output["tactile"] = self.tactile_decoder(z, batch, train=train)
        return output

    def compute_trlt_loss(self, batch: Batch, *, train=True):
        prediction = self.reconstruct(batch, train=train)
        visual = visual_reconstruction_loss(prediction["visual"], batch["visual_features"], batch["visual_valid_mask"])
        info = {
            "visual_loss": visual,
            "visual_count": jnp.sum(batch["visual_valid_mask"]) * batch["visual_features"].shape[-1],
            "visual_total_count": jnp.asarray(batch["visual_features"].size),
            "visual_weighted": self.config.lambda_visual * visual,
            "visual_valid_fraction": jnp.mean(batch["visual_valid_mask"]),
        }
        if self.config.mode == "rlt_original":
            # Baseline objective is exactly the original unweighted visual MSE.
            info.update(total_loss=visual, mse=visual)
            return visual, info
        tactile, tactile_info = tactile_reconstruction_loss(prediction["tactile"], batch)
        info.update(tactile_info)
        info["tactile_weighted"] = self.config.lambda_tactile * tactile
        total = self.config.lambda_visual * visual + self.config.lambda_tactile * tactile
        info["total_loss"] = total
        return total, info

    def __call__(self, batch: Batch, *, train=True):
        return self.compute_trlt_loss(batch, train=train)


def build_model(config: ModelConfig) -> RepresentationModel:
    return RepresentationModel(config)


def encode_trlt(model: RepresentationModel, variables, batch: Batch):
    """Stateless serving helper; rolling contexts are owned by the caller."""
    return model.apply(variables, batch, method=model.encode, train=False)
