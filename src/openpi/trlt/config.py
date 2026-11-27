"""Shared contracts. Timestamps are episode-local seconds; True masks mean valid."""

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]
CAMERAS = tuple(f"observation.images.{name}" for name in ("head_left", "left_wrist", "right_wrist"))
CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
SENSORS = tuple(
    f"observation.images.tactile_{side}_deform_{finger}"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
)
MODELS = ("rlt_original", "trlt_self_only", "trlt_cross_only", "trlt_alternating")
PROBE_ARCHITECTURES = ("mlp", "resmlp", "part_attention", "touch_part_attention")
PROBE_TOUCH_KV_MODES = ("layernorm", "raw_mlp", "raw_gru")
PROBE_ATTENTION_MASK_MODES = ("full", "kinematic", "hierarchical")
PROBE_LR_SCHEDULES = ("constant", "cosine")
SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def root_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class TemporalConfig:
    fps: float = 30.0
    visual_refresh_frames: int = 15
    query_stride_frames: int = 3
    tactile_history_frames: int = 10
    tactile_history_stride_frames: int = 1
    action_horizon_frames: int = 6
    max_observation_lag_seconds: float = 1.5 / 30

    def __post_init__(self):
        if self.fps <= 0 or self.max_observation_lag_seconds < 0:
            raise ValueError("Invalid fps or allowed observation lag")
        if (
            min(
                self.visual_refresh_frames,
                self.query_stride_frames,
                self.tactile_history_frames,
                self.tactile_history_stride_frames,
                self.action_horizon_frames,
            )
            <= 0
        ):
            raise ValueError("Temporal frame counts must be positive")


@dataclass(frozen=True)
class ModelConfig:
    mode: Literal["rlt_original", "trlt_self_only", "trlt_cross_only", "trlt_alternating"] = "trlt_alternating"
    input_dim: int = 2048
    embed_dim: int = 512
    num_heads: int = 8
    num_stages: int = 4
    rlt_encoder_layers: int = 2
    decoder_layers: int = 2
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    use_modality_embedding: bool = True
    use_sensor_id_embedding: bool = True
    num_sensors: int = 10
    tactile_format: Literal["image", "taxels"] = "image"
    num_taxels: int = 15
    tactile_channels: int = 1
    tactile_resolution: int = 64
    history_frames: int = 10
    state_dim: int = 58
    num_cameras: int = 3
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    lambda_visual: float = 1.0
    lambda_tactile: float = 1.0
    geometry_available: bool = False  # Legacy flag; enabled modes require this to be True.
    geometry_mode: Literal["none", "mlp", "rope3d"] = "none"
    geometry_spec: str = "configs/trlt/geometry/sharpa_wave_v1.json"
    geometry_scale_m: float = 0.01
    geometry_mlp_hidden: int = 128
    geometry_rope_axis_dim: int = 16
    geometry_rope_base: float = 10000.0

    def __post_init__(self):
        object.__setattr__(self, "cnn_channels", tuple(self.cnn_channels))
        if (
            min(
                self.input_dim,
                self.embed_dim,
                self.num_heads,
                self.num_stages,
                self.rlt_encoder_layers,
                self.decoder_layers,
                self.num_sensors,
                self.tactile_channels,
                self.tactile_resolution,
                self.history_frames,
                self.state_dim,
                self.num_cameras,
                *self.cnn_channels,
            )
            <= 0
        ):
            raise ValueError("Model dimensions and layer counts must be positive")
        if self.tactile_format not in ("image", "taxels") or self.num_taxels <= 0:
            raise ValueError("Invalid tactile format/taxel count")
        if self.tactile_format == "taxels" and (self.tactile_channels != 3 or self.geometry_mode != "none"):
            raise ValueError("Native taxels require three axes and disabled geometry")
        if not isinstance(self.use_modality_embedding, bool):
            raise ValueError("use_modality_embedding must be a boolean")
        if not isinstance(self.use_sensor_id_embedding, bool):
            raise ValueError("use_sensor_id_embedding must be a boolean")
        if self.mode == "rlt_original" and not self.use_sensor_id_embedding:
            raise ValueError("Sensor ID ablation applies to TRLT encoder tactile tokens only")
        if min(self.lambda_visual, self.lambda_tactile) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.mode not in MODELS or self.embed_dim % self.num_heads:
            raise ValueError("Invalid mode or attention dimensions")
        if self.geometry_mode not in ("none", "mlp", "rope3d"):
            raise ValueError("Unknown geometry_mode")
        if self.geometry_available != (self.geometry_mode != "none"):
            raise ValueError("geometry_available must match geometry_mode != none")
        if self.geometry_mode != "none":
            if self.mode == "rlt_original" or self.num_sensors != 10 or self.num_cameras != 3:
                raise ValueError("Geometry requires TRLT with canonical ten sensors and three cameras")
            if not math.isfinite(self.geometry_scale_m) or self.geometry_scale_m <= 0 or self.geometry_mlp_hidden <= 0:
                raise ValueError("Invalid geometry scale/MLP width")
            if self.geometry_mode == "rope3d" and (
                self.geometry_rope_axis_dim < 2
                or self.geometry_rope_axis_dim % 2
                or 3 * self.geometry_rope_axis_dim > self.embed_dim // self.num_heads
                or not math.isfinite(self.geometry_rope_base)
                or self.geometry_rope_base <= 1
            ):
                raise ValueError("3D RoPE needs three even axis blocks fitting in each attention head")


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    microbatch_size: int = 4
    max_epochs: int | None = 100
    patience: int | None = 10
    gradient_clip: float = 1.0
    max_steps: int | None = None


@dataclass(frozen=True)
class ProbeConfig:
    hidden_dim: int = 256
    depth: int = 2
    action_dim: int = 58
    horizon: int = 6
    dropout_rate: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    architecture: Literal["mlp", "resmlp", "part_attention"] = "mlp"
    num_heads: int = 4
    mlp_ratio: float = 4.0
    attention_mask_mode: Literal["full", "kinematic", "hierarchical"] = "full"
    lr_schedule: Literal["constant", "cosine"] = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    variant_tag: str | None = None
    # touch_part_attention only: per-finger cross-attention over frozen tactile tokens.
    touch_num_heads: int = 4
    touch_use_pos_xyz: bool = True
    touch_kv_mode: Literal["layernorm", "raw_mlp", "raw_gru"] = "layernorm"
    # Optional raw-token encoder settings (omitted from identity when None, so H1-H3 stay unchanged).
    touch_raw_hidden_dim: int | None = None  # raw_mlp first-layer width (None = hidden_dim)
    touch_gru_hidden_dim: int | None = None  # raw_gru hidden width (required for raw_gru)
    touch_raw_frames: int | None = None  # raw_*: required history frames per token (C = 3 * frames)

    def __post_init__(self):
        if min(self.hidden_dim, self.depth, self.action_dim, self.horizon, self.batch_size) <= 0:
            raise ValueError("Probe dimensions, horizon and batch size must be positive")
        if self.architecture not in PROBE_ARCHITECTURES:
            raise ValueError(f"Unknown probe architecture: {self.architecture}")
        if self.attention_mask_mode not in PROBE_ATTENTION_MASK_MODES:
            raise ValueError(f"Unknown attention mask mode: {self.attention_mask_mode}")
        if self.lr_schedule not in PROBE_LR_SCHEDULES:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        if not math.isfinite(self.dropout_rate) or not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("dropout_rate must be finite and in [0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if not math.isfinite(self.min_lr_ratio) or not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be finite and in [0, 1]")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        if self.architecture in ("part_attention", "touch_part_attention"):
            if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
                raise ValueError("part_attention requires positive num_heads dividing hidden_dim")
            if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
                raise ValueError("part_attention requires positive finite mlp_ratio")
            if self.action_dim not in (58, 22):
                raise ValueError("part_attention supports 58D T-Rex and 22D OMY+LEAP action dimensions")
        if self.architecture == "touch_part_attention":
            if self.touch_num_heads <= 0 or self.hidden_dim % self.touch_num_heads != 0:
                raise ValueError("touch_num_heads must be positive and divide hidden_dim")
            if self.touch_kv_mode not in PROBE_TOUCH_KV_MODES:
                raise ValueError(f"Unknown touch_kv_mode: {self.touch_kv_mode}")
            if not isinstance(self.touch_use_pos_xyz, bool):
                raise ValueError("touch_use_pos_xyz must be boolean")
            if (self.touch_gru_hidden_dim is not None) != (self.touch_kv_mode == "raw_gru"):
                raise ValueError("touch_gru_hidden_dim is required for, and only valid with, touch_kv_mode=raw_gru")
            if self.touch_raw_hidden_dim is not None and self.touch_kv_mode != "raw_mlp":
                raise ValueError("touch_raw_hidden_dim is only valid with touch_kv_mode=raw_mlp")
            if self.touch_raw_frames is not None and self.touch_kv_mode not in ("raw_mlp", "raw_gru"):
                raise ValueError("touch_raw_frames is only valid with raw touch_kv_modes")
            for name in ("touch_raw_hidden_dim", "touch_gru_hidden_dim", "touch_raw_frames"):
                value = getattr(self, name)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                    raise ValueError(f"{name} must be a positive integer")
        elif any(
            getattr(self, name) is not None
            for name in ("touch_raw_hidden_dim", "touch_gru_hidden_dim", "touch_raw_frames")
        ):
            raise ValueError("touch_* raw encoder settings require architecture=touch_part_attention")
        if self.variant_tag is not None and (
            not isinstance(self.variant_tag, str)
            or "__" in self.variant_tag
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?", self.variant_tag)
        ):
            raise ValueError("variant_tag must be a lowercase alphanumeric/underscore/hyphen identifier without '__'")


# Tensor batch contract consumed by model and runner:
# visual_features float32 [B,Nv,Dv]; visual_valid_mask bool [B,Nv]
# tactile float32 [B,L,Ns,H,W,C]; tactile_valid_mask bool [B,L,Ns]
# spatial_valid_mask optional bool broadcastable to tactile shape
# tactile_timestamps float32 [B,L,Ns] (canonical), or [B,L] accepted by model
# tactile_positions optional float32 [B,L,Ns,3] hand-local meters; required for enabled geometry
# geometry_valid_mask optional bool [B,L,Ns]; accompanies tactile_positions
# proprioception float32 [B,58] normalized; query_timestamp / visual_timestamp [B]
# action_target float32 [B,Ha,58] normalized; action_valid_mask bool [B,Ha]
# episode_id / anchor_frame_index / query_frame_index int arrays [B]
# object and caption are host metadata, excluded from JIT inputs.


def config_dict(config):
    return asdict(config)


def model_config_dict(config):
    """Keep disabled-geometry checkpoint identity compatible with pre-geometry runs."""
    value = asdict(config)
    if config.tactile_format == "image":
        value.pop("tactile_format")
        value.pop("num_taxels")
    if config.use_modality_embedding:
        value.pop("use_modality_embedding")
    if config.use_sensor_id_embedding:
        value.pop("use_sensor_id_embedding")
    if config.geometry_mode == "none":
        for key in (
            "geometry_mode",
            "geometry_spec",
            "geometry_scale_m",
            "geometry_mlp_hidden",
            "geometry_rope_axis_dim",
            "geometry_rope_base",
        ):
            value.pop(key)
    return value


def probe_config_dict(config: ProbeConfig) -> dict:
    """Keep default MLP probe checkpoint identity and configs compatible with legacy runs."""
    value = asdict(config)
    if config.architecture == "mlp":
        for key in ("architecture", "num_heads", "mlp_ratio", "attention_mask_mode"):
            value.pop(key)
    elif config.architecture == "resmlp":
        for key in ("num_heads", "mlp_ratio", "attention_mask_mode"):
            value.pop(key)
    if config.lr_schedule == "constant":
        value.pop("lr_schedule")
        value.pop("min_lr_ratio")
        if config.warmup_steps == 0:
            value.pop("warmup_steps")
    if config.gradient_clip == 1.0:
        value.pop("gradient_clip")
    if config.variant_tag is None:
        value.pop("variant_tag")
    if config.architecture != "touch_part_attention":
        for key in ("touch_num_heads", "touch_use_pos_xyz", "touch_kv_mode"):
            value.pop(key)
    for key in ("touch_raw_hidden_dim", "touch_gru_hidden_dim", "touch_raw_frames"):
        if value[key] is None:
            value.pop(key)
    return value


def probe_variant_name(config: ProbeConfig) -> str:
    """Return canonical directory/model suffix identifying the Stage B probe variant."""
    base = (
        f"{config.architecture}_{config.attention_mask_mode}"
        if config.architecture in ("part_attention", "touch_part_attention")
        else config.architecture
    )
    return f"{base}_{config.variant_tag}" if config.variant_tag else base
