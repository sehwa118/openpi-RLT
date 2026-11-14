"""Shared contracts. Timestamps are episode-local seconds; True masks mean valid."""

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
    num_sensors: int = 10
    tactile_channels: int = 1
    tactile_resolution: int = 64
    history_frames: int = 10
    state_dim: int = 58
    num_cameras: int = 3
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    lambda_visual: float = 1.0
    lambda_tactile: float = 1.0
    geometry_available: bool = False

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
        if min(self.lambda_visual, self.lambda_tactile) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.mode not in MODELS or self.embed_dim % self.num_heads:
            raise ValueError("Invalid mode or attention dimensions")
        if self.geometry_available:
            raise ValueError("T-Rex geometry is unavailable; pose contribution must remain zero")


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


# Tensor batch contract consumed by model and runner:
# visual_features float32 [B,Nv,Dv]; visual_valid_mask bool [B,Nv]
# tactile float32 [B,L,Ns,H,W,C]; tactile_valid_mask bool [B,L,Ns]
# spatial_valid_mask optional bool broadcastable to tactile shape
# tactile_timestamps float32 [B,L,Ns] (canonical), or [B,L] accepted by model
# proprioception float32 [B,58] normalized; query_timestamp / visual_timestamp [B]
# action_target float32 [B,Ha,58] normalized; action_valid_mask bool [B,Ha]
# episode_id / anchor_frame_index / query_frame_index int arrays [B]
# object and caption are host metadata, excluded from JIT inputs.


def config_dict(config):
    return asdict(config)
