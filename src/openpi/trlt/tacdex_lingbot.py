"""3DTacDex PaXini -> LingBot-Touch magnetic input adapter (NumPy only, no accelerator).

Contract (see docs/trlt/lingbot_3dtacdex.md):
- Source: OXT/FTP-1 3DTacDex Zarr, `data/right_tactile_data_fingers` [T, 8, 15, 3], right LEAP hand.
  Channels alternate tip/pulp for thumb, index, middle, ring (OXT areas [0,6,1,7,2,8,3,9]).
- Target: LingBot-Touch `magnetic` route, signal [.., 5, M, 3] in fixed finger order
  (thumb, index, middle, ring, pinky). LEAP has no pinky: slot 4 is zero with coverage False.
- Signal: X_norm = clip(sx*X/s_x, -1, 1), Y_norm likewise, Z_norm = clip(sz*Z/s_z, 0, 1).
  PaXini axis 2 is the (strictly positive) normal channel; axes 0/1 are shear.
- Geometry: PaXini sensor-frame taxel centers (mm, from the 3DTacDex source repo) mapped by a fixed
  isotropic transform p_canonical = (p_mm - origin_mm) / radius_mm. The default transform is an
  approximate template alignment to LingBot's magnetic fingertip reference, not a measured calibration.
"""

from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openpi.trlt.config import digest

LINGBOT_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
TACDEX_PAD_NAMES = (
    "thumb_tip",
    "thumb_pulp",
    "index_tip",
    "index_pulp",
    "middle_tip",
    "middle_pulp",
    "ring_tip",
    "ring_pulp",
)
TACDEX_TACTILE_AREAS = (0, 6, 1, 7, 2, 8, 3, 9)
TACDEX_TAXELS = 15
TACDEX_FINGERS = ("thumb", "index", "middle", "ring")
GEOMETRY_SOURCE = (
    "https://github.com/tianhaowuhz/3dtacdex/blob/a4a367514ee25c4ad43eb2400cac0141bcb85331/"
    "diffusion_policy/real_world/fk/tactile.py"
)

# PaXini sensor-frame taxel centers in mm (x lateral, y along the finger toward the tip, z outward normal).
# Order is column-major: index k = 5 * column + row, columns x = (-, 0, +), rows proximal -> distal.
PAXINI_TIP_TAXELS_MM = np.array(
    [
        [-4.50000, 4.30109, 2.97814],
        [-4.50000, 8.15109, 2.89349],
        [-4.50000, 12.18660, 2.64440],
        [-4.50000, 15.99390, 2.06277],
        [-2.25000, 21.45300, 0.09510],
        [0.00000, 4.30109, 3.10726],
        [0.00000, 8.15109, 3.03111],
        [0.00000, 12.20620, 2.80133],
        [0.00000, 16.01800, 2.25633],
        [0.00000, 24.50520, -2.49584],
        [4.50000, 4.30109, 2.97814],
        [4.50000, 8.15109, 2.89349],
        [4.50000, 12.18660, 2.64440],
        [4.50000, 15.99390, 2.06277],
        [2.25000, 21.45300, 0.09510],
    ],
    dtype=np.float64,
)
PAXINI_PULP_TAXELS_MM = np.array(
    [[x, y, 2.94543 if x else 3.09994] for x in (-4.7, 0.0, 4.7) for y in (3.3, 7.8, 12.3, 16.8, 21.3)],
    dtype=np.float64,
)

PAD_MODES = ("tip", "tip_pulp")


@dataclass(frozen=True)
class PaxiniLingBotConfig:
    """Fixed PaXini -> LingBot magnetic preparation settings (recorded with every extraction)."""

    pads: str = "tip"
    frames: int = 2
    scales_xyz: tuple[float, float, float] = (30.0, 30.0, 30.0)
    axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    radius_mm: float = 14.0
    origin_mm: tuple[float, float, float] = (0.0, 13.25, 3.0)
    pulp_gap_mm: float = 10.0
    modality: str = "magnetic"

    def __post_init__(self):
        if self.pads not in PAD_MODES:
            raise ValueError(f"pads must be one of {PAD_MODES}")
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames <= 0:
            raise ValueError("frames must be a positive integer")
        scales = tuple(float(s) for s in self.scales_xyz)
        signs = tuple(float(s) for s in self.axis_signs)
        origin = tuple(float(s) for s in self.origin_mm)
        if len(scales) != 3 or not all(np.isfinite(s) and s > 0 for s in scales):
            raise ValueError("scales_xyz must be three positive finite values")
        if len(signs) != 3 or any(s not in (-1.0, 1.0) for s in signs):
            raise ValueError("axis_signs must be three values in {-1, +1}")
        if len(origin) != 3 or not all(np.isfinite(origin)):
            raise ValueError("origin_mm must be three finite values")
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError("radius_mm must be positive and finite")
        if not np.isfinite(self.pulp_gap_mm) or self.pulp_gap_mm < 0:
            raise ValueError("pulp_gap_mm must be nonnegative and finite")
        if self.modality != "magnetic":
            raise ValueError("PaXini tri-axial taxels use the LingBot magnetic route only")
        object.__setattr__(self, "scales_xyz", scales)
        object.__setattr__(self, "axis_signs", signs)
        object.__setattr__(self, "origin_mm", origin)
        object.__setattr__(self, "radius_mm", float(self.radius_mm))
        object.__setattr__(self, "pulp_gap_mm", float(self.pulp_gap_mm))

    @property
    def taxels_per_finger(self) -> int:
        return TACDEX_TAXELS * (2 if self.pads == "tip_pulp" else 1)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            {
                "scales_xyz": list(self.scales_xyz),
                "axis_signs": list(self.axis_signs),
                "origin_mm": list(self.origin_mm),
                "geometry_source": GEOMETRY_SOURCE,
                "geometry_calibrated": False,
            }
        )
        return out

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "PaxiniLingBotConfig":
        keys = {"pads", "frames", "scales_xyz", "axis_signs", "radius_mm", "origin_mm", "pulp_gap_mm", "modality"}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mapping.items() if k in keys})

    @property
    def identity(self) -> str:
        return digest(self.to_dict())


def pad_channels(config: PaxiniLingBotConfig) -> np.ndarray:
    """Source channel index for every [finger(4), slot(M)] position, tip taxels first."""
    tips = np.array([2 * f for f in range(4)])
    if config.pads == "tip":
        return tips[:, None]
    return np.stack((tips, tips + 1), axis=1)


def canonical_geometry(config: PaxiniLingBotConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return LingBot canonical pos_xyz [5, M, 3] float32 and coverage [5, M] bool (pinky slot empty)."""
    origin = np.asarray(config.origin_mm, dtype=np.float64)
    tip = (PAXINI_TIP_TAXELS_MM - origin) / config.radius_mm
    parts = [tip]
    if config.pads == "tip_pulp":
        # Straight-finger approximation: pulp's distal row sits `pulp_gap_mm` proximal of the tip's
        # proximal row. The pulp is on the middle phalanx and is outside LingBot's fingertip pretraining.
        shift = PAXINI_TIP_TAXELS_MM[:, 1].min() - config.pulp_gap_mm - PAXINI_PULP_TAXELS_MM[:, 1].max()
        pulp_mm = PAXINI_PULP_TAXELS_MM + np.array([0.0, shift, 0.0])
        parts.append((pulp_mm - origin) / config.radius_mm)
    per_finger = np.concatenate(parts, axis=0)
    m = per_finger.shape[0]
    pos = np.zeros((5, m, 3), dtype=np.float32)
    pos[:4] = per_finger.astype(np.float32)
    coverage = np.zeros((5, m), dtype=bool)
    coverage[:4] = True
    return pos, coverage


@dataclass(frozen=True)
class SignalStats:
    """Clipping diagnostics over valid slots (guide: monitor excessive clipping)."""

    elements: int
    frac_xy_clipped: float
    frac_z_clipped_high: float
    frac_z_negative_clipped: float
    frac_exact_zero: float

    def to_dict(self):
        return asdict(self)


def prepare_signal(raw: np.ndarray, config: PaxiniLingBotConfig) -> tuple[np.ndarray, SignalStats]:
    """Map raw PaXini [..., 8, 15, 3] to normalized LingBot magnetic signal [..., 5, M, 3] float32."""
    values = np.asarray(raw)
    if values.ndim < 3 or values.shape[-3:] != (8, TACDEX_TAXELS, 3):
        raise ValueError("raw tactile must have shape [..., 8, 15, 3]")
    if np.iscomplexobj(values) or values.dtype == bool:
        raise ValueError("raw tactile must be real numeric")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("raw tactile contains nonfinite values")
    channels = pad_channels(config)  # [4, P]
    selected = values[..., channels, :, :]  # [..., 4, P, 15, 3]
    selected = selected.reshape(*values.shape[:-3], 4, config.taxels_per_finger, 3)
    scaled = selected * np.asarray(config.axis_signs) / np.asarray(config.scales_xyz)
    xy, z = scaled[..., :2], scaled[..., 2]
    stats = SignalStats(
        elements=int(selected.size),
        frac_xy_clipped=float(np.mean(np.abs(xy) > 1.0)),
        frac_z_clipped_high=float(np.mean(z > 1.0)),
        frac_z_negative_clipped=float(np.mean(z < 0.0)),
        frac_exact_zero=float(np.mean(selected == 0.0)),
    )
    signal = np.zeros((*values.shape[:-3], 5, config.taxels_per_finger, 3), dtype=np.float32)
    signal[..., :4, :, :2] = np.clip(xy, -1.0, 1.0)
    signal[..., :4, :, 2] = np.clip(z, 0.0, 1.0)
    return signal, stats


def causal_windows(
    episode_ends: Sequence[int], frames: int, *, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal per-anchor frame windows [t-frames+1 .. t], never crossing segment starts.

    Early anchors replicate the segment's first frame; `padded` counts replicated frames.
    Returns (anchor [N], window [N, frames], padded [N], segment [N]).
    """
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0 or np.any(np.diff(ends) <= 0) or ends[0] <= 0:
        raise ValueError("episode_ends must be strictly increasing positive exclusive ends")
    if frames <= 0 or stride <= 0:
        raise ValueError("frames and stride must be positive")
    starts = np.concatenate(([0], ends[:-1]))
    anchors, windows, padded, segments = [], [], [], []
    for seg, (start, end) in enumerate(zip(starts, ends, strict=True)):
        t = np.arange(start, end, stride, dtype=np.int64)
        offsets = np.arange(-frames + 1, 1, dtype=np.int64)
        raw_idx = t[:, None] + offsets[None, :]
        anchors.append(t)
        windows.append(np.maximum(raw_idx, start))
        padded.append(np.sum(raw_idx < start, axis=1))
        segments.append(np.full(t.shape, seg, dtype=np.int64))
    return (np.concatenate(anchors), np.concatenate(windows), np.concatenate(padded), np.concatenate(segments))


@dataclass
class TacdexTask:
    """Arrays read from one OXT/FTP-1 3DTacDex task archive (images are not loaded)."""

    path: str
    group: str
    tactile: np.ndarray
    hand_joints: np.ndarray
    arm_joints: np.ndarray
    timestamps: np.ndarray
    episode_ends: np.ndarray
    trajectory_id: np.ndarray
    hand_command: np.ndarray | None
    hand_command_present: np.ndarray | None

    @property
    def frames(self) -> int:
        return int(self.tactile.shape[0])

    def segment_ids(self) -> np.ndarray:
        return np.searchsorted(self.episode_ends, np.arange(self.frames), side="right")


def _open_zip_group(path: Path, group: str | None):
    import zarr

    store_cls = getattr(zarr.storage, "ZipStore", None) or zarr.ZipStore
    store = store_cls(str(path), mode="r")
    if group is None:
        names = sorted({n.split("/")[0] for n in _zip_names(path) if n.split("/")[0].endswith(".zarr")})
        if len(names) != 1:
            store.close()
            raise ValueError(f"Cannot infer Zarr group in {path}; pass group explicitly (found {names})")
        group = names[0]
    return store, zarr.open_group(store=store, path=group, mode="r"), group


def _zip_names(path: Path) -> list[str]:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return [n for n in archive.namelist() if n.endswith((".zgroup", ".zarray"))]


def _strings(root, key: str) -> np.ndarray | None:
    try:
        return np.asarray(root[key][:]).astype(str)
    except Exception:
        return None


def load_oxt_task(path: str | Path, group: str | None = None) -> TacdexTask:
    """Read tactile, joints, segment ends and trajectory ids from a 3DTacDex `<task>.zarr.zip`."""
    path = Path(path)
    store, root, group = _open_zip_group(path, group)
    try:
        tactile = np.asarray(root["data/right_tactile_data_fingers"][:])
        areas = np.asarray(root["data/right_tactile_area_fingers"][:])
        if tactile.ndim != 4 or tactile.shape[1:] != (8, TACDEX_TAXELS, 3):
            raise ValueError(f"Unexpected tactile shape {tactile.shape}")
        if not np.all(areas == np.asarray(TACDEX_TACTILE_AREAS)):
            raise ValueError("Tactile channel areas differ from the expected OXT [0,6,1,7,2,8,3,9] layout")
        sensors = _strings(root, "data/right_tactile_sensor_fingers")
        if sensors is not None and not np.all(sensors == "PaXini"):
            raise ValueError("Tactile sensor is not PaXini for every frame")
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        if ends[-1] != tactile.shape[0]:
            raise ValueError("episode_ends does not cover all frames")
        traj = _strings(root, "data/source_episode_id")
        if traj is None:
            traj = np.searchsorted(ends, np.arange(tactile.shape[0]), side="right").astype(str)
        keys = set(root["raw"].array_keys()) if "raw" in root else set()
        cmd = present = None
        if {"hand_commanded_joint_position", "hand_command_present"} <= keys:
            cmd = np.asarray(root["raw/hand_commanded_joint_position"][:], dtype=np.float64)
            present = np.asarray(root["raw/hand_command_present"][:], dtype=bool)
        return TacdexTask(
            path=str(path),
            group=group,
            tactile=tactile,
            hand_joints=np.asarray(root["data/right_hand_joints"][:], dtype=np.float64),
            arm_joints=np.asarray(root["data/right_arm_joints"][:], dtype=np.float64),
            timestamps=np.asarray(root["data/timestamps"][:], dtype=np.float64),
            episode_ends=ends,
            trajectory_id=traj,
            hand_command=cmd,
            hand_command_present=present,
        )
    finally:
        store.close()


def grouped_split(groups: np.ndarray, fractions=(0.6, 0.2, 0.2), seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic group split (by original trajectory) into boolean train/validation/test masks."""
    unique = np.array(sorted(set(groups.tolist())))
    if unique.size < 3:
        raise ValueError("Need at least three groups for a train/validation/test split")
    order = np.random.default_rng(seed).permutation(unique.size)
    n_train = max(1, round(fractions[0] * unique.size))
    n_val = max(1, round(fractions[1] * unique.size))
    n_train = min(n_train, unique.size - 2)
    names = {
        "train": unique[order[:n_train]],
        "validation": unique[order[n_train : n_train + n_val]],
        "test": unique[order[n_train + n_val :]],
    }
    return {k: np.isin(groups, v) for k, v in names.items()}
