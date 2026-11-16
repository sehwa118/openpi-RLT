"""CPU hand-local FK and immutable episode caches; no simulator or GPU dependency."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from openpi.trlt.config import SENSORS
from openpi.trlt.config import digest
from openpi.trlt.config import root_path

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
SOURCE_REPO = "ZhuoyangLiu2005/T-Rex"
SOURCE_REVISION = "ba4047ce3e9f3d2013d60de600e136897e90ccb8"
DEFAULT_SPEC = "configs/trlt/geometry/sharpa_wave_v1.json"


def quaternion_matrix(wxyz):
    q = np.asarray(wxyz, np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vector(node, name, default):
    return [float(x) for x in node.get(name, default).split()]


def derive_spec(source_dir):
    """Extract hinge chains and elastomer AABB centers from pinned MJCF/STL."""
    root = root_path(source_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["side"] / Path(record["path"]).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
    sensors = []
    for side in ("left", "right"):
        xml = ET.parse(root / side / f"{side}_sharpa_wave_with_wrist.xml").getroot()
        if xml.find("compiler").get("angle") != "radian":
            raise ValueError("Expected radian MJCF")
        meshes = {m.get("name"): m for m in xml.findall("./asset/mesh")}
        chains = {}

        def visit(body, chain, chains=chains):
            joints = body.findall("joint")
            if len(joints) > 1:
                raise ValueError("Expected one hinge per body")
            step = {
                "body": body.get("name"),
                "pos": _vector(body, "pos", "0 0 0"),
                "quat_wxyz": _vector(body, "quat", "1 0 0 0"),
            }
            if joints:
                j = joints[0]
                if j.get("type", "hinge") != "hinge" or _vector(j, "pos", "0 0 0") != [0, 0, 0]:
                    raise ValueError("Only origin-centered hinges supported")
                step.update(joint=j.get("name"), axis=_vector(j, "axis", "0 0 1"))
            current = [*chain, step]
            chains[body.get("name")] = (current, body)
            for child in body.findall("body"):
                visit(child, current)

        hand = xml.find("./worldbody/body")
        # The hand root is the fixed local wrist reference, independently per hand.
        for child in hand.findall("body"):
            visit(child, [])
        names = {step["joint"] for chain, _ in chains.values() for step in chain if "joint" in step}
        if names != {f"{side}_{j}" for j in JOINTS}:
            raise ValueError("Unexpected hand joint mapping")
        for finger in FINGERS:
            chain, body = chains[f"{side}_{finger}_DP"]
            geom = next(g for g in body.findall("geom") if g.get("name") == f"{side}_{finger}_elastomer")
            mesh = meshes[geom.get("mesh")]
            raw = (root / side / mesh.get("file")).read_bytes()
            count = int.from_bytes(raw[80:84], "little")
            if len(raw) != 84 + 50 * count:
                raise ValueError("Expected binary STL")
            triangles = np.frombuffer(
                raw,
                dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]),
                offset=84,
            )
            vertices = triangles["vertices"].reshape(-1, 3).astype(np.float64)
            vertices *= np.array(_vector(mesh, "scale", "1 1 1"))
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            offset = quaternion_matrix(_vector(geom, "quat", "1 0 0 0")) @ center + _vector(geom, "pos", "0 0 0")
            sensors.append({"side": side, "finger": finger, "chain": chain, "pad_offset_m": offset.tolist()})
    result = {
        "schema": 1,
        "frame": "per-hand root hand_C_MC; fixed wrist proxy at [0,0,0]",
        "units": "meters",
        "joint_units": "radians",
        "point_definition": "center of axis-aligned bounding box of elastomer mesh, transformed by MJCF geom pose; approximate pad volume center, not calibrated surface/contact center",
        "source": {k: manifest[k] for k in ("repository", "revision", "files")},
        "source_license": "Apache-2.0; Sharpa hand model; see upstream third_party/README.md",
        "joint_order": list(JOINTS),
        "sensor_order": list(SENSORS),
        "sensors": sensors,
    }
    result["hash"] = digest(result)
    return result


class HandGeometry:
    def __init__(self, spec=DEFAULT_SPEC):
        self.spec = json.loads(root_path(spec).read_text())
        if self.spec.get("hash") != digest({k: v for k, v in self.spec.items() if k != "hash"}):
            raise ValueError("Geometry spec hash mismatch")
        if self.spec["joint_order"] != list(JOINTS) or self.spec["sensor_order"] != list(SENSORS):
            raise ValueError("Geometry joint/sensor order mismatch")
        if [(s["side"], s["finger"]) for s in self.spec["sensors"]] != [
            (h, f) for h in ("left", "right") for f in FINGERS
        ]:
            raise ValueError("Geometry sensor definitions out of order")
        self.identity = self.spec["hash"]

    def positions(self, state):
        """Raw (...,58) radians -> (...,10,3) meters; arm joints are unused."""
        state = np.asarray(state, np.float64)
        if state.shape[-1] != 58 or not np.isfinite(state).all():
            raise ValueError("FK requires finite raw 58D state")
        points = []
        for sensor in self.spec["sensors"]:
            side = sensor["side"]
            q = state[..., 7:29] if side == "left" else state[..., 36:58]
            rotation = np.broadcast_to(np.eye(3), (*state.shape[:-1], 3, 3)).copy()
            position = np.zeros((*state.shape[:-1], 3))
            for step in sensor["chain"]:
                position += np.einsum("...ij,j->...i", rotation, step["pos"])
                rotation = rotation @ quaternion_matrix(step["quat_wxyz"])
                if "joint" in step:
                    index = JOINTS.index(step["joint"].removeprefix(side + "_"))
                    axis = np.asarray(step["axis"], np.float64)
                    axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    angle = q[..., index, None, None]
                    turn = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
                    rotation = rotation @ turn
            points.append(position + np.einsum("...ij,j->...i", rotation, sensor["pad_offset_m"]))
        return np.stack(points, axis=-2).astype(np.float32)


class GeometryCache:
    """Explicitly prepared cache. Loading never builds or downloads source data."""

    def __init__(self, data_root, spec=DEFAULT_SPEC):
        self.root = root_path(data_root)
        self.model = HandGeometry(spec)
        self.directory = self.root / "geometry_cache" / self.model.identity

    @lru_cache(maxsize=256)  # noqa: B019
    def episode(self, episode_id):
        path = self.directory / f"episode_{episode_id:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            identity = json.loads(str(data["identity"]))
            times, positions = data["timestamps"].copy(), data["positions"].copy()
        from openpi.trlt.data import _read

        subset = _read(self.root / "subset_manifest.json")
        if identity != {
            "geometry_hash": self.model.identity,
            "revision": subset["revision"],
            "subset_hash": subset["hash"],
            "episode_id": episode_id,
            "values_hash": _values_hash(times, positions),
        }:
            raise ValueError("Geometry cache provenance/content mismatch")
        if (
            positions.shape != (len(times), 10, 3)
            or not len(times)
            or not np.isfinite(times).all()
            or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Malformed geometry episode")
        return times, positions

    def window(self, episode_id, tactile_timestamps, tactile_valid, max_lag):
        times, points = self.episode(episode_id)
        stamps = np.asarray(tactile_timestamps, np.float64)
        # Same float32 precision tolerance as the tactile sampler, not a frame shift.
        tolerance = np.maximum(1e-7, 2 * np.abs(np.spacing(stamps.astype(np.float32))))
        indices = np.searchsorted(times, stamps + tolerance, side="right") - 1
        safe = np.clip(indices, 0, len(times) - 1)
        lag = stamps - times[safe]
        valid = np.asarray(tactile_valid, bool) & (indices >= 0) & (lag <= max_lag) & (lag >= -tolerance)
        positions = points[safe, np.arange(10)[None]]
        return {
            "tactile_positions": np.where(valid[..., None], positions, 0).astype(np.float32),
            "geometry_valid_mask": valid,
        }


def _values_hash(times, positions):
    return hashlib.sha256(np.ascontiguousarray(times).tobytes() + np.ascontiguousarray(positions).tobytes()).hexdigest()


def prepare_geometry(data_root, *, spec=DEFAULT_SPEC, split="train", episode_ids=None):
    from openpi.trlt.data import _read
    from openpi.trlt.data import read_episode

    root = root_path(data_root)
    subset = _read(root / "subset_manifest.json")
    splits = _read(root / "splits.json")
    ids = set(splits["splits"][split])
    if episode_ids is not None:
        if not set(episode_ids) <= ids:
            raise ValueError("Requested episodes do not belong to selected split")
        ids = set(episode_ids)
    cache = GeometryCache(root, spec)
    cache.directory.mkdir(parents=True, exist_ok=True)
    records = []
    for row in subset["episodes"]:
        eid = row["episode_index"]
        if eid not in ids:
            continue
        path = cache.directory / f"episode_{eid:06d}.npz"
        if not path.exists():
            ep = read_episode(root, row)
            times = ep["timestamp"].astype(np.float64)
            points = cache.model.positions(ep["observation.state"])
            identity = {
                "geometry_hash": cache.model.identity,
                "revision": subset["revision"],
                "subset_hash": subset["hash"],
                "episode_id": eid,
                "values_hash": _values_hash(times, points),
            }
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle, timestamps=times, positions=points, identity=np.array(json.dumps(identity, sort_keys=True))
                )
            temporary.replace(path)
        times, _ = cache.episode(eid)
        records.append({"episode_id": eid, "frames": len(times), "path": str(path.relative_to(root))})
    return {
        "geometry_hash": cache.model.identity,
        "split": split,
        "episodes": records,
        "frames": sum(r["frames"] for r in records),
    }
