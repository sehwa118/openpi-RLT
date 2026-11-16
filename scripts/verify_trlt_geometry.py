#!/usr/bin/env python3
"""Bounded CPU-only geometry verification; never reads test or launches GPU work."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
from dataclasses import replace
from datetime import UTC
from datetime import datetime
import importlib.util
import json
import subprocess
import xml.etree.ElementTree as ET

import jax
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.models.trlt import build_model
from openpi.models.trlt_test import small_config
from openpi.models.trlt_test import synthetic_batch
from openpi.trlt.cache import FeatureCache
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TrainConfig
from openpi.trlt.config import model_config_dict
from openpi.trlt.config import root_path
from openpi.trlt.data import TrexDataset
from openpi.trlt.data import read_episode
from openpi.trlt.geometry import DEFAULT_SPEC
from openpi.trlt.geometry import JOINTS
from openpi.trlt.geometry import SOURCE_REVISION
from openpi.trlt.geometry import HandGeometry
from openpi.trlt.loading import OrderedBatchSampler
from openpi.trlt.loading import TrainingCollator
from openpi.trlt.loading import training_loader
from openpi.trlt.runner import OBSERVATION_FIELDS
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import export_frozen_latents
from openpi.trlt.runner import train_action_probe
from openpi.trlt.runner import train_representation


def reference_fk(source, state):
    """Independent scipy/4x4 traversal of original MJCF; does not read extracted chains."""
    result = []
    for side, section in (("left", slice(7, 29)), ("right", slice(36, 58))):
        xml = ET.parse(source / side / f"{side}_sharpa_wave_with_wrist.xml").getroot()
        q = dict(zip((f"{side}_{j}" for j in JOINTS), state[section], strict=True))
        meshes = {m.get("name"): m for m in xml.findall("./asset/mesh")}
        points = {}

        def vector(node, key, default):
            return np.fromstring(node.get(key, default), sep=" ")

        def transform(node):
            quat = vector(node, "quat", "1 0 0 0")
            t = np.eye(4)
            t[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            t[:3, 3] = vector(node, "pos", "0 0 0")
            return t

        def visit(body, parent, q=q, meshes=meshes, side=side, points=points):
            t = parent @ transform(body)
            for joint in body.findall("joint"):
                hinge = np.eye(4)
                axis = vector(joint, "axis", "0 0 1")
                hinge[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * q[joint.get("name")]).as_matrix()
                t = t @ hinge
            for geom in body.findall("geom"):
                if geom.get("name", "").endswith("_elastomer"):
                    mesh = meshes[geom.get("mesh")]
                    raw = (source / side / mesh.get("file")).read_bytes()
                    count = int.from_bytes(raw[80:84], "little")
                    vertices = np.stack(
                        [
                            np.frombuffer(raw, dtype="<f4", count=9, offset=84 + i * 50 + 12).reshape(3, 3)
                            for i in range(count)
                        ]
                    )
                    flat = vertices.reshape(-1, 3) * vector(mesh, "scale", "1 1 1")
                    center = (flat.min(0) + flat.max(0)) / 2
                    points[geom.get("name")] = (t @ transform(geom) @ np.r_[center, 1])[:3]
            for child in body.findall("body"):
                visit(child, t)

        visit(xml.find("./worldbody/body"), np.eye(4))
        result.extend(points[f"{side}_{f}_elastomer"] for f in ("thumb", "index", "middle", "ring", "pinky"))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/trlt/geometry_validation")
    parser.add_argument("--source-dir", default=f"datasets/trex_squeeze/geometry_sources/{SOURCE_REVISION}")
    parser.add_argument("--baseline-ref", default="fef34900b7c903c5829eefe40490b80583ce0395")
    args = parser.parse_args()
    output = root_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_relative_to(root_path("experiments")):
        raise ValueError("Results belong under experiments/")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "date_utc": datetime.now(UTC).isoformat(),
        "device": str(jax.devices()),
        "test": "not-run",
        "gpu": "not-run",
    }
    assert jax.default_backend() == "cpu"
    model = HandGeometry()
    dataset = TrexDataset(geometry_spec=DEFAULT_SPEC, visual_cache=FeatureCache("datasets/trex_squeeze/visual_cache"))
    eid = dataset.queries[0][0]
    episode = read_episode(dataset.root, dataset.rows[eid])
    states = episode["observation.state"][[0, len(episode["timestamp"]) // 2, -1]]
    errors = [float(np.max(np.abs(model.positions(s) - reference_fk(root_path(args.source_dir), s)))) for s in states]
    assert max(errors) < 1e-7
    report["reference_fk"] = {
        "episode": eid,
        "frames": 3,
        "max_error_m": max(errors),
        "method": "independent raw MJCF/STL traversal using scipy Rotation and 4x4 transforms",
    }
    sampler = OrderedBatchSampler(3)
    sampler.indices = list(range(8))
    collator = TrainingCollator(OBSERVATION_FIELDS, 2, probe=False)
    with training_loader(dataset, sampler, collator, num_workers=0) as loader:
        sync = list(loader)
    with training_loader(dataset, sampler, collator, num_workers=2) as loader:
        worker = list(loader)
    for a, b in zip(jax.tree.leaves(sync), jax.tree.leaves(worker), strict=True):
        np.testing.assert_array_equal(a, b)
    report["real_loading"] = {
        "queries": 8,
        "episode": eid,
        "workers": 2,
        "exact_equal": True,
        "geometry_shape": list(dataset[4]["tactile_positions"].shape),
    }
    # Compare against actual pre-change source rather than two calls to current code.
    previous = output / "baseline_trlt.py"
    previous.write_bytes(subprocess.check_output(["git", "show", f"{args.baseline_ref}:src/openpi/models/trlt.py"]))
    spec = importlib.util.spec_from_file_location("baseline_trlt_geometry_validation", previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = []
    for arch in ("rlt_original", "trlt_self_only", "trlt_cross_only", "trlt_alternating"):
        cfg = small_config(arch)
        batch = synthetic_batch(cfg, batch_size=1)
        before = module.build_model(cfg)
        after = build_model(cfg)
        a = before.init(jax.random.key(0), batch)
        b = after.init(jax.random.key(0), batch)
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(
            before.apply(a, batch, method=before.encode), after.apply(b, batch, method=after.encode)
        )
        baseline.append({"architecture": arch, "params_and_latent_exact_equal": True})
    old_config = json.loads(root_path("configs/trlt/trlt_cross_only.json").read_text())["model"]
    from openpi.trlt.config import ModelConfig

    assert json.loads(json.dumps(model_config_dict(ModelConfig(**old_config)))) == old_config
    report["legacy_baseline"] = {"ref": args.baseline_ref, "models": baseline, "model_config_identity_preserved": True}
    training = []
    # Avoid expensive monolithic LLVM optimizer compilation in this CPU-only smoke.
    # Model JIT and gradients are separately covered by the tensor regression tests.
    jax.config.update("jax_disable_jit", val=True)
    for variant in ("mlp", "rope3d"):
        cfg = replace(
            small_config("trlt_cross_only"),
            num_sensors=10,
            embed_dim=8,
            num_heads=1,
            num_stages=2,
            decoder_layers=1,
            history_frames=2,
            tactile_resolution=4,
            cnn_channels=(2,),
            geometry_mlp_hidden=8,
            geometry_available=True,
            geometry_mode=variant,
            geometry_rope_axis_dim=2,
        )
        train = SyntheticDataset(cfg, size=3)
        val = SyntheticDataset(cfg, size=1, split="validation")
        train.provenance["geometry_hash"] = val.provenance["geometry_hash"] = "synthetic-geometry-fixture-v1"
        checkpoints = root_path(f"checkpoints/trlt/geometry_validation/{stamp}/{variant}")
        logs = output / stamp / variant
        rep = train_representation(
            train,
            val,
            cfg,
            TrainConfig(batch_size=2, microbatch_size=1, max_epochs=1),
            checkpoints / "representation",
            max_steps=2,
            log_dir=logs / "representation",
            progress=False,
            num_workers=2,
        )
        assert rep["manifest"]["actual_train_steps"] == 2
        latent_root = root_path(f"datasets/trex_squeeze/latents/geometry_validation/{stamp}/{variant}")
        tr = export_frozen_latents(train, rep["best_checkpoint"], latent_root / "train", batch_size=2)
        va = export_frozen_latents(val, rep["best_checkpoint"], latent_root / "validation", batch_size=2)
        probe = train_action_probe(
            tr,
            va,
            ProbeConfig(hidden_dim=8, depth=1, batch_size=2, max_epochs=1),
            checkpoints / "probe",
            max_steps=1,
            log_dir=logs / "probe",
            progress=False,
            num_workers=0,
        )
        assert probe["manifest"]["actual_train_steps"] == 1
        training.append(
            {
                "mode": variant,
                "representation_steps": 2,
                "probe_steps": 1,
                "representation_checkpoint": rep["best_checkpoint"],
                "workers": 2,
                "finite": True,
                "latent_geometry_hash": tr.provenance["geometry_hash"],
            }
        )
        jax.clear_caches()
    report["synthetic_pipeline"] = training
    report["synthetic_optimizer_jit"] = "disabled for bounded CPU smoke; model JIT covered separately"
    # Geometry sanity figure is an execution artifact, not a calibrated reconstruction.
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from openpi.trlt.geometry import quaternion_matrix

    fig = plt.figure(figsize=(11, 5))
    for i, side in enumerate(("left", "right")):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        state = states[1]
        points = model.positions(state)
        for local, sensor in enumerate(model.spec["sensors"][i * 5 : i * 5 + 5]):
            q = state[7:29] if i == 0 else state[36:58]
            rot = np.eye(3)
            pos = np.zeros(3)
            chain = [pos.copy()]
            for step in sensor["chain"]:
                pos = pos + rot @ step["pos"]
                rot = rot @ quaternion_matrix(step["quat_wxyz"])
                if "joint" in step:
                    axis = np.asarray(step["axis"])
                    axis = axis / np.linalg.norm(axis)
                    rot = (
                        rot
                        @ Rotation.from_rotvec(
                            axis * q[JOINTS.index(step["joint"].removeprefix(side + "_"))]
                        ).as_matrix()
                    )
                chain.append(pos.copy())
            chain = np.array([*chain, points[i * 5 + local]]) * 100
            ax.plot(*chain.T, marker=".", label=sensor["finger"])
            ax.scatter(*(points[i * 5 + local] * 100), s=35)
        ax.scatter(0, 0, 0, c="black", marker="x", s=60, label="wrist proxy")
        ax.set(xlabel="X [cm]", ylabel="Y [cm]", zlabel="Z [cm]", title=f"{side}, train episode {eid}, middle frame")
        ax.set_box_aspect((1, 1, 1))
        ax.legend(fontsize=7)
    fig.suptitle("FK pad-center proxies in independent hand-local frames (approximate)")
    fig.tight_layout()
    fig.savefig(output / "geometry_sanity.png", dpi=160)
    plt.close(fig)
    report["plot"] = str(output / "geometry_sanity.png")
    report["status"] = "passed"
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
