"""Numerical metrics and tiny GPU trainer integration; never consumes source test."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.trlt.config import ModelConfig
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TrainConfig
from openpi.trlt.metrics import ActionMetrics
from openpi.trlt.metrics import summarize_results
from openpi.trlt.runner import OBSERVATION_FIELDS
from openpi.trlt.runner import ActionProbe
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import _counts
from openpi.trlt.runner import _mean_metrics
from openpi.trlt.runner import collate
from openpi.trlt.runner import evaluate_action_probe
from openpi.trlt.runner import export_frozen_latents
from openpi.trlt.runner import load_checkpoint
from openpi.trlt.runner import parameter_hash
from openpi.trlt.runner import probe_loss
from openpi.trlt.runner import representation_objective
from openpi.trlt.runner import train_action_probe
from openpi.trlt.runner import train_representation


@pytest.fixture(autouse=True)
def highest_matmul_precision():
    # Numerical equivalence must compare float32 kernels, not batch-dependent
    # reduced-precision GPU GEMM selections.
    with jax.default_matmul_precision("highest"):
        yield


def tiny(mode="rlt_original"):
    return ModelConfig(
        mode=mode,
        input_dim=8,
        embed_dim=8,
        num_heads=2,
        num_stages=2,
        rlt_encoder_layers=1,
        decoder_layers=1,
        num_sensors=2,
        tactile_resolution=4,
        history_frames=2,
        cnn_channels=(4,),
    )


def test_metrics_episode_macro_and_mask():
    metric = ActionMetrics(np.zeros(58), np.ones(58) * 2)
    gt = np.zeros((3, 2, 58))
    pred = np.ones_like(gt)
    pred[0] *= 2
    pred[0, 1] = np.nan  # invalid future label/prediction cannot poison metrics
    metric.update(pred, gt, [[1, 0], [1, 1], [1, 1]], [1, 2, 2], ["a", "b", "b"])
    result = metric.result()
    assert result["primary_normalized_episode_macro_mse"] == 2.5
    assert result["normalized_micro_mse"] == pytest.approx(1.6)
    assert result["mse"] == pytest.approx(6.4)
    assert result["group_mse"]["left_hand"] == pytest.approx(6.4)
    assert result["first_action_mse"] == pytest.approx(8.0)


def test_metric_zero_baseline_and_paired_episodes(tmp_path):
    records = []
    for name, value in (("rlt_original", 0.0), ("trlt_cross_only", 1.0)):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "model": name,
                    "seed": 0,
                    "metrics": {
                        "primary_normalized_episode_macro_mse": value,
                        "episodes": {"0": {"normalized_mse": value, "valid_elements": 58}},
                    },
                }
            )
        )
        records.append(path)
    summary = summarize_results(records, bootstrap_samples=20)
    pair = summary["paired_vs_rlt"]["trlt_cross_only"]["0"]
    assert pair["relative_change_percent"] is None
    assert pair["paired_episode_bootstrap_95_ci"] == [1.0, 1.0]
    assert summary["models"]["rlt_original"]["std"] is None


def test_probe_mask_gradient_and_shape():
    config = ProbeConfig(hidden_dim=8)
    probe = ActionProbe(config)
    z, state, age = jnp.ones((2, 1, 8)), jnp.zeros((2, 58)), jnp.zeros(2)
    params = probe.init(jax.random.PRNGKey(0), z, state, age)
    out = jax.jit(lambda latent: probe.apply(params, latent, state, age))(z)
    assert out.shape == (2, 6, 58)
    mask = jnp.asarray([[True, False, False, False, False, False]] * 2)
    target = jnp.where(mask[..., None], 0.0, jnp.nan)
    value, gradient = jax.value_and_grad(probe_loss)(out, target, mask)
    assert np.isfinite(value)
    assert np.all(np.asarray(gradient)[:, 1:] == 0)
    z_grad = jax.grad(lambda latent: probe.apply(params, latent, state, age).sum())(z)
    assert np.all(np.asarray(z_grad) == 0)


@pytest.mark.parametrize("mode", ["rlt_original", "trlt_cross_only"])
def test_resume_accumulation_freeze_roundtrip(tmp_path, mode):
    cfg = tiny(mode)
    train = SyntheticDataset(cfg, size=4)
    val = SyntheticDataset(cfg, split="validation", size=2, seed=1)
    train_cfg = TrainConfig(batch_size=2, microbatch_size=1, max_epochs=3, patience=5)
    full = train_representation(
        train, val, cfg, train_cfg, tmp_path / "full", max_steps=2, log_dir=tmp_path / "full_logs"
    )
    first = train_representation(
        train, val, cfg, train_cfg, tmp_path / "resumed", max_steps=1, log_dir=tmp_path / "resume_logs"
    )
    assert first["manifest"]["cursor"] == 2
    # Serialization/device restore must preserve every parameter AND optimizer
    # buffer exactly before taking another step.
    from flax import serialization
    from flax.training import train_state

    from openpi.models.trlt import build_model
    from openpi.trlt.runner import _optimizer

    first_params, first_metadata = load_checkpoint(first["last_checkpoint"])
    template = train_state.TrainState.create(
        apply_fn=build_model(cfg).apply, params=first_params, tx=_optimizer(train_cfg)
    )
    restored, restored_metadata = load_checkpoint(first["last_checkpoint"], template)
    saved_dict = serialization.msgpack_restore((tmp_path / "resumed" / "last" / "state.msgpack").read_bytes())
    restored_dict = serialization.to_state_dict(restored)
    assert jax.tree_util.tree_structure(saved_dict) == jax.tree_util.tree_structure(restored_dict)
    for saved_leaf, restored_leaf in zip(
        jax.tree_util.tree_leaves(saved_dict), jax.tree_util.tree_leaves(restored_dict), strict=True
    ):
        np.testing.assert_array_equal(saved_leaf, restored_leaf)
        assert np.asarray(saved_leaf).dtype == np.asarray(restored_leaf).dtype
    assert restored_metadata["parameter_hash"] == first_metadata["parameter_hash"] == parameter_hash(restored.params)
    assert restored_metadata["cursor"] == 2
    assert restored_metadata["epoch"] == 0
    assert restored_metadata["seed"] == 0
    assert int(restored.step) == 1
    resumed = train_representation(
        train, val, cfg, train_cfg, tmp_path / "resumed", max_steps=2, resume=True, log_dir=tmp_path / "resume_logs"
    )
    a, am = load_checkpoint(full["last_checkpoint"])
    b, bm = load_checkpoint(resumed["last_checkpoint"])
    assert am["parameter_hash"] == parameter_hash(a)
    assert bm["parameter_hash"] == parameter_hash(b)
    # Independently compiled cuDNN continuation differed by one float32 ULP in
    # one CNN kernel (3.725e-9 measured on Blackwell). Roundtrip above is exact;
    # compare subsequent arithmetic tightly without assuming GPU bitwise math.
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-7, rtol=1e-6)
    assert (am["epoch"], am["cursor"], am["step"], am["seed"], am["rng_protocol"]) == (
        bm["epoch"],
        bm["cursor"],
        bm["step"],
        bm["seed"],
        bm["rng_protocol"],
    )
    full_events = [json.loads(line) for line in (tmp_path / "full_logs" / "train.jsonl").read_text().splitlines()]
    resumed_events = [json.loads(line) for line in (tmp_path / "resume_logs" / "train.jsonl").read_text().splitlines()]
    assert [(row["step"], row["epoch"], row["examples"]) for row in full_events] == [
        (row["step"], row["epoch"], row["examples"]) for row in resumed_events
    ]
    np.testing.assert_allclose(
        [row["loss"] for row in full_events], [row["loss"] for row in resumed_events], atol=1e-6, rtol=1e-6
    )
    # Compare the actual accumulation helper against the full-batch objective
    # before AdamW. Fused batch kernels can differ at float32 roundoff; Adam's
    # division by sqrt(v)+epsilon amplifies near-zero gradient differences, so
    # parameter equality across different microbatch sizes is not a valid test.
    from openpi.models.trlt import build_model

    model = build_model(cfg)
    samples = [train[0], train[1]]
    counts = _counts(samples, probe=False)
    denominators = jnp.asarray([counts["visual"], counts["tactile"]], jnp.float32)
    macro_loss, macro_grads = jax.jit(
        jax.value_and_grad(
            lambda params: model.apply({"params": params}, collate(samples, OBSERVATION_FIELDS), train=False)[0]
        )
    )(a)
    micro_fn = jax.jit(
        jax.value_and_grad(
            lambda params, batch: representation_objective(
                model, params, batch, denominators, jax.random.PRNGKey(0), cfg
            )[0]
        )
    )
    parts = [micro_fn(a, collate([sample], OBSERVATION_FIELDS)) for sample in samples]
    np.testing.assert_allclose(sum(part[0] for part in parts), macro_loss, rtol=2e-5, atol=2e-6)
    accumulated = jax.tree_util.tree_map(jnp.add, parts[0][1], parts[1][1])
    for x, y in zip(jax.tree_util.tree_leaves(accumulated), jax.tree_util.tree_leaves(macro_grads), strict=True):
        np.testing.assert_allclose(x, y, atol=3e-6, rtol=1e-4)
    if mode != "rlt_original":
        return
    frozen_train = export_frozen_latents(train, full["best_checkpoint"], tmp_path / "latents_train", batch_size=2)
    frozen_val = export_frozen_latents(val, full["best_checkpoint"], tmp_path / "latents_val", batch_size=2)
    repeat = export_frozen_latents(train, full["best_checkpoint"], tmp_path / "latents_train", batch_size=2)
    np.testing.assert_array_equal(frozen_train[0]["z"], repeat[0]["z"])
    probe_cfg = ProbeConfig(hidden_dim=8, batch_size=2, max_epochs=3)
    probe = train_action_probe(
        frozen_train, frozen_val, probe_cfg, tmp_path / "probe", max_steps=2, log_dir=tmp_path / "probe_logs"
    )
    unchanged, unchanged_meta = load_checkpoint(full["best_checkpoint"])
    assert unchanged_meta["parameter_hash"] == parameter_hash(unchanged)
    norms = train.normalization
    report = evaluate_action_probe(
        frozen_val, probe["best_checkpoint"], norms, tmp_path / "validation_metrics", batch_size=2
    )
    assert np.isfinite(report["metrics"]["primary_normalized_episode_macro_mse"])
    assert (tmp_path / "validation_metrics" / "predictions.jsonl.gz").exists()


def test_host_sensor_diagnostics_use_sensor_denominators():
    rows = [
        {
            "visual_loss": 2.0,
            "visual_weighted": 2.0,
            "total_loss": 3.0,
            "tactile_loss": 1.0,
            "tactile_weighted": 1.0,
            "tactile_per_sensor": np.array([1.0, 0.0]),
            "tactile_sensor_count": np.array([2, 0]),
            "tactile_per_sample": np.array([1.0, 0.0]),
        },
        {
            "visual_loss": 4.0,
            "visual_weighted": 4.0,
            "total_loss": 9.0,
            "tactile_loss": 5.0,
            "tactile_weighted": 5.0,
            "tactile_per_sensor": np.array([3.0, 7.0]),
            "tactile_sensor_count": np.array([1, 1]),
            "tactile_per_sample": np.array([5.0]),
        },
    ]
    result = _mean_metrics(rows, [{"visual": 2, "tactile": 2}, {"visual": 1, "tactile": 2}])
    np.testing.assert_allclose(result["tactile_per_sensor"], [5 / 3, 7])
    np.testing.assert_array_equal(result["tactile_sensor_count"], [3, 1])
    assert "tactile_per_sample" not in result
    assert result["total_loss"] == pytest.approx(8 / 3 + 3)


def test_host_summary_all_seed_metrics(tmp_path):
    paths = []
    for seed in range(3):
        value = float(seed + 1)
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "model": "rlt_original",
                    "seed": seed,
                    "metrics": {
                        "primary_normalized_episode_macro_mse": value,
                        "normalized_micro_mse": value * 2,
                        "mse": value * 3,
                        "rmse": value**0.5,
                        "group_mse": {"left_arm": value},
                        "joint_mse": [value] * 58,
                        "horizon_mse": [value] * 6,
                        "objects": {"ball": {"normalized_mse": value, "mse": value * 3}},
                        "episodes": {"11": {"normalized_mse": value, "mse": value * 3, "valid_elements": 58}},
                    },
                }
            )
        )
        paths.append(path)
    result = summarize_results(paths, bootstrap_samples=0)["models"]["rlt_original"]
    assert result["mean"] == 2
    assert result["std"] == 1
    metrics = result["metric_summary"]
    assert metrics["normalized_micro_mse"]["mean"] == 4
    assert metrics["group_mse"]["left_arm"]["std"] == 1
    assert metrics["joint_mse"][57]["seeds"] == {"0": 1.0, "1": 2.0, "2": 3.0}
    assert metrics["horizon_mse"][5]["missing_expected_seeds"] == []
    assert metrics["objects"]["ball"]["mse"]["mean"] == 6
    assert "valid_elements" not in metrics["episodes"]["11"]


def test_host_missing_normalization_cannot_train():
    from types import SimpleNamespace

    from openpi.trlt.runner import _compatible

    training = SimpleNamespace(provenance={"split": "train", "normalization_hash": None})
    validation = SimpleNamespace(provenance={"split": "validation", "normalization_hash": None})
    with pytest.raises(ValueError, match="train-only"):
        _compatible(training, validation)


def test_host_normalization_content_tamper_rejected(tmp_path):
    import copy

    from openpi.trlt.config import digest
    from openpi.trlt.runner import _validated_normalization

    stats = {"action": {"mean": [0.0] * 58, "std": [1.0] * 58}}
    stats["hash"] = digest(stats)
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(stats))
    assert _validated_normalization(path, stats["hash"])["action"]["mean"] == [0.0] * 58
    tampered = copy.deepcopy(stats)
    tampered["action"]["mean"][0] = 100.0
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="content hash"):
        _validated_normalization(path, stats["hash"])
    with pytest.raises(ValueError, match="content hash"):
        _validated_normalization(tampered, stats["hash"])


def test_host_preprocessing_compatibility_and_json_roundtrip():
    from openpi.trlt.runner import _check_provenance

    expected = {"tactile_resolution": 64, "camera_order": ["a", "b", "c"], "sensor_order": ["left", "right"]}
    same = {**expected, "camera_order": ("a", "b", "c")}
    _check_provenance(expected, same, "fixture")
    for key, changed in (
        ("tactile_resolution", 63),
        ("camera_order", ["b", "a", "c"]),
        ("sensor_order", ["right", "left"]),
    ):
        with pytest.raises(ValueError, match=key):
            _check_provenance(expected, {**same, key: changed}, "fixture")


def test_host_query_manifest_mismatch_rejected(tmp_path):
    from openpi.trlt.runner import _check_provenance

    with pytest.raises(ValueError, match="query_manifest_hash"):
        _check_provenance({"query_manifest_hash": "schedule-a"}, {"query_manifest_hash": "schedule-b"}, "fixture")
    paths = []
    for model, query_hash in (("rlt_original", "schedule-a"), ("trlt_cross_only", "schedule-b")):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "seed": 0,
                    "provenance": {"query_manifest_hash": query_hash},
                    "metrics": {
                        "primary_normalized_episode_macro_mse": 1.0,
                        "episodes": {"0": {"normalized_mse": 1.0, "valid_elements": 58}},
                    },
                }
            )
        )
        paths.append(path)
    with pytest.raises(ValueError, match="query_manifest_hash"):
        summarize_results(paths, bootstrap_samples=0)
