"""CPU unit and integration checks for Stage B attention-actor architectures."""

import importlib.util
import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.trlt_actor import ActionProbe as ActorActionProbe
from openpi.models.trlt_actor import PartAttentionActionProbe
from openpi.models.trlt_actor import ResMLPActionProbe
from openpi.models.trlt_actor import build_action_probe
from openpi.models.trlt_actor import part_attention_mask
from openpi.models.trlt_actor import part_specs_for_dim
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import probe_config_dict
from openpi.trlt.config import probe_variant_name
from openpi.trlt.config import root_path
from openpi.trlt.metrics import summarize_results
from openpi.trlt.runner import ActionProbe
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import evaluate_action_probe
from openpi.trlt.runner import probe_loss
from openpi.trlt.runner import train_action_probe
from openpi.trlt.runner_test import tiny


class FrozenLatentFixture(SyntheticDataset):
    """Lightweight deterministic frozen Stage A cache fixture for CPU probe tests."""

    def __init__(self, split: str, *, size: int = 4, latent_dim: int = 16):
        super().__init__(tiny(), size=size, split="validation" if split == "test" else split)
        self.latent_dim = latent_dim
        self.provenance.update(
            {
                "split": split,
                "frozen": True,
                "model": "trlt_cross_only",
                "split_manifest_hash": "split-h",
                "query_manifest_hash": "query-h",
                "revision": "rev-1",
                "dataset_revision": "drev-1",
                "temporal": {"action_horizon_frames": 6},
                "camera_order": ["head_left"],
                "sensor_order": ["left_thumb"],
                "feature_cache_hash": "fcache-h",
            }
        )
        self.cache_hash = f"latent-cache-{split}"

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        rng = np.random.default_rng(1000 + index)
        sample["z"] = rng.normal(0.0, 0.5, size=(1, self.latent_dim)).astype(np.float32)
        sample["proprioception"] = rng.normal(0.0, 0.5, size=(58,)).astype(np.float32)
        sample["action_target"] = rng.normal(0.0, 0.5, size=(6, 58)).astype(np.float32)
        sample["visual_age_seconds"] = np.float32(0.02 * (index + 1))
        return sample


def _param_count(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))


@pytest.mark.parametrize(
    ("action_dim", "expected_parts", "expected_counts"),
    [
        (58, 12, {"full": 144, "kinematic": 34, "hierarchical": 64}),
        (22, 5, {"full": 25, "kinematic": 13, "hierarchical": 21}),
    ],
)
def test_part_specs_and_mask_topology(action_dim, expected_parts, expected_counts):
    parts = part_specs_for_dim(action_dim)
    assert len(parts) == expected_parts
    assert parts[0].start == 0
    assert parts[-1].end == action_dim
    for left, right in itertools.pairwise(parts):
        assert left.end == right.start
    for part in parts:
        assert part.dim > 0

    for mode, expected_edges in expected_counts.items():
        mask = np.asarray(part_attention_mask(parts, mode))
        assert mask.shape == (expected_parts, expected_parts)
        assert mask.dtype == bool
        assert bool(np.all(np.diag(mask)))
        assert int(mask.sum()) == expected_edges

    with pytest.raises(ValueError, match="attention_mask_mode"):
        part_attention_mask(parts, "invalid")


def test_invalid_action_dim_and_config_validation():
    with pytest.raises(ValueError, match="action_dim"):
        part_specs_for_dim(14)
    with pytest.raises(ValueError, match="part_attention"):
        ProbeConfig(architecture="part_attention", action_dim=14)
    with pytest.raises(ValueError, match="dividing"):
        ProbeConfig(architecture="part_attention", hidden_dim=10, num_heads=4)
    with pytest.raises(ValueError, match="architecture"):
        ProbeConfig(architecture="transformer")


@pytest.mark.parametrize("action_dim", [58, 22])
@pytest.mark.parametrize(
    ("architecture", "mask_mode"),
    [
        ("mlp", "full"),
        ("resmlp", "full"),
        ("part_attention", "full"),
        ("part_attention", "kinematic"),
        ("part_attention", "hierarchical"),
    ],
)
def test_forward_shapes_finiteness_and_stop_gradient(action_dim, architecture, mask_mode):
    cfg = ProbeConfig(
        architecture=architecture,
        attention_mask_mode=mask_mode,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=action_dim,
        horizon=4,
    )
    model = build_action_probe(cfg)
    z = jnp.ones((3, 1, 12), dtype=jnp.float32) * 0.25
    prop = jnp.linspace(-0.5, 0.5, 3 * action_dim, dtype=jnp.float32).reshape((3, action_dim))
    age = jnp.array([0.0, 0.02, 0.04], dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), z, prop, age, train=False)
    assert set(variables) == {"params"}
    out = model.apply(variables, z, prop, age, train=False)
    assert out.shape == (3, 4, action_dim)
    assert bool(jnp.all(jnp.isfinite(out)))

    def scalar_out(z_in, prop_in):
        return jnp.sum(model.apply(variables, z_in, prop_in, age, train=False) ** 2)

    grad_z, grad_prop = jax.grad(scalar_out, argnums=(0, 1))(z, prop)
    assert bool(jnp.all(grad_z == 0))
    assert float(jnp.max(jnp.abs(grad_prop))) > 0.0


def test_adaln_zero_non_deadlock_across_steps():
    cfg = ProbeConfig(
        architecture="part_attention",
        attention_mask_mode="full",
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=58,
        horizon=2,
    )
    model = PartAttentionActionProbe(cfg)
    rng = np.random.default_rng(7)
    z = jnp.asarray(rng.normal(size=(4, 1, 12)), dtype=jnp.float32)
    prop = jnp.asarray(rng.normal(size=(4, 58)), dtype=jnp.float32)
    age = jnp.array([0.01, 0.02, 0.03, 0.04], dtype=jnp.float32)
    target = jnp.asarray(rng.normal(size=(4, 2, 58)), dtype=jnp.float32)
    valid = jnp.ones((4, 2), dtype=bool)

    params = model.init(jax.random.PRNGKey(1), z, prop, age, train=False)["params"]
    tx = optax.adamw(learning_rate=1e-2)
    opt_state = tx.init(params)

    def loss_fn(p):
        pred = model.apply({"params": p}, z, prop, age, train=False)
        return probe_loss(pred, target, valid)

    # Step 1: gates start at 0, so attention/FFN internal projections have 0 grad while gates/heads learn.
    _, grads_step1 = jax.value_and_grad(loss_fn)(params)
    assert float(jnp.max(jnp.abs(grads_step1["block_0"]["query"]["kernel"]))) == 0.0
    assert float(jnp.max(jnp.abs(grads_step1["block_0"]["ffn_fc1"]["kernel"]))) == 0.0
    assert float(jnp.max(jnp.abs(grads_step1["block_0"]["adaln_modulation"]["kernel"]))) > 0.0
    assert float(jnp.max(jnp.abs(grads_step1["final_adaln"]["kernel"]))) > 0.0
    assert float(jnp.max(jnp.abs(grads_step1["part_out_left_arm"]["kernel"]))) > 0.0

    updates_step1, opt_state = tx.update(grads_step1, opt_state, params)
    params_after_1 = optax.apply_updates(params, updates_step1)
    assert float(jnp.max(jnp.abs(params_after_1["block_0"]["adaln_modulation"]["kernel"]))) > 0.0

    # Step 2: once gates open, attention Q/K/V/out, FFN, and conditioning projection all receive nonzero gradients.
    _, grads_step2 = jax.value_and_grad(loss_fn)(params_after_1)
    for weight_name in ("query", "key", "value", "attn_out", "ffn_fc1", "ffn_fc2"):
        assert float(jnp.max(jnp.abs(grads_step2["block_0"][weight_name]["kernel"]))) > 0.0
    assert float(jnp.max(jnp.abs(grads_step2["cond_fc1"]["kernel"]))) > 0.0
    assert float(jnp.max(jnp.abs(grads_step2["cond_fc2"]["kernel"]))) > 0.0

    updates_step2, _ = tx.update(grads_step2, opt_state, params_after_1)
    params_after_2 = optax.apply_updates(params_after_1, updates_step2)
    diff_q = jnp.max(
        jnp.abs(params_after_2["block_0"]["query"]["kernel"] - params["block_0"]["query"]["kernel"])
    )
    assert float(diff_q) > 0.0


def _trained_params_for_mask(mode: str, *, depth: int):
    cfg = ProbeConfig(
        architecture="part_attention",
        attention_mask_mode=mode,
        hidden_dim=16,
        depth=depth,
        num_heads=4,
        action_dim=58,
        horizon=2,
    )
    model = PartAttentionActionProbe(cfg)
    rng = np.random.default_rng(13)
    z = jnp.asarray(rng.normal(size=(2, 1, 12)), dtype=jnp.float32)
    prop = jnp.asarray(rng.normal(size=(2, 58)), dtype=jnp.float32)
    age = jnp.array([0.01, 0.03], dtype=jnp.float32)
    target = jnp.asarray(rng.normal(size=(2, 2, 58)), dtype=jnp.float32)
    valid = jnp.ones((2, 2), dtype=bool)
    params = model.init(jax.random.PRNGKey(2), z, prop, age, train=False)["params"]
    tx = optax.adamw(learning_rate=5e-2)
    opt_state = tx.init(params)

    def loss_fn(p):
        return probe_loss(model.apply({"params": p}, z, prop, age, train=False), target, valid)

    for _ in range(3):
        _, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return model, params, z, prop, age


def test_mask_isolation_kinematic_and_hierarchical():
    parts = {part.name: part for part in part_specs_for_dim(58)}

    # 1. Kinematic 1-layer isolation: left_thumb connects only to left_thumb + left_arm.
    kin_model, kin_params, z, prop, age = _trained_params_for_mask("kinematic", depth=1)
    perturbed = prop.at[:, parts["left_thumb"].start : parts["left_thumb"].end].add(2.0)
    base_out = kin_model.apply({"params": kin_params}, z, prop, age, train=False)
    pert_out = kin_model.apply({"params": kin_params}, z, perturbed, age, train=False)
    delta = np.asarray(jnp.abs(pert_out - base_out))

    assert delta[:, :, parts["left_thumb"].start : parts["left_thumb"].end].max() > 1e-4
    assert delta[:, :, parts["left_arm"].start : parts["left_arm"].end].max() > 1e-4
    for isolated in ("left_index", "right_arm", "right_thumb"):
        assert delta[:, :, parts[isolated].start : parts[isolated].end].max() == 0.0

    # 2. Hierarchical 2-layer isolation: top-down only (no finger->arm, no cross-hand finger leak).
    hier_model, hier_params, z, prop, age = _trained_params_for_mask("hierarchical", depth=2)
    base_h = hier_model.apply({"params": hier_params}, z, prop, age, train=False)

    # Finger perturbations cannot reach any arm.
    pert_lt = prop.at[:, parts["left_thumb"].start : parts["left_thumb"].end].add(2.0)
    delta_lt = np.asarray(
        jnp.abs(hier_model.apply({"params": hier_params}, z, pert_lt, age, train=False) - base_h)
    )
    assert delta_lt[:, :, parts["left_arm"].start : parts["left_arm"].end].max() == 0.0
    assert delta_lt[:, :, parts["right_arm"].start : parts["right_arm"].end].max() == 0.0

    # Right finger perturbations cannot reach any left finger.
    pert_rt = prop.at[:, parts["right_thumb"].start : parts["right_thumb"].end].add(2.0)
    delta_rt = np.asarray(
        jnp.abs(hier_model.apply({"params": hier_params}, z, pert_rt, age, train=False) - base_h)
    )
    assert delta_rt[:, :, parts["left_arm"].start : parts["left_arm"].end].max() == 0.0
    assert delta_rt[:, :, parts["left_thumb"].start : parts["left_thumb"].end].max() == 0.0

    # Right arm DOES propagate to left arm and (over 2 hops) top-down to left_thumb.
    pert_ra = prop.at[:, parts["right_arm"].start : parts["right_arm"].end].add(2.0)
    delta_ra = np.asarray(
        jnp.abs(hier_model.apply({"params": hier_params}, z, pert_ra, age, train=False) - base_h)
    )
    assert delta_ra[:, :, parts["left_arm"].start : parts["left_arm"].end].max() > 1e-4
    assert delta_ra[:, :, parts["left_thumb"].start : parts["left_thumb"].end].max() > 1e-4


def test_parameter_counts_matched_for_configs():
    z = jnp.zeros((1, 1, 512), dtype=jnp.float32)
    prop = jnp.zeros((1, 58), dtype=jnp.float32)
    age = jnp.zeros((1,), dtype=jnp.float32)

    counts = {}
    for name in ("probe_mlp.json", "probe_resmlp.json", "probe_part_attn_full.json"):
        raw = json.loads(root_path(f"configs/trlt/attention_actor/{name}").read_text())
        cfg = ProbeConfig(**raw["probe"])
        model = build_action_probe(cfg)
        params = model.init(jax.random.PRNGKey(0), z, prop, age, train=False)["params"]
        counts[name] = _param_count(params)

    assert counts["probe_mlp.json"] == 301_660
    assert counts["probe_resmlp.json"] == 764_412
    assert counts["probe_part_attn_full.json"] == 764_380
    rel_diff = abs(counts["probe_resmlp.json"] - counts["probe_part_attn_full.json"]) / counts["probe_resmlp.json"]
    assert rel_diff < 0.01


def test_legacy_probe_config_and_action_probe_compatibility():
    legacy_keys = {
        "hidden_dim",
        "depth",
        "action_dim",
        "horizon",
        "dropout_rate",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
    }
    assert set(probe_config_dict(ProbeConfig())) == legacy_keys

    mlp_raw = json.loads(root_path("configs/trlt/attention_actor/probe_mlp.json").read_text())
    assert ProbeConfig(**mlp_raw["probe"]) == ProbeConfig()
    assert probe_variant_name(ProbeConfig()) == "mlp"
    assert probe_variant_name(ProbeConfig(architecture="resmlp")) == "resmlp"
    assert (
        probe_variant_name(ProbeConfig(architecture="part_attention", attention_mask_mode="kinematic"))
        == "part_attention_kinematic"
    )

    assert ActionProbe is ActorActionProbe
    probe = ActionProbe(ProbeConfig(hidden_dim=8))
    params = probe.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 1, 8), dtype=jnp.float32),
        jnp.zeros((1, 58), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        train=False,
    )["params"]
    assert set(params) == {"hidden_0", "hidden_1", "output"}
    assert probe_config_dict(ProbeConfig(architecture="resmlp"))["architecture"] == "resmlp"
    cosine_dict = probe_config_dict(
        ProbeConfig(architecture="part_attention", attention_mask_mode="kinematic", lr_schedule="cosine", warmup_steps=5)
    )
    assert cosine_dict["lr_schedule"] == "cosine"
    assert cosine_dict["warmup_steps"] == 5


def test_end_to_end_cpu_train_resume_evaluate_and_summarize(tmp_path):
    train_ds = FrozenLatentFixture("train", size=4)
    val_ds = FrozenLatentFixture("validation", size=2)
    test_ds = FrozenLatentFixture("test", size=2)
    norm = train_ds.normalization

    configs = [
        ("mlp", ProbeConfig(hidden_dim=8, batch_size=2, max_epochs=2, patience=5)),
        (
            "resmlp",
            ProbeConfig(
                architecture="resmlp",
                hidden_dim=16,
                depth=2,
                batch_size=2,
                max_epochs=2,
                patience=5,
                learning_rate=5e-4,
                lr_schedule="cosine",
                warmup_steps=1,
            ),
        ),
        (
            "part_attention_kinematic",
            ProbeConfig(
                architecture="part_attention",
                attention_mask_mode="kinematic",
                hidden_dim=16,
                depth=2,
                num_heads=4,
                batch_size=2,
                max_epochs=2,
                patience=5,
                learning_rate=5e-4,
                lr_schedule="cosine",
                warmup_steps=1,
            ),
        ),
    ]

    metrics_paths = []
    for tag, cfg in configs:
        ckpt_dir = tmp_path / tag / "ckpt"
        log_dir = tmp_path / tag / "logs"
        result = train_action_probe(
            train_ds,
            val_ds,
            cfg,
            ckpt_dir,
            seed=0,
            max_steps=1 if tag == "part_attention_kinematic" else 2,
            log_dir=log_dir,
            num_workers=0,
            progress=False,
        )
        if tag == "part_attention_kinematic":
            result = train_action_probe(
                train_ds,
                val_ds,
                cfg,
                ckpt_dir,
                seed=0,
                max_steps=2,
                resume=True,
                log_dir=log_dir,
                num_workers=0,
                progress=False,
            )
            assert result["manifest"]["actual_train_steps"] == 2
        eval_dir = tmp_path / tag / "eval"
        report = evaluate_action_probe(test_ds, result["best_checkpoint"], norm, eval_dir, batch_size=2)
        expected_model = "trlt_cross_only" if tag == "mlp" else f"trlt_cross_only__{tag}"
        assert report["model"] == expected_model
        assert report["probe_variant"] == tag
        assert np.isfinite(report["metrics"]["primary_normalized_episode_macro_mse"])
        metrics_paths.append(eval_dir / "metrics.json")

    assert isinstance(ResMLPActionProbe(configs[1][1]), ResMLPActionProbe)
    summary = summarize_results(metrics_paths, tmp_path / "summary.json", bootstrap_samples=10, baseline_model="trlt_cross_only")
    assert summary["baseline_model"] == "trlt_cross_only"
    assert set(summary["models"]) == {
        "trlt_cross_only",
        "trlt_cross_only__resmlp",
        "trlt_cross_only__part_attention_kinematic",
    }
    assert set(summary["paired_vs_baseline"]) == {
        "trlt_cross_only__resmlp",
        "trlt_cross_only__part_attention_kinematic",
    }


def test_cli_dry_run_all_five_attention_actor_configs(capsys):
    script_path = root_path("scripts/trex_offline.py")
    spec = importlib.util.spec_from_file_location("trex_offline", script_path)
    trex_offline = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trex_offline)

    expected = {
        "probe_mlp.json": ("mlp", "full"),
        "probe_resmlp.json": ("resmlp", "full"),
        "probe_part_attn_full.json": ("part_attention", "full"),
        "probe_part_attn_kinematic.json": ("part_attention", "kinematic"),
        "probe_part_attn_hierarchical.json": ("part_attention", "hierarchical"),
    }
    for filename, (arch, mask_mode) in expected.items():
        cfg_path = root_path(f"configs/trlt/attention_actor/{filename}")
        trex_offline.main(
            [
                "train-probe",
                "--config",
                str(cfg_path),
                "--train-latents",
                "/tmp/t",
                "--validation-latents",
                "/tmp/v",
                "--dry-run",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == "train-probe"
        assert payload["probe"]["architecture"] == arch
        assert payload["probe"]["attention_mask_mode"] == mask_mode
        assert "plan-only" in payload["status"]


@pytest.mark.parametrize(
    ("architecture", "mask_mode"),
    [
        ("resmlp", "full"),
        ("part_attention", "full"),
        ("part_attention", "kinematic"),
        ("part_attention", "hierarchical"),
    ],
)
def test_dropout_train_mode_and_single_sample_batch(architecture, mask_mode):
    cfg = ProbeConfig(
        architecture=architecture,
        attention_mask_mode=mask_mode,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=58,
        horizon=6,
        dropout_rate=0.1,
    )
    model = build_action_probe(cfg)
    z = jnp.ones((1, 1, 12), dtype=jnp.float32) * 0.5
    prop = jnp.linspace(-0.2, 0.2, 58, dtype=jnp.float32).reshape((1, 58))
    age = jnp.array([0.03], dtype=jnp.float32)
    target = jnp.zeros((1, 6, 58), dtype=jnp.float32)
    valid = jnp.ones((1, 6), dtype=bool)

    variables = model.init(jax.random.PRNGKey(0), z, prop, age, train=False)
    assert set(variables) == {"params"}

    def loss_with_dropout(params, z_in):
        pred = model.apply(
            {"params": params},
            z_in,
            prop,
            age,
            train=True,
            rngs={"dropout": jax.random.PRNGKey(42)},
        )
        return probe_loss(pred, target, valid), pred

    (loss_val, pred_out), (grad_params, grad_z) = jax.value_and_grad(
        lambda p, z_in: (loss_with_dropout(p, z_in)[0], loss_with_dropout(p, z_in)[1]),
        argnums=(0, 1),
        has_aux=True,
    )(variables["params"], z)
    assert pred_out.shape == (1, 6, 58)
    assert bool(jnp.isfinite(loss_val))
    assert bool(jnp.all(jnp.isfinite(pred_out)))
    assert bool(jnp.all(grad_z == 0))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grad_params))
    max_param_grad = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(grad_params))
    assert max_param_grad > 0.0


def test_assembly_22d_parameter_parity_and_dry_run_configs(capsys):
    script_path = root_path("scripts/trex_offline.py")
    spec = importlib.util.spec_from_file_location("trex_offline", script_path)
    trex_offline = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trex_offline)

    z = jnp.zeros((2, 1, 512), dtype=jnp.float32)
    prop = jnp.zeros((2, 22), dtype=jnp.float32)
    age = jnp.zeros((2,), dtype=jnp.float32)

    expected = {
        "probe_mlp.json": ("mlp", "full", 219_970),
        "probe_resmlp.json": ("resmlp", "full", 724_882),
        "probe_part_attn_full.json": ("part_attention", "full", 721_602),
        "probe_part_attn_kinematic.json": ("part_attention", "kinematic", 721_602),
        "probe_part_attn_hierarchical.json": ("part_attention", "hierarchical", 721_602),
    }
    counts = {}
    for filename, (arch, mask_mode, expected_count) in expected.items():
        cfg_path = root_path(f"configs/trlt/3dtacdex_assembly/attention_actor/{filename}")
        raw = json.loads(cfg_path.read_text())
        assert raw["tracking"]["mode"] == "disabled"
        assert raw["dataset_kind"] == "3dtacdex_assembly"
        cfg = ProbeConfig(**raw["probe"])
        params = build_action_probe(cfg).init(jax.random.PRNGKey(0), z, prop, age, train=False)["params"]
        counts[filename] = _param_count(params)
        assert counts[filename] == expected_count

        trex_offline.main(
            [
                "train-probe",
                "--config",
                str(cfg_path),
                "--train-latents",
                "/tmp/t",
                "--validation-latents",
                "/tmp/v",
                "--dry-run",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["probe"]["architecture"] == arch
        assert payload["probe"]["attention_mask_mode"] == mask_mode
        assert payload["probe"]["action_dim"] == 22
        assert payload["probe"]["horizon"] == 3
        assert payload["tracking"]["mode"] == "disabled"

    rel_diff = (
        abs(counts["probe_resmlp.json"] - counts["probe_part_attn_full.json"]) / counts["probe_part_attn_full.json"]
    )
    assert rel_diff < 0.005


def test_assembly_elementwise_mask_and_action_groups(tmp_path):
    class AssemblyLatentFixture(FrozenLatentFixture):
        def __init__(self, split: str, *, size: int = 4):
            super().__init__(split, size=size, latent_dim=16)
            norm = {
                "state": {"mean": [0.0] * 22, "std": [1.0] * 22},
                "action": {"mean": [0.0] * 22, "std": [1.0] * 22},
                "action_groups": {"arm": list(range(6)), "hand": list(range(6, 22))},
                "split_hash": "assembly-split-h",
            }
            norm["hash"] = digest(norm)
            self.normalization = norm
            self.provenance.update(
                {
                    "dataset_kind": "3dtacdex_assembly",
                    "model": "trlt_self_only",
                    "normalization_hash": norm["hash"],
                    "temporal": {**vars(TemporalConfig()), "fps": 5.0, "action_horizon_frames": 3},
                }
            )

        def __getitem__(self, index: int):
            sample = super().__getitem__(index)
            rng = np.random.default_rng(2000 + index)
            sample["proprioception"] = rng.normal(0.0, 0.5, size=(22,)).astype(np.float32)
            target = rng.normal(0.0, 0.5, size=(3, 22)).astype(np.float32)
            mask = np.ones((3, 22), dtype=bool)
            if index % 2 == 1:
                mask[:, 6:] = False
                target[:, 6:] = 0.0
            sample["action_target"] = target
            sample["action_valid_mask"] = mask
            return sample

    train_ds = AssemblyLatentFixture("train", size=4)
    val_ds = AssemblyLatentFixture("validation", size=2)
    test_ds = AssemblyLatentFixture("test", size=2)
    cfg = ProbeConfig(
        architecture="part_attention",
        attention_mask_mode="hierarchical",
        hidden_dim=16,
        depth=2,
        num_heads=4,
        action_dim=22,
        horizon=3,
        batch_size=2,
        max_epochs=2,
        patience=5,
        learning_rate=5e-4,
        lr_schedule="cosine",
        warmup_steps=1,
    )
    trained = train_action_probe(
        train_ds,
        val_ds,
        cfg,
        tmp_path / "ckpt",
        seed=0,
        max_steps=2,
        log_dir=tmp_path / "logs",
        num_workers=0,
        progress=False,
    )
    report = evaluate_action_probe(
        test_ds,
        trained["best_checkpoint"],
        train_ds.normalization,
        tmp_path / "eval",
        batch_size=2,
    )
    assert report["model"] == "trlt_self_only__part_attention_hierarchical"
    assert set(report["metrics"]["group_mse"]) == {"arm", "hand"}
    assert np.isfinite(report["metrics"]["group_mse"]["arm"])
    assert np.isfinite(report["metrics"]["group_mse"]["hand"])
