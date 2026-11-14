"""Bounded real-data GPU smoke checks. Never reads the test split or runs full training."""

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("features", "models"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--data-root", default="datasets/trex_squeeze")
    parser.add_argument("--backbone-checkpoint", default="checkpoints/openpi-assets/checkpoints/pi05_base/params")
    parser.add_argument("--train-episodes", nargs="+", type=int, required=True)
    parser.add_argument("--validation-episodes", nargs="+", type=int, required=True)
    parser.add_argument("--queries-per-episode", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--output-dir", default="experiments/trlt/real_smoke")
    parser.add_argument("--checkpoint-root", default="checkpoints/trlt/smoke")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force-cache-rebuild", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 20 or not 1 <= args.queries_per_episode <= 20:
        parser.error("Smoke is bounded to20steps and20queries per episode")
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi.models.trlt import Fusion
    from openpi.models.trlt import build_model
    from openpi.trlt.cache import FeatureCache
    from openpi.trlt.cache import write_json
    from openpi.trlt.config import MODELS
    from openpi.trlt.config import ModelConfig
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.config import TrainConfig
    from openpi.trlt.config import root_path
    from openpi.trlt.data import TrexDataset
    from openpi.trlt.features import extract_visual_cache
    from openpi.trlt.runner import OBSERVATION_FIELDS
    from openpi.trlt.runner import PROBE_FIELDS
    from openpi.trlt.runner import ActionProbe
    from openpi.trlt.runner import assert_gpu
    from openpi.trlt.runner import collate
    from openpi.trlt.runner import export_frozen_latents
    from openpi.trlt.runner import load_checkpoint
    from openpi.trlt.runner import parameter_hash
    from openpi.trlt.runner import probe_loss
    from openpi.trlt.runner import train_action_probe
    from openpi.trlt.runner import train_representation

    environment = assert_gpu()
    output = root_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features_root = root_path(args.data_root) / "features_smoke"
    datasets = []
    for split, episode_ids in (("train", args.train_episodes), ("validation", args.validation_episodes)):
        dataset = TrexDataset(data_root=args.data_root, split=split)
        allowed = set(episode_ids)
        if not allowed.issubset(dataset.rows):
            raise ValueError(f"Requested {split} IDs are outside fixed split")
        count = {}
        queries = []
        for episode, query in dataset.queries:
            if episode in allowed and count.get(episode, 0) < args.queries_per_episode:
                queries.append((episode, query))
                count[episode] = count.get(episode, 0) + 1
        dataset.queries = queries
        dataset.provenance = {
            **dataset.provenance,
            "debug_query_hash": __import__("hashlib").sha256(str(queries).encode()).hexdigest(),
        }
        datasets.append(dataset)
    train, validation = datasets
    if args.stage == "features":
        # One backbone instance for both allowed splits, then process exits to release memory.
        train.rows = {**train.rows, **validation.rows}
        train.queries = train.queries + validation.queries
        report = extract_visual_cache(
            train,
            backbone_checkpoint=args.backbone_checkpoint,
            output_dir=features_root,
            revision=train.provenance["revision"],
            episode_manifest_hash=train.provenance["episode_manifest_hash"],
            force=args.force_cache_rebuild,
        )
        write_json(output / "features.json", report)
        print(json.dumps(report, indent=2))
        return
    feature_cache = FeatureCache(features_root)
    for dataset in datasets:
        dataset.visual_cache = feature_cache
        dataset.provenance["feature_cache_hash"] = feature_cache.cache_hash
    sample = train[0]
    batch = collate([sample], OBSERVATION_FIELDS)
    results = {
        "environment": environment,
        "train_episode_ids": args.train_episodes,
        "validation_episode_ids": args.validation_episodes,
        "test_used": False,
        "models": {},
    }
    modes = args.models or MODELS
    if any(mode not in MODELS for mode in modes):
        raise ValueError("Unknown model")
    for mode in modes:
        cfg = ModelConfig(mode=mode, input_dim=sample["visual_features"].shape[-1])
        model = build_model(cfg)
        variables = model.init(jax.random.key(0), batch, train=False)
        initial_loss = float(model.apply(variables, batch, train=False)[0])
        rep_dir = root_path(args.checkpoint_root) / mode / "representation"
        rep = train_representation(
            train,
            validation,
            cfg,
            replace(TrainConfig(), batch_size=2, microbatch_size=1, max_epochs=20),
            rep_dir,
            max_steps=args.max_steps,
            log_dir=output / mode / "representation",
        )
        params, metadata = load_checkpoint(rep["last_checkpoint"])
        final_loss = float(model.apply({"params": params}, batch, train=False)[0])
        frozen_before = parameter_hash(params)
        latent_root = root_path(args.data_root) / "latents_smoke" / mode
        train_latents = export_frozen_latents(train, rep["best_checkpoint"], latent_root / "train", batch_size=2)
        val_latents = export_frozen_latents(
            validation, rep["best_checkpoint"], latent_root / "validation", batch_size=2
        )
        probe_cfg = replace(ProbeConfig(), batch_size=4, max_epochs=20)
        probe = train_action_probe(
            train_latents,
            val_latents,
            probe_cfg,
            root_path(args.checkpoint_root) / mode / "probe",
            max_steps=args.max_steps,
            log_dir=output / mode / "probe",
        )
        if parameter_hash(params) != frozen_before:
            raise AssertionError("Representation mutated during probe training")
        # Verify supervised optimizer movement on the same cached, fixed train query.
        fixed_probe = collate([train_latents[0]], PROBE_FIELDS)
        probe_model = ActionProbe(probe_cfg)
        initial_probe_variables = probe_model.init(
            jax.random.PRNGKey(0),
            fixed_probe["z"],
            fixed_probe["proprioception"],
            fixed_probe["visual_age_seconds"],
            train=False,
        )

        def fixed_probe_loss(variables, _probe=probe_model, _batch=fixed_probe):
            predicted = _probe.apply(
                variables, _batch["z"], _batch["proprioception"], _batch["visual_age_seconds"], train=False
            )
            return float(probe_loss(predicted, _batch["action_target"], _batch["action_valid_mask"]))

        initial_probe_loss = fixed_probe_loss(initial_probe_variables)
        last_probe_params, _ = load_checkpoint(probe["last_checkpoint"])
        final_probe_loss = fixed_probe_loss({"params": last_probe_params})
        # Latency includes an explicit compile/first call and synchronized steady state.
        encode = jax.jit(
            lambda b, _model=model, _params=params: _model.apply(
                {"params": _params}, b, method=_model.encode, train=False
            )
        )
        start = time.perf_counter()
        compiled_encode = encode.lower(batch).compile()
        encode_compile = time.perf_counter() - start
        start = time.perf_counter()
        z = compiled_encode(batch).block_until_ready()
        encode_first_execution = time.perf_counter() - start
        first_call = encode_compile + encode_first_execution
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            compiled_encode(batch).block_until_ready()
            timings.append(time.perf_counter() - start)
        probe_params, _ = load_checkpoint(probe["best_checkpoint"])
        action_probe = ActionProbe(probe_cfg)
        probe_forward = jax.jit(
            lambda z, s, age, _probe=action_probe, _params=probe_params: _probe.apply(
                {"params": _params}, z, s, age, train=False
            )
        )
        age = batch["query_timestamp"] - batch["visual_timestamp"]
        start = time.perf_counter()
        compiled_probe = probe_forward.lower(z, batch["proprioception"], age).compile()
        probe_compile = time.perf_counter() - start
        start = time.perf_counter()
        actions = compiled_probe(z, batch["proprioception"], age).block_until_ready()
        probe_first_execution = time.perf_counter() - start
        start = time.perf_counter()
        compiled_probe(z, batch["proprioception"], age).block_until_ready()
        probe_latency = time.perf_counter() - start
        fusion_latency = None
        if mode != "rlt_original":
            fusion = Fusion(cfg)
            nv = batch["visual_features"].shape[1]
            nu = cfg.history_frames * cfg.num_sensors + 1
            v, u = jnp.zeros((1, nv, cfg.embed_dim)), jnp.zeros((1, nu, cfg.embed_dim))
            vm, um = batch["visual_valid_mask"], jnp.ones((1, nu), bool)
            fp = {"params": params["encoder"]["fusion"]}
            ff = jax.jit(lambda v, u, vm, um, _fusion=fusion, _fp=fp: _fusion.apply(_fp, v, u, vm, um)[0])
            ff(v, u, vm, um).block_until_ready()
            start = time.perf_counter()
            ff(v, u, vm, um).block_until_ready()
            fusion_latency = time.perf_counter() - start
        results["models"][mode] = {
            "status": "pass" if final_loss < initial_loss else "fail_loss_decrease",
            "config": asdict(cfg),
            "initial_fixed_batch_loss": initial_loss,
            "initial_fixed_probe_loss": initial_probe_loss,
            "final_fixed_probe_loss": final_probe_loss,
            "fixed_probe_loss_decreased": final_probe_loss < initial_probe_loss,
            "final_fixed_batch_loss": final_loss,
            "fixed_batch_loss_decreased": final_loss < initial_loss,
            "representation_steps": rep["manifest"]["actual_train_steps"],
            "probe_steps": probe["manifest"]["actual_train_steps"],
            "latent_shape": list(z.shape),
            "probe_shape": list(actions.shape),
            "parameter_count": sum(x.size for x in jax.tree.leaves(params)),
            "parameter_devices": sorted({str(x.device) for x in jax.tree.leaves(params)}),
            "latent_device": str(z.device),
            "input_devices": sorted({str(jax.device_put(x).device) for x in batch.values()}),
            "input_shapes": {key: list(value.shape) for key, value in batch.items()},
            "dtype": str(z.dtype),
            "first_encode_seconds_including_compile": first_call,
            "encode_compile_seconds": encode_compile,
            "encode_first_execution_seconds": encode_first_execution,
            "cnn_fusion_seconds_median": float(np.median(timings)),
            "fusion_only_seconds": fusion_latency,
            "probe_first_seconds_including_compile": probe_compile + probe_first_execution,
            "probe_compile_seconds": probe_compile,
            "probe_first_execution_seconds": probe_first_execution,
            "probe_seconds": probe_latency,
            "representation_unchanged": True,
            "checkpoint_step": metadata["step"],
        }
        write_json(output / "report.json", results)
        del variables, params, model
        jax.clear_caches()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
