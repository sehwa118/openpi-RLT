"""Two-stage offline training. No action-expert or online-RL dependency.

Model computation is JAX/Linen; host batches use bounded CPU worker prefetch. Checkpoints include optimizer state and deterministic shuffle position.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from flax import linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from openpi.trlt.config import ModelConfig
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import TrainConfig
from openpi.trlt.config import digest
from openpi.trlt.config import root_path
from openpi.trlt.loading import OrderedBatchSampler
from openpi.trlt.loading import TrainingCollator
from openpi.trlt.loading import training_loader
from openpi.trlt.metrics import ActionMetrics
from openpi.trlt.tracking import TrainingTracker

OBSERVATION_FIELDS = (
    "visual_features",
    "visual_valid_mask",
    "tactile",
    "tactile_valid_mask",
    "spatial_valid_mask",
    "tactile_timestamps",
    "proprioception",
    "query_timestamp",
    "visual_timestamp",
)
PROBE_FIELDS = ("z", "proprioception", "visual_age_seconds", "action_target", "action_valid_mask")


def _progress(message):
    # Keep CLI stdout available for its final JSON result, including when piped.
    tqdm.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr)


SHARED_PROVENANCE_KEYS = (
    "revision",
    "dataset_revision",
    "split_hash",
    "split_manifest_hash",
    "query_manifest_hash",
    "normalization_hash",
    "feature_cache_hash",
    "representation_checkpoint_hash",
    "temporal",
    "subset_hash",
    "episode_manifest_hash",
    "caption_hash",
    "tactile_resolution",
    "camera_order",
    "sensor_order",
    "matmul_precision",
    "model_config",
    "model",
    "representation_seed",
)


def _check_provenance(expected, actual, label):
    for key in SHARED_PROVENANCE_KEYS:
        value = expected.get(key)
        if value is not None and digest(actual.get(key)) != digest(value):
            raise ValueError(f"{label} provenance mismatch: {key}")


def _validated_normalization(value, expected_hash=None):
    normalization = json.loads(root_path(value).read_text()) if isinstance(value, str | Path) else value
    if not isinstance(normalization, dict) or normalization.get("hash") != digest(
        {key: item for key, item in normalization.items() if key != "hash"}
    ):
        raise ValueError("Normalization content hash mismatch; edited statistics cannot be used")
    if expected_hash is not None and normalization["hash"] != expected_hash:
        raise ValueError("Evaluation normalization hash mismatch")
    return normalization


def _json(value):
    if isinstance(value, jax.Array | np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json, allow_nan=False))
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parameter_hash(params):
    result = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.asarray(value)
        result.update(str(path).encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def _atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def save_checkpoint(directory, state, metadata):
    """Write params and optimizer together; JSON is committed last."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    encoded = serialization.to_bytes(state)
    params = serialization.to_bytes(state.params)
    _atomic_bytes(directory / "state.msgpack", encoded)
    _atomic_bytes(directory / "params.msgpack", params)
    metadata = dict(
        metadata,
        parameter_hash=parameter_hash(state.params),
        state_hash=hashlib.sha256(encoded).hexdigest(),
        params_file_hash=hashlib.sha256(params).hexdigest(),
    )
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_checkpoint(directory, template=None):
    directory = root_path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if file_hash(directory / "params.msgpack") != metadata["params_file_hash"]:
        raise ValueError("Checkpoint parameters failed integrity check")
    if template is None:
        params = serialization.msgpack_restore((directory / "params.msgpack").read_bytes())
        params = jax.tree_util.tree_map(jnp.asarray, params)
        if parameter_hash(params) != metadata["parameter_hash"]:
            raise ValueError("Checkpoint parameter hash mismatch")
        return params, metadata
    if file_hash(directory / "state.msgpack") != metadata["state_hash"]:
        raise ValueError("Checkpoint training state failed integrity check")
    restored = serialization.from_bytes(template, (directory / "state.msgpack").read_bytes())
    # Flax msgpack restores NumPy arrays; place params AND optimizer buffers on
    # the selected device before reporting placement or resuming a compiled step.
    return jax.device_put(restored, jax.devices()[0]), metadata


def environment_manifest():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root_path("."), capture_output=True, text=True, check=True)
        return result.stdout.strip()

    devices = jax.devices()
    try:
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "jax_version": jax.__version__,
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "python": platform.python_version(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }


def assert_gpu():
    # Default GPU GEMM precision can depend on batch shape (notably TF32 on
    # Blackwell); keep accumulation and all comparison modes on full float32.
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != "gpu" or not all(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU validation/training requires a working JAX GPU backend; CPU fallback is forbidden")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly")
    return environment_manifest()


def collate(samples, fields):
    if not samples:
        raise ValueError("Empty batch")
    required = set(fields) - {"spatial_valid_mask"}
    if any(not required.issubset(sample) for sample in samples):
        raise ValueError(f"Missing tensor field(s); expected {sorted(required)}")
    output = {}
    for key in fields:
        if key in samples[0]:
            if any(key not in sample for sample in samples):
                raise ValueError(f"Mixed presence of optional field {key}")
            output[key] = np.stack([sample[key] for sample in samples])
    return output


def _counts(samples, probe):
    if probe:
        return {
            "action": sum(
                int(np.asarray(row["action_valid_mask"], bool).sum()) * row["action_target"].shape[-1]
                for row in samples
            )
        }
    visual = sum(
        int(np.asarray(row["visual_valid_mask"], bool).sum()) * row["visual_features"].shape[-1] for row in samples
    )
    tactile = 0
    for row in samples:
        valid = np.asarray(row["tactile_valid_mask"], bool)
        if "spatial_valid_mask" in row:
            pixel = np.broadcast_to(np.asarray(row["spatial_valid_mask"], bool), row["tactile"].shape)
            valid = valid & pixel.any(axis=(-3, -2, -1))
        tactile += int(valid.sum())
    return {"visual": visual, "tactile": tactile}


def _tree_all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)]))


def _optimizer(config, clip=1.0):
    return optax.chain(
        optax.clip_by_global_norm(clip), optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    )


def _mean_metrics(rows, counts):
    output = {}
    denominator_keys = {
        "tactile_per_sensor": "tactile_sensor_count",
        "tactile_per_history": "tactile_history_count",
        "tactile_latest": "tactile_latest_count",
        "tactile_history": "tactile_past_count",
        "tactile_valid_fraction": "tactile_total_count",
        "visual_valid_fraction": "visual_total_count",
    }
    for key in rows[0] if rows else ():
        if key.endswith("_per_sample") or key == "total_loss":
            continue
        if key.endswith("_count"):
            output[key] = sum(np.asarray(row[key]) for row in rows)
            continue
        denominator_key = denominator_keys.get(key)
        if denominator_key and denominator_key in rows[0]:
            weights = [np.asarray(row[denominator_key]) for row in rows]
        else:
            weights = [
                count["tactile"]
                if key.startswith("tactile") and "tactile" in count
                else count.get("visual", count.get("action", 1))
                for count in counts
            ]
        total = sum(weights)
        output[key] = sum(
            np.asarray(row[key]) * weight for row, weight in zip(rows, weights, strict=True)
        ) / np.maximum(total, 1)
    if rows and "total_loss" in rows[0]:
        output["total_loss"] = (
            output.get("visual_weighted", output["visual_loss"]) + output["tactile_weighted"]
            if "tactile_weighted" in output
            else output["visual_loss"]
        )
    return output


class ActionProbe(nn.Module):
    config: ProbeConfig

    @nn.compact
    def __call__(self, z, proprioception, visual_age_seconds, *, train=False):
        z = jax.lax.stop_gradient(z)
        age = visual_age_seconds.reshape((z.shape[0], 1))
        x = jnp.concatenate((z.reshape((z.shape[0], -1)), proprioception, age), axis=-1)
        for index in range(self.config.depth):
            x = nn.Dense(self.config.hidden_dim, name=f"hidden_{index}")(x)
            x = nn.gelu(x)
            x = nn.Dropout(self.config.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.config.horizon * self.config.action_dim, name="output")(x)
        return x.reshape((x.shape[0], self.config.horizon, self.config.action_dim))


def probe_loss(prediction, target, valid_mask):
    valid = valid_mask[..., None].astype(bool)
    error = jnp.where(valid, prediction - jax.lax.stop_gradient(target), 0)
    return jnp.sum(error**2) / jnp.maximum(jnp.sum(valid) * target.shape[-1], 1)


def representation_objective(model, params, batch, denominators, rng, config):
    """Contribution of one microbatch to the effective-batch reconstruction mean."""
    _, info = model.apply({"params": params}, batch, train=True, rngs={"dropout": rng})
    weighted = info["visual_loss"] * info["visual_count"] / jnp.maximum(denominators[0], 1)
    if config.mode != "rlt_original":
        weighted = config.lambda_visual * weighted + config.lambda_tactile * info["tactile_loss"] * info[
            "tactile_count"
        ] / jnp.maximum(denominators[1], 1)
    return weighted, info


def _compatible(train_dataset, validation_dataset):
    a, b = getattr(train_dataset, "provenance", {}), getattr(validation_dataset, "provenance", {})
    if any("normalization_hash" in source and not source["normalization_hash"] for source in (a, b)):
        raise ValueError("Training requires saved train-only state/action normalization")
    _check_provenance(a, b, "Train/validation")
    _check_provenance(b, a, "Train/validation")
    if a.get("split") not in (None, "train") or b.get("split") not in (None, "validation"):
        raise ValueError("Training must use train and validation splits only")


def _train(
    train_dataset,
    validation_dataset,
    model_config,
    config,
    output_dir,
    *,
    seed,
    max_steps,
    resume,
    provenance,
    log_dir,
    probe,
    tracking=None,
    progress=True,
    num_workers=2,
    prefetch_factor=2,
):
    jax.config.update("jax_default_matmul_precision", "highest")
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be nonnegative and prefetch_factor must be positive")
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Both train and validation datasets must be nonempty")
    _compatible(train_dataset, validation_dataset)
    if max_steps is None:
        max_steps = getattr(config, "max_steps", None)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.max_epochs is not None and config.max_epochs <= 0:
        raise ValueError("max_epochs must be positive or None")
    if config.max_epochs is None and max_steps is None:
        raise ValueError("Training requires max_steps or max_epochs")
    if config.patience is not None and config.patience <= 0:
        raise ValueError("patience must be positive or None to disable early stopping")
    output = root_path(output_dir)
    log_dir = root_path(log_dir or f"experiments/trlt/{'probe' if probe else 'representation'}/{output.name}")
    if (output / "last" / "metadata.json").exists() and not resume:
        raise FileExistsError("Output checkpoint exists; use resume or another output directory")
    stage = "probe" if probe else f"representation/{model_config.mode}"
    _progress(f"{stage}: train={len(train_dataset)} validation={len(validation_dataset)} seed={seed}")
    _progress("Loading initial sample (uncached tactile episodes require CPU video decoding).")
    fields = PROBE_FIELDS if probe else OBSERVATION_FIELDS
    first = collate([train_dataset[0]], fields)
    _progress("Initializing model and optimizer on the selected device.")
    if probe:
        model = ActionProbe(config)
        variables = model.init(
            jax.random.PRNGKey(seed), first["z"], first["proprioception"], first["visual_age_seconds"], train=False
        )
    else:
        from openpi.models.trlt import build_model

        model = build_model(model_config)
        variables = model.init(jax.random.PRNGKey(seed), first, train=False)
    if set(variables) != {"params"}:
        raise ValueError("Unexpected mutable model state; frozen/export contract requires explicit handling")
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=_optimizer(config, getattr(config, "gradient_clip", 1.0))
    )
    resolved_provenance = dict(getattr(train_dataset, "provenance", {}), **(provenance or {}))
    resolved_provenance.pop("split", None)
    identity = {
        "stage": "probe" if probe else "representation",
        "matmul_precision": "highest",
        "schema_version": 1,
        "normalization": getattr(train_dataset, "normalization", None),
        "batch_schema": {
            key: {"shape": list(value.shape[1:]), "dtype": str(value.dtype)} for key, value in first.items()
        },
        "seed": seed,
        "model_config": asdict(model_config) if model_config else None,
        # Preserve legacy checkpoint identity when no configured step limit exists.
        "train_config": {
            key: value for key, value in asdict(config).items() if key != "max_steps" or value is not None
        },
        "provenance": resolved_provenance,
    }
    identity_hash = digest(identity)
    epoch, cursor, best, bad_epochs = 0, 0, float("inf"), 0
    if resume:
        state, previous = load_checkpoint(output / "last", state)
        if previous["identity_hash"] != identity_hash:
            raise ValueError("Resume config/provenance mismatch")
        epoch, cursor = previous["epoch"], previous["cursor"]
        best, bad_epochs = previous["best_validation"], previous["bad_epochs"]
    manifest = dict(
        identity,
        identity_hash=identity_hash,
        environment=environment_manifest(),
        parameter_count=sum(value.size for value in jax.tree_util.tree_leaves(state.params)),
        parameter_dtypes=sorted({str(value.dtype) for value in jax.tree_util.tree_leaves(state.params)}),
        parameter_devices=sorted(
            {str(device) for value in jax.tree_util.tree_leaves(state.params) for device in value.devices()}
        ),
        input_shapes={key: list(value.shape) for key, value in first.items()},
        input_dtypes={key: str(value.dtype) for key, value in first.items()},
        effective_batch_size=config.batch_size,
        microbatch_size=config.batch_size if probe else config.microbatch_size,
        validation_selection="normalized_action_micro_mse"
        if probe
        else "visual_loss"
        if model_config.mode == "rlt_original"
        else "weighted_total_reconstruction_loss",
        data_loading={"num_workers": num_workers, "prefetch_factor": prefetch_factor, "backend": "torch_spawn"},
        actual_train_steps=int(state.step),
        max_steps=max_steps,
        started_at=time.time(),
    )
    write_json(log_dir / "manifest.json", manifest)
    _progress(
        f"Ready: devices={manifest['parameter_devices']} parameters={manifest['parameter_count']:,} "
        f"batch={config.batch_size} microbatch={manifest['microbatch_size']} "
        f"step={int(state.step)} max_steps={max_steps}; logs={log_dir}"
    )
    if config.batch_size <= 0 or (
        not probe and (config.microbatch_size <= 0 or config.batch_size % config.microbatch_size)
    ):
        raise ValueError("Effective batch size must be a positive multiple of microbatch size")
    micro_size = config.batch_size if probe else config.microbatch_size

    def objective(params, batch, denominators, rng):
        if probe:
            predicted = model.apply(
                {"params": params},
                batch["z"],
                batch["proprioception"],
                batch["visual_age_seconds"],
                train=True,
                rngs={"dropout": rng},
            )
            loss = probe_loss(predicted, batch["action_target"], batch["action_valid_mask"])
            count = jnp.sum(batch["action_valid_mask"]) * config.action_dim
            return loss * count / jnp.maximum(denominators[0], 1), {"action_loss": loss, "action_count": count}
        return representation_objective(model, params, batch, denominators, rng, model_config)

    @jax.jit
    def train_step(current, microbatches, denominator):
        def gradient(part, batch):
            rng = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), current.step), part)
            (value, info), grads = jax.value_and_grad(objective, has_aux=True)(current.params, batch, denominator, rng)
            return grads, value, info, jnp.isfinite(value) & _tree_all_finite(grads)

        grads, loss_sum, info, finite = gradient(0, microbatches[0])
        infos = [info]

        def accumulate(carry, inputs):
            accumulated, total, valid = carry
            part, batch = inputs
            more, value, info, ok = gradient(part, batch)
            return (jax.tree.map(jnp.add, accumulated, more), total + value, valid & ok), info

        # Scan keeps accumulation sequential and compiles a single gradient body,
        # rather than unrolling up to eight copies with concurrent activations.
        # A smaller final microbatch is compiled separately, without padding.
        count = len(microbatches)
        same_shape_end = count
        if count > 1 and jax.tree.leaves(microbatches[-1])[0].shape[0] != jax.tree.leaves(microbatches[0])[0].shape[0]:
            same_shape_end -= 1
        if same_shape_end > 1:
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *microbatches[1:same_shape_end])
            (grads, loss_sum, finite), scanned = jax.lax.scan(
                accumulate, (grads, loss_sum, finite), (jnp.arange(1, same_shape_end), stacked)
            )
            infos.extend(jax.tree.map(lambda x, i=i: x[i], scanned) for i in range(same_shape_end - 1))
        if same_shape_end < count:
            (grads, loss_sum, finite), info = accumulate((grads, loss_sum, finite), (count - 1, microbatches[-1]))
            infos.append(info)
        finite = finite & _tree_all_finite(grads) & jnp.isfinite(loss_sum)
        # Keep invalid gradients out of the optimizer, without copying the tree
        # to the host. Only the small diagnostics and this boolean leave JAX.
        updated = jax.lax.cond(finite, lambda s: s.apply_gradients(grads=grads), lambda s: s, current)
        return updated, loss_sum, infos, finite

    @jax.jit
    def validate(params, batch):
        if probe:
            predicted = model.apply(
                {"params": params}, batch["z"], batch["proprioception"], batch["visual_age_seconds"], train=False
            )
            return {"action_loss": probe_loss(predicted, batch["action_target"], batch["action_valid_mask"])}
        return model.apply({"params": params}, batch, train=False)[1]

    def validation():
        infos, weights = [], []
        batches = (len(validation_dataset) + micro_size - 1) // micro_size
        _progress(f"Validation started: {batches} batches; cold tactile caches may require CPU decoding.")
        validation_sampler.indices = list(range(len(validation_dataset)))
        for batch_number, (parts, _, counts, _) in enumerate(
            tqdm(validation_batches, total=batches, desc="Validation", unit="batch", disable=not progress, leave=False),
            1,
        ):
            infos.append(jax.device_get(validate(state.params, parts[0])))
            weights.append(counts[0])
            if batch_number == 1 or batch_number % 50 == 0 or batch_number == batches:
                _progress(f"Validation batch {batch_number}/{batches}")
        info = _mean_metrics(infos, weights)
        if probe:
            value = info["action_loss"]
        elif model_config.mode == "rlt_original":
            value = info["visual_loss"]
        else:
            value = (
                model_config.lambda_visual * info["visual_loss"] + model_config.lambda_tactile * info["tactile_loss"]
            )
        return float(value), info

    def checkpoint_metadata():
        return dict(
            identity,
            identity_hash=identity_hash,
            epoch=epoch,
            cursor=cursor,
            best_validation=best,
            bad_epochs=bad_epochs,
            step=int(state.step),
            rng_protocol="PRNGKey(seed), fold_in(step), fold_in(microbatch); epoch shuffle default_rng(SeedSequence([seed,epoch]))",
        )

    start_time = time.perf_counter()
    log_dir.mkdir(parents=True, exist_ok=True)
    first_update = True
    steps_per_epoch = (len(train_dataset) + config.batch_size - 1) // config.batch_size
    total_steps = steps_per_epoch * config.max_epochs if config.max_epochs is not None else max_steps
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    train_sampler = OrderedBatchSampler(config.batch_size)
    validation_sampler = OrderedBatchSampler(micro_size)
    with ExitStack() as stack:
        train_batches = stack.enter_context(
            training_loader(
                train_dataset,
                train_sampler,
                TrainingCollator(fields, micro_size, probe),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                seed=seed,
            )
        )
        validation_batches = stack.enter_context(
            training_loader(
                validation_dataset,
                validation_sampler,
                TrainingCollator(fields, micro_size, probe),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                seed=seed,
            )
        )
        tracker = stack.enter_context(TrainingTracker(log_dir, manifest, options=tracking, resume=resume))
        tracker.bar = tqdm(
            total=total_steps,
            initial=int(state.step),
            desc=stage,
            unit="step",
            disable=not progress,
        )
        while (config.max_epochs is None or epoch < config.max_epochs) and (
            config.patience is None or bad_epochs < config.patience
        ):
            if max_steps is not None and int(state.step) >= max_steps:
                break
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(train_dataset))
            # Limit dispatch to the requested steps; prefetched samples never
            # advance cursor or enter a checkpoint until an update succeeds.
            end = (
                len(order)
                if max_steps is None
                else min(len(order), cursor + (max_steps - int(state.step)) * config.batch_size)
            )
            train_sampler.indices = order[cursor:end].tolist()
            batch_iterator = iter(train_batches)
            while cursor < end:
                if max_steps is not None and int(state.step) >= max_steps:
                    break
                batch_started = time.perf_counter()
                if first_update:
                    _progress(f"Loading first training batch; CPU DataLoader workers={num_workers}.")
                parts, denominator, weights, examples = next(batch_iterator)
                data_seconds = time.perf_counter() - batch_started
                if first_update:
                    _progress(
                        f"Batch loaded in {data_seconds:.2f}s; first gradient/update includes JAX JIT compilation."
                    )
                update_started = time.perf_counter()
                state, loss_sum, infos, finite = train_step(state, parts, denominator)
                loss_sum, infos, finite = jax.device_get((loss_sum, infos, finite))
                if not finite:
                    raise FloatingPointError("Non-finite reconstruction/probe gradient; no optimizer update saved")
                jax.block_until_ready(state.params)
                loss_sum = float(loss_sum)
                cursor += examples
                event = {
                    "step": int(state.step),
                    "epoch": epoch,
                    "loss": loss_sum,
                    "examples": examples,
                    "metrics": _mean_metrics(infos, weights),
                    "data_seconds": data_seconds,
                    "update_seconds": time.perf_counter() - update_started,
                }
                with (log_dir / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(event, default=_json, allow_nan=False) + "\n")
                tracker.log("train", event)
                tracker.bar.set_postfix(
                    loss=f"{loss_sum:.5g}",
                    epoch=epoch + 1,
                    data_s=f"{data_seconds:.2f}",
                    update_s=f"{event['update_seconds']:.2f}",
                )
                tracker.bar.update(1)
                if first_update or event["step"] % 10 == 0 or cursor >= len(order):
                    _progress(
                        f"epoch={epoch + 1}/{config.max_epochs if config.max_epochs is not None else 'unbounded'} "
                        f"step={event['step']} "
                        f"samples={cursor}/{len(order)} loss={loss_sum:.6g} "
                        f"data={data_seconds:.2f}s update={event['update_seconds']:.2f}s"
                    )
                first_update = False
            # A debug interruption receives a validation checkpoint but preserves cursor
            # so resume continues exactly at the next unconsumed training sample.
            completed_epoch = cursor >= len(order)
            validation_value, validation_info = validation()
            if not np.isfinite(validation_value):
                raise FloatingPointError("Non-finite validation objective")
            improved = validation_value < best
            if improved:
                best, bad_epochs = validation_value, 0
            elif completed_epoch:
                bad_epochs += 1
            if completed_epoch:
                epoch, cursor = epoch + 1, 0
            metadata = dict(checkpoint_metadata(), validation=validation_info)
            if improved:
                save_checkpoint(output / "best", state, metadata)
            save_checkpoint(output / "last", state, metadata)
            _progress(
                f"Validation complete: step={int(state.step)} objective={validation_value:.6g} "
                f"best={improved}; saved checkpoints to {output}"
            )
            with (log_dir / "validation.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": int(state.step),
                            "epoch": epoch,
                            "value": validation_value,
                            "best": improved,
                            "metrics": validation_info,
                        },
                        default=_json,
                        allow_nan=False,
                    )
                    + "\n"
                )
            tracker.log(
                "validation",
                {
                    "step": int(state.step),
                    "epoch": epoch,
                    "value": validation_value,
                    "best_value": best,
                    "improved": improved,
                    "metrics": validation_info,
                },
            )
            if max_steps is not None and int(state.step) >= max_steps:
                break
        manifest.update(
            actual_train_steps=int(state.step),
            completed_epochs=epoch,
            cursor=cursor,
            best_validation=best if np.isfinite(best) else None,
            elapsed_seconds=time.perf_counter() - start_time,
            stopped_by_max_steps=max_steps is not None and int(state.step) >= max_steps,
        )
        write_json(log_dir / "manifest.json", manifest)
        _progress(f"{stage} finished: steps={int(state.step)} elapsed={manifest['elapsed_seconds']:.1f}s")
        return {"best_checkpoint": str(output / "best"), "last_checkpoint": str(output / "last"), "manifest": manifest}


def train_representation(
    train_dataset,
    validation_dataset,
    model_config: ModelConfig,
    config: TrainConfig,
    output_dir,
    *,
    seed=0,
    max_steps=None,
    resume=False,
    provenance=None,
    log_dir=None,
    tracking=None,
    progress=True,
    num_workers=2,
    prefetch_factor=2,
):
    return _train(
        train_dataset,
        validation_dataset,
        model_config,
        config,
        output_dir,
        seed=seed,
        max_steps=max_steps,
        resume=resume,
        provenance=provenance,
        log_dir=log_dir,
        probe=False,
        tracking=tracking,
        progress=progress,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


class LatentDataset:
    """Random access to bounded NPZ shards; cache at most two decoded shards."""

    def __init__(self, directory, expected_metadata=None):
        self.root = root_path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.provenance = self.manifest["provenance"]
        self.normalization = self.manifest.get("normalization")
        if expected_metadata and any(self.provenance.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("Stale latent cache metadata")
        if (
            digest({key: value for key, value in self.manifest.items() if key != "cache_hash"})
            != self.manifest["cache_hash"]
        ):
            raise ValueError("Latent manifest integrity failure")
        self.cache_hash = self.manifest["cache_hash"]
        self.provenance = dict(self.provenance, latent_cache_hash=self.cache_hash)
        self.ends = np.cumsum([shard["count"] for shard in self.manifest["shards"]])
        self._cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.ends, index, side="right"))
        if shard_index not in self._cache:
            entry = self.manifest["shards"][shard_index]
            if file_hash(self.root / entry["file"]) != entry["sha256"]:
                raise ValueError("Corrupted latent shard")
            with np.load(self.root / entry["file"], allow_pickle=False) as values:
                self._cache[shard_index] = {key: values[key] for key in values.files}
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(shard_index)
        local = index - (self.ends[shard_index - 1] if shard_index else 0)
        return {key: value[local] for key, value in self._cache[shard_index].items()}


def export_frozen_latents(
    dataset, representation_checkpoint, output_dir, *, batch_size=32, force_cache_rebuild=False, max_samples=None
):
    from openpi.models.trlt import build_model

    jax.config.update("jax_default_matmul_precision", "highest")
    params, metadata = load_checkpoint(representation_checkpoint)
    if metadata["stage"] != "representation":
        raise ValueError("Expected representation checkpoint")
    source = getattr(dataset, "provenance", {})
    _check_provenance(metadata["provenance"], source, "Representation/cache")
    if batch_size <= 0 or (max_samples is not None and max_samples <= 0):
        raise ValueError("batch_size/max_samples must be positive")
    config = ModelConfig(**metadata["model_config"])
    model = build_model(config)
    provenance = dict(
        getattr(dataset, "provenance", {}),
        representation_checkpoint_hash=metadata["parameter_hash"],
        representation_checkpoint_file_hash=metadata["params_file_hash"],
        model=config.mode,
        representation_seed=metadata["seed"],
        model_config=metadata["model_config"],
        frozen=True,
        matmul_precision="highest",
    )
    limit = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    provenance["max_samples"] = max_samples
    fingerprint = digest(provenance)
    output = root_path(output_dir)
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if not force_cache_rebuild:
            if previous["fingerprint"] != fingerprint:
                raise ValueError("Stale latent cache; explicitly set force_cache_rebuild")
            return LatentDataset(output)
    output.mkdir(parents=True, exist_ok=True)
    before = parameter_hash(params)
    encode = jax.jit(lambda batch: model.apply({"params": params}, batch, train=False, method=model.encode))
    shards = []
    for start in range(0, limit, batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, limit))]
        z = np.asarray(encode(collate(samples, OBSERVATION_FIELDS)))
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("Non-finite frozen representation")
        values = {"z": z}
        for key in (
            "proprioception",
            "action_target",
            "action_valid_mask",
            "episode_id",
            "anchor_frame_index",
            "query_frame_index",
            "query_timestamp",
            "object",
        ):
            values[key] = np.stack([sample[key] for sample in samples])
        values["visual_age_seconds"] = np.asarray(
            [sample["query_timestamp"] - sample["visual_timestamp"] for sample in samples], np.float32
        )
        filename = f"chunk_{len(shards):06d}.npz"
        with (output / (filename + ".tmp")).open("wb") as handle:
            np.savez_compressed(handle, **values)
        (output / (filename + ".tmp")).replace(output / filename)
        shards.append({"file": filename, "count": len(samples), "sha256": file_hash(output / filename)})
    if parameter_hash(params) != before:
        raise AssertionError("Representation changed during frozen export")
    manifest = {
        "version": 1,
        "provenance": provenance,
        "fingerprint": fingerprint,
        "shards": shards,
        "count": limit,
        "representation_unchanged": True,
        "normalization": metadata.get("normalization"),
    }
    manifest["cache_hash"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return LatentDataset(output)


def train_action_probe(
    train_latents,
    validation_latents,
    config: ProbeConfig,
    output_dir,
    *,
    seed=0,
    max_steps=None,
    resume=False,
    provenance=None,
    log_dir=None,
    tracking=None,
    progress=True,
    num_workers=2,
    prefetch_factor=2,
):
    for dataset in (train_latents, validation_latents):
        if not getattr(dataset, "provenance", {}).get("frozen"):
            raise ValueError("Probe requires frozen latent caches")
    provenance = dict(
        provenance or {},
        train_latent_cache_hash=train_latents.cache_hash,
        validation_latent_cache_hash=validation_latents.cache_hash,
    )
    # Representation parameters and mutable state are absent from this optimizer.
    return _train(
        train_latents,
        validation_latents,
        None,
        config,
        output_dir,
        seed=seed,
        max_steps=max_steps,
        resume=resume,
        provenance=provenance,
        log_dir=log_dir,
        probe=True,
        tracking=tracking,
        progress=progress,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def evaluate_action_probe(latents, probe_checkpoint, normalization, output_dir, *, batch_size=256):
    jax.config.update("jax_default_matmul_precision", "highest")
    params, metadata = load_checkpoint(probe_checkpoint)
    if metadata["stage"] != "probe":
        raise ValueError("Expected probe checkpoint")
    provenance = getattr(latents, "provenance", {})
    _check_provenance(metadata["provenance"], provenance, "Evaluation cache/checkpoint")
    normalization = _validated_normalization(normalization, provenance.get("normalization_hash"))
    config = ProbeConfig(**metadata["train_config"])
    model = ActionProbe(config)
    predict = jax.jit(
        lambda batch: model.apply(
            {"params": params}, batch["z"], batch["proprioception"], batch["visual_age_seconds"], train=False
        )
    )
    metrics = ActionMetrics(normalization["action"]["mean"], normalization["action"]["std"])
    output = root_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_hash = parameter_hash(params)
    with gzip.open(output / "predictions.jsonl.gz", "wt") as handle:
        for start in range(0, len(latents), batch_size):
            samples = [latents[index] for index in range(start, min(start + batch_size, len(latents)))]
            batch = collate(samples, PROBE_FIELDS)
            prediction = np.asarray(predict(batch))
            metrics.update(
                prediction,
                batch["action_target"],
                batch["action_valid_mask"],
                [sample["episode_id"] for sample in samples],
                [sample["object"] for sample in samples],
            )
            raw_prediction = prediction * metrics.std + metrics.mean
            raw_target = batch["action_target"] * metrics.std + metrics.mean
            for i, sample in enumerate(samples):
                for offset in range(config.horizon):
                    row = {
                        "model": provenance.get("model"),
                        "seed": metadata["seed"],
                        "episode_id": int(sample["episode_id"]),
                        "object": str(sample["object"]),
                        "query_frame_index": int(sample["query_frame_index"]),
                        "query_timestamp": float(sample["query_timestamp"]),
                        "anchor_frame_index": int(sample["anchor_frame_index"]),
                        "horizon_offset": offset,
                        "valid": bool(batch["action_valid_mask"][i, offset]),
                        "gt_action": raw_target[i, offset],
                        "predicted_action": raw_prediction[i, offset],
                        "per_joint_squared_error": (raw_prediction[i, offset] - raw_target[i, offset]) ** 2,
                    }
                    handle.write(json.dumps(row, default=_json, allow_nan=False) + "\n")
    if parameter_hash(params) != original_hash:
        raise AssertionError("Probe mutated during evaluation")
    report = {
        "model": provenance.get("model"),
        "seed": metadata["seed"],
        "metrics": metrics.result(),
        "provenance": provenance,
        "probe_checkpoint": str(probe_checkpoint),
        "probe_checkpoint_hash": original_hash,
        "environment": environment_manifest(),
    }
    write_json(output / "metrics.json", report)
    return report


class SyntheticDataset:
    """Deterministic train/validation fixture; never represents source test data."""

    def __init__(self, config, *, split="train", size=4, seed=0):
        if split not in ("train", "validation"):
            raise ValueError("Synthetic debug fixture permits train/validation only")
        self.config, self.split, self.size, self.seed = config, split, size, seed
        self.normalization = {
            "state": {"mean": [0.0] * 58, "std": [1.0] * 58},
            "action": {"mean": [0.0] * 58, "std": [1.0] * 58},
            "split_hash": "synthetic-v1",
        }
        self.normalization["hash"] = digest(self.normalization)
        self.provenance = {
            "split": split,
            "dataset_revision": "synthetic",
            "split_manifest_hash": "synthetic-v1",
            "normalization_hash": self.normalization["hash"],
            "temporal": asdict(TemporalConfig()),
        }

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        c = self.config
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        q = np.float32(index / 30)
        history = c.history_frames
        tactile_valid = np.ones((history, c.num_sensors), bool)
        tactile_valid[: max(history - index - 1, 0)] = False
        visual_valid = np.ones(c.num_cameras * 2, bool)
        visual_valid[-1] = index % 2 == 0
        state = rng.normal(size=58).astype(np.float32) * 0.1
        return {
            "visual_features": rng.normal(size=(len(visual_valid), c.input_dim)).astype(np.float32),
            "visual_valid_mask": visual_valid,
            "tactile": rng.uniform(
                size=(history, c.num_sensors, c.tactile_resolution, c.tactile_resolution, c.tactile_channels)
            ).astype(np.float32),
            "tactile_valid_mask": tactile_valid,
            "tactile_timestamps": np.broadcast_to(
                q + np.arange(1 - history, 1, dtype=np.float32)[:, None] / 30, (history, c.num_sensors)
            ).copy(),
            "proprioception": state,
            "query_timestamp": q,
            "visual_timestamp": np.float32(0),
            "action_target": np.tile(state[None], (6, 1)),
            "action_valid_mask": np.asarray([True] * 5 + [index % 2 == 0]),
            "episode_id": np.int32(index // 2 + (0 if self.split == "train" else 100)),
            "anchor_frame_index": np.int32(0),
            "query_frame_index": np.int32(index),
            "object": "synthetic",
            "caption": "squeeze",
        }


def synthetic_smoke(output_dir, *, mode="trlt_cross_only", seed=0, max_steps=2):
    """Tiny GPU train -> freeze -> probe exercise; no source/test evaluation."""
    assert_gpu()
    config = ModelConfig(
        mode=mode,
        input_dim=8,
        embed_dim=8,
        num_heads=2,
        num_stages=2,
        decoder_layers=1,
        rlt_encoder_layers=1,
        history_frames=2,
        num_sensors=2,
        tactile_resolution=4,
        cnn_channels=(4,),
    )
    train = SyntheticDataset(config, seed=seed)
    validation = SyntheticDataset(config, split="validation", size=2, seed=seed + 1)
    root = root_path(output_dir)
    rep = train_representation(
        train,
        validation,
        config,
        TrainConfig(batch_size=2, microbatch_size=1, max_epochs=max_steps + 1),
        root / "representation",
        seed=seed,
        max_steps=max_steps,
        log_dir=f"experiments/trlt/smoke/{mode}/seed{seed}/representation",
    )
    frozen_train = export_frozen_latents(
        train,
        rep["best_checkpoint"],
        root_path(f"datasets/trex_squeeze/synthetic_smoke/{mode}/seed{seed}/train"),
        batch_size=2,
    )
    frozen_val = export_frozen_latents(
        validation,
        rep["best_checkpoint"],
        root_path(f"datasets/trex_squeeze/synthetic_smoke/{mode}/seed{seed}/validation"),
        batch_size=2,
    )
    probe = train_action_probe(
        frozen_train,
        frozen_val,
        ProbeConfig(hidden_dim=8, batch_size=2, max_epochs=max_steps + 1),
        root / "probe",
        seed=seed,
        max_steps=max_steps,
        log_dir=f"experiments/trlt/smoke/{mode}/seed{seed}/probe",
    )
    return {"representation": rep, "probe": probe, "test_evaluation": "not-run", "fixture": "synthetic"}
