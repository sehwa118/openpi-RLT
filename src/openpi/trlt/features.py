"""Frozen pi0.5 final-layer image feature extraction, never the action expert."""

import hashlib
import inspect
import os
import shutil
import time

import numpy as np

from openpi.trlt.cache import FeatureCache
from openpi.trlt.cache import feature_metadata
from openpi.trlt.cache import write_json
from openpi.trlt.config import CAMERA_KEYS
from openpi.trlt.config import TemporalConfig
from openpi.trlt.config import digest
from openpi.trlt.config import root_path


class FrozenPrefixExtractor:
    def __init__(self, checkpoint, *, max_token_len=200, camera_keys=CAMERA_KEYS, prefix_microbatch_size=None):
        self.prefix_microbatch_size = prefix_microbatch_size
        self.camera_keys = tuple(camera_keys)
        os.environ.setdefault("OPENPI_DATA_HOME", str(root_path("checkpoints")))
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        from openpi.models import model as model_lib
        from openpi.models import pi0_config
        from openpi.models import tokenizer

        jax.config.update("jax_default_matmul_precision", "highest")
        self.config = pi0_config.Pi0Config(pi05=True, discrete_state_input=False, max_token_len=max_token_len)
        self.tokenizer = tokenizer.PaligemmaTokenizer(max_token_len)
        self.tokenizer_hash = digest(
            {
                "sentencepiece_model_sha256": hashlib.sha256(
                    self.tokenizer._tokenizer.serialized_model_proto()  # noqa: SLF001
                ).hexdigest(),
                "implementation_sha256": hashlib.sha256(
                    inspect.getsource(tokenizer.PaligemmaTokenizer.tokenize).encode()
                ).hexdigest(),
            }
        )
        self.model = self.config.load(model_lib.restore_params(root_path(checkpoint), dtype=jnp.bfloat16))
        self.model.eval()
        # Existing NNX/Linen bridge remains the sole backbone implementation.
        self.forward = nnx.jit(
            lambda model, obs: model.extract_prefix_embeddings(
                jax.random.key(0),
                obs,
                train=False,
                image_only=True,
                image_keys=self.camera_keys,
                prefix_microbatch_size=self.prefix_microbatch_size,
            )
        )
        state = nnx.state(self.model, nnx.Param)
        leaves = jax.tree.leaves(state)
        self.placement = {
            "backend": jax.default_backend(),
            "devices": sorted({str(x.device) for x in leaves}),
            "parameter_count": sum(x.size for x in leaves),
            "parameter_devices": sorted({str(device) for x in leaves for device in x.devices()}),
        }

    def prepare_visual_batch(self, anchors, *, pad_to=None):
        import jax
        import jax.numpy as jnp

        from openpi.models.model import Observation

        count = len(anchors)
        if not count or (pad_to is not None and pad_to < count):
            raise ValueError("Invalid feature batch/padding")
        padded = list(anchors) + [anchors[-1]] * ((pad_to or count) - count)
        if any(tuple(anchor["images"]) != self.camera_keys for anchor in padded):
            raise ValueError("Camera order/mapping mismatch")
        tokenized = [self.tokenizer.tokenize(anchor["caption"], state=None) for anchor in padded]
        prompts = np.stack([item[0] for item in tokenized]).astype(np.int32)
        prompt_masks = np.stack([item[1] for item in tokenized])
        # Keep original uint8->float32 scaling and GPU resize; no PIL/CPU resampling.
        observation = Observation.from_dict(
            {
                "image": {
                    key: np.stack([np.asarray(a["images"][key], np.uint8) for a in padded]) for key in self.camera_keys
                },
                "image_mask": {key: np.ones(len(padded), bool) for key in self.camera_keys},
                "state": np.zeros((len(padded), self.config.action_dim), np.float32),
                "tokenized_prompt": prompts,
                "tokenized_prompt_mask": prompt_masks,
            }
        )
        start = time.perf_counter()
        device_observation = jax.tree.map(jnp.asarray, observation)
        features, mask = self.forward(self.model, device_observation)
        features.block_until_ready()
        elapsed = time.perf_counter() - start
        if features.shape[1] % len(self.camera_keys):
            raise ValueError("Image token count does not cover requested cameras")
        return {
            "input_devices": sorted({str(d) for x in jax.tree.leaves(device_observation) for d in x.devices()}),
            "feature_devices": sorted(str(d) for d in features.devices()),
            "visual_features": np.asarray(features[:count], np.float32),
            "visual_valid_mask": np.asarray(mask[:count], bool),
            "elapsed_seconds": elapsed,
            "caption_tokenization_hashes": [
                digest(
                    {"caption": anchors[i]["caption"], "tokens": prompts[i].tolist(), "mask": prompt_masks[i].tolist()}
                )
                for i in range(count)
            ],
        }

    def prepare_visual_context(self, anchor):
        result = self.prepare_visual_batch([anchor])
        return {
            **{
                key: value
                for key, value in result.items()
                if key not in ("visual_features", "visual_valid_mask", "caption_tokenization_hashes")
            },
            "visual_features": result["visual_features"][0],
            "visual_valid_mask": result["visual_valid_mask"][0],
            "caption_tokenization_hash": result["caption_tokenization_hashes"][0],
        }


def prepare_visual_context(extractor, observation_at_anchor):
    return extractor.prepare_visual_context(observation_at_anchor)


def extract_visual_cache(
    dataset,
    *,
    backbone_checkpoint,
    output_dir,
    revision,
    episode_manifest_hash,
    temporal=None,
    max_anchors=None,
    episode_ids=None,
    force=False,
    plan_only=False,
    batch_size=1,
    num_workers=0,
    prefetch_factor=2,
    prefix_microbatch_size=None,
):
    total_start = time.perf_counter()
    if prefix_microbatch_size not in (None, 1):
        raise ValueError("Only prefix_microbatch_size=1 or None is supported")
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("Invalid feature batch/worker/prefetch options")
    temporal = temporal or TemporalConfig()
    # Enumerate only query-reachable anchors; this is the identical schedule used by all models.
    selected = set(map(int, episode_ids)) if episode_ids else None
    keys = sorted(
        {
            (int(ep), int(query) // temporal.visual_refresh_frames * temporal.visual_refresh_frames)
            for ep, query in dataset.queries
            if selected is None or int(ep) in selected
        }
    )
    if max_anchors is not None:
        if max_anchors <= 0:
            raise ValueError("max_anchors must be positive")
        keys = keys[:max_anchors]
    camera_keys = tuple(getattr(dataset, "camera_keys", CAMERA_KEYS))
    tokens = 256 * len(camera_keys)
    # Standard pi0.5 So400m/14 produces 256 tokens/camera at224 and Gemma width2048.
    # Verify these assumptions against the first real forward; estimates aren't measurements.
    estimate = {
        "anchors_requested": len(keys),
        "estimated_shape": [tokens, 2048],
        "estimated_float32_bytes": len(keys) * (tokens * 2048 * 4 + tokens),
        "plan_only": plan_only,
        "prefix_microbatch_size": prefix_microbatch_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
    }
    destination = root_path(output_dir)
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    estimate["available_disk_bytes"] = shutil.disk_usage(ancestor).free
    missing = (
        len(keys)
        if force
        else sum(
            not (destination / f"episode_{episode:06d}" / f"anchor_{anchor:06d}.npz").exists()
            for episode, anchor in keys
        )
    )
    estimate["estimated_new_bytes"] = missing * (tokens * 2048 * 4 + tokens)
    if plan_only:
        return estimate
    if estimate["estimated_new_bytes"] * 1.05 > estimate["available_disk_bytes"]:
        raise ValueError("Insufficient disk space for planned visual cache; no backbone initialized")
    extractor_options = {"camera_keys": camera_keys}
    if prefix_microbatch_size is not None:
        extractor_options["prefix_microbatch_size"] = prefix_microbatch_size
    extractor = FrozenPrefixExtractor(backbone_checkpoint, **extractor_options)
    caption_hash = dataset.provenance["caption_hash"]
    metadata = feature_metadata(
        revision=revision,
        episode_manifest_hash=episode_manifest_hash,
        backbone_checkpoint=backbone_checkpoint,
        caption_hash=caption_hash,
        tokenizer_hash=digest({"model": extractor.tokenizer_hash, "max_len": extractor.config.max_token_len}),
        temporal=temporal,
        camera_order=getattr(dataset, "camera_order", None),
        camera_keys=camera_keys,
    )
    if batch_size > 1 or num_workers > 0:
        metadata["extraction"] = {
            "version": "batched_anchor_v1",
            "prefix_microbatch_size": prefix_microbatch_size,
            "fixed_batch_size": batch_size,
            "tail": "repeat_last_then_discard",
            "reader": "sequential_native_mp4" if hasattr(dataset, "iter_anchors") else "anchor_reader",
            "preprocessing": "original_float32_scale_and_gpu_resize",
        }
        return _extract_batched(
            dataset,
            keys,
            extractor,
            output_dir,
            metadata,
            estimate,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            force=force,
            total_start=total_start,
        )
    cache = FeatureCache.create(output_dir, metadata, force=force)
    timings = []
    actual = None
    for key in keys:
        if key in cache:
            continue
        anchor = dataset.read_anchor(*key)
        result = extractor.prepare_visual_context(anchor)
        cache.put(
            key,
            result["visual_features"],
            result["visual_valid_mask"],
            caption_hash=result["caption_tokenization_hash"],
            visual_timestamp=anchor["visual_timestamp"],
            camera_timestamps=anchor["camera_timestamps"],
        )
        timings.append(result["elapsed_seconds"])
        actual = {
            "shape": list(result["visual_features"].shape),
            "tokens_per_camera": result["visual_features"].shape[0] // len(camera_keys),
            "input_devices": result["input_devices"],
            "feature_devices": result["feature_devices"],
        }
    result = {
        **estimate,
        "plan_only": False,
        "cache_hash": cache.cache_hash,
        "written": len(timings),
        "available_anchors": len(cache.keys()),
        "backbone_placement": extractor.placement,
        "actual_features": actual,
        "first_call_seconds_including_compile": timings[0] if timings else None,
        "subsequent_call_seconds": timings[1:],
    }
    write_json(root_path(output_dir) / "extraction_report.json", result)
    return result


def _extract_batched(
    dataset,
    keys,
    extractor,
    output_dir,
    metadata,
    estimate,
    *,
    batch_size,
    num_workers,
    prefetch_factor,
    force,
    total_start,
):
    from tqdm.auto import tqdm

    from openpi.trlt.feature_loading import anchor_loader

    cache = FeatureCache.create(output_dir, metadata, force=force)
    missing = [key for key in keys if key not in cache]
    seen = set()
    timings = []
    data_seconds = write_seconds = 0.0
    actual = None
    started = time.perf_counter()
    with (
        anchor_loader(
            dataset, missing, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor
        ) as batches,
        tqdm(total=len(missing), desc="Frozen anchor features", unit="anchor") as bar,
    ):
        while True:
            before = time.perf_counter()
            try:
                samples = next(batches)
            except StopIteration:
                break
            data_seconds += time.perf_counter() - before
            batch_keys = [tuple(key) for key, _ in samples]
            if any(key in seen for key in batch_keys) or len(batch_keys) != len(set(batch_keys)):
                raise ValueError("Duplicate anchor from feature workers")
            result = extractor.prepare_visual_batch([anchor for _, anchor in samples], pad_to=batch_size)
            if result["visual_features"].shape[0] != len(samples):
                raise ValueError("Padded output count differs from actual anchors")
            before = time.perf_counter()
            for index, (key, anchor) in enumerate(samples):
                cache.put(
                    key,
                    result["visual_features"][index],
                    result["visual_valid_mask"][index],
                    caption_hash=result["caption_tokenization_hashes"][index],
                    visual_timestamp=anchor["visual_timestamp"],
                    camera_timestamps=anchor["camera_timestamps"],
                )
                seen.add(tuple(key))
            write_seconds += time.perf_counter() - before
            timings.append(result["elapsed_seconds"])
            actual = {
                "shape": list(result["visual_features"].shape[1:]),
                "tokens_per_camera": result["visual_features"].shape[1] // len(extractor.camera_keys),
                "input_devices": result["input_devices"],
                "feature_devices": result["feature_devices"],
            }
            bar.update(len(samples))
            bar.set_postfix(batch=batch_size, data_s=f"{data_seconds:.1f}", gpu_s=f"{sum(timings):.1f}")
    if seen != set(missing):
        raise ValueError("Missing/unexpected anchor keys from feature workers")
    elapsed = time.perf_counter() - started
    report = {
        **estimate,
        "plan_only": False,
        "cache_hash": cache.cache_hash,
        "written": len(seen),
        "available_anchors": len(cache.keys()),
        "backbone_placement": extractor.placement,
        "actual_features": actual,
        "batches": len(timings),
        "elapsed_seconds": elapsed,
        "total_seconds_including_backbone_load": time.perf_counter() - total_start,
        "anchors_per_second": len(seen) / elapsed if elapsed else None,
        "data_wait_seconds": data_seconds,
        "forward_seconds": sum(timings),
        "write_seconds": write_seconds,
        "first_call_seconds_including_compile": timings[0] if timings else None,
        "subsequent_call_seconds": timings[1:],
    }
    write_json(root_path(output_dir) / "extraction_report.json", report)
    return report
