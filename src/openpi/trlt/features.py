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
    def __init__(self, checkpoint, *, max_token_len=200):
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
            lambda model, obs: model.extract_prefix_embeddings(jax.random.key(0), obs, train=False, image_only=True)
        )
        state = nnx.state(self.model, nnx.Param)
        leaves = jax.tree.leaves(state)
        self.placement = {
            "backend": jax.default_backend(),
            "devices": sorted({str(x.device) for x in leaves}),
            "parameter_count": sum(x.size for x in leaves),
            "parameter_devices": sorted({str(device) for x in leaves for device in x.devices()}),
        }

    def prepare_visual_context(self, anchor):
        import jax
        import jax.numpy as jnp

        from openpi.models.model import Observation

        images = anchor["images"]
        if tuple(images) != CAMERA_KEYS:
            raise ValueError(f"Camera order/mapping mismatch: {tuple(images)}")
        prompt, prompt_mask = self.tokenizer.tokenize(anchor["caption"], state=None)
        # Prefix-only embed_prefix reads images and caption, never Observation.state.
        # This placeholder is not truncated T-Rex state and is never probe input.
        observation = Observation.from_dict(
            {
                "image": {name: np.asarray(images[name], np.uint8)[None] for name in CAMERA_KEYS},
                "image_mask": {name: np.ones((1,), bool) for name in CAMERA_KEYS},
                "state": np.zeros((1, self.config.action_dim), np.float32),
                "tokenized_prompt": prompt[None].astype(np.int32),
                "tokenized_prompt_mask": prompt_mask[None],
            }
        )
        start = time.perf_counter()
        features, mask = self.forward(self.model, jax.tree.map(jnp.asarray, observation))
        features.block_until_ready()
        elapsed = time.perf_counter() - start
        if features.shape[1] % len(CAMERA_KEYS) != 0:
            raise ValueError("Image token count does not cover all three cameras")
        return {
            "input_devices": sorted(
                {str(d) for x in jax.tree.leaves(jax.tree.map(jnp.asarray, observation)) for d in x.devices()}
            ),
            "feature_devices": sorted(str(d) for d in features.devices()),
            "visual_features": np.asarray(features[0], np.float32),
            "visual_valid_mask": np.asarray(mask[0], bool),
            "elapsed_seconds": elapsed,
            "caption_tokenization_hash": digest(
                {"caption": anchor["caption"], "tokens": prompt.tolist(), "mask": prompt_mask.tolist()}
            ),
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
):
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
    # Standard pi0.5 So400m/14 produces 256 tokens/camera at224 and Gemma width2048.
    # Verify these assumptions against the first real forward; estimates aren't measurements.
    estimate = {
        "anchors_requested": len(keys),
        "estimated_shape": [768, 2048],
        "estimated_float32_bytes": len(keys) * (768 * 2048 * 4 + 768),
        "plan_only": plan_only,
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
    estimate["estimated_new_bytes"] = missing * (768 * 2048 * 4 + 768)
    if plan_only:
        return estimate
    if estimate["estimated_new_bytes"] * 1.05 > estimate["available_disk_bytes"]:
        raise ValueError("Insufficient disk space for planned visual cache; no backbone initialized")
    extractor = FrozenPrefixExtractor(backbone_checkpoint)
    caption_hash = dataset.provenance["caption_hash"]
    metadata = feature_metadata(
        revision=revision,
        episode_manifest_hash=episode_manifest_hash,
        backbone_checkpoint=backbone_checkpoint,
        caption_hash=caption_hash,
        tokenizer_hash=digest({"model": extractor.tokenizer_hash, "max_len": extractor.config.max_token_len}),
        temporal=temporal,
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
            "tokens_per_camera": result["visual_features"].shape[0] // len(CAMERA_KEYS),
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
