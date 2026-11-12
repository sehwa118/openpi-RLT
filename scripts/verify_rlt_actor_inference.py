"""Benchmark two-camera pi05 -> RLT encoder and VLA actions -> RL ChunkActor.

Uses pretrained pi05 and explicitly untrained RLT/actor modules. This is a
compute/connectivity benchmark, not a trained policy or robot deployment test.
Inputs are synthetic normalized images/state; robot transforms and RPC excluded.
"""

"""
terminal scripts

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync python scripts/verify_rlt_actor_inference.py \
  --device gpu \
  --init-untrained-modules \
  --runs 20 \
  --num-steps 10 \
  --chunk-len 10 \
  --benchmark-rlt-decode
  
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from verify_pi05_inference import positive_int
from verify_pi05_inference import repo_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--output-dir", default="experiments/rlt_actor_inference")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--runs", type=positive_int, default=20)
    parser.add_argument("--num-steps", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--chunk-len", type=positive_int, default=10, help="Online RL default; at most 50")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-untrained-modules",
        action="store_true",
        required=True,
        help="Acknowledge that RLT and actor use random initial weights",
    )
    parser.add_argument(
        "--benchmark-rlt-decoder",
        action="store_true",
        help="Also time reconstruction decoder separately; it is not used by the actor",
    )
    args = parser.parse_args()
    if args.chunk_len > 50:
        parser.error("--chunk-len must be <= pi05 action horizon (50)")
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / "checkpoints"))
    sys.path.insert(0, str(ROOT / "rlt_online_rl" / "src"))
    output = repo_path(args.output_dir) / datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    report = {
        "status": "failed",
        "arguments": vars(args),
        "weights": {"vla": "pretrained pi05_base", "rlt": "random initialization", "actor": "random initialization"},
        "scope": "Two actual camera tensors, synthetic normalized inputs; no robot transforms, RPC or task validation",
    }
    stage = "imports"
    try:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import numpy as np

        from openpi.models import model as model_lib
        from openpi.models.pi0 import PrefixInferenceCache
        from openpi.models.pi0 import make_attn_mask
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.rl_token import RLTokenConfig
        from openpi.models.rl_token import RLTokenModel
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared.nnx_utils import module_jit

        # The serving path constructs this cache inside a single JIT. Here it
        # crosses module boundaries so register its tensor fields as a pytree.
        jax.tree_util.register_dataclass(
            PrefixInferenceCache,
            data_fields=["observation", "prefix_mask", "image_prefix_out", "kv_cache"],
            meta_fields=[],
        )
        from rlt_online_rl.config import RLTOnlineRLConfig
        from rlt_online_rl.networks import ChunkActor

        device = jax.devices()[0]
        if args.device == "gpu" and device.platform != "gpu":
            raise RuntimeError(f"Expected GPU, got {device}")
        report.update(device=str(device), device_kind=device.device_kind, jax_version=jax.__version__)
        logging.info("Device: %s (%s). RLT and actor weights are UNTRAINED.", device, device.device_kind)

        class Stages(nnx.Module):
            def __init__(self, vla):
                self.vla = vla

            def embed(self, obs):
                # Already normalized 224x224 inputs. embed_prefix iterates only
                # the two present images; no third-camera padding/computation.
                return self.vla.embed_prefix(obs)

            def backbone(self, obs, embedded):
                tokens, mask, ar_mask = embedded
                (prefix, _), kv = self.vla.PaliGemma.llm(
                    [tokens, None], mask=make_attn_mask(mask, ar_mask), positions=jnp.cumsum(mask, axis=1) - 1
                )
                image_len = tokens.shape[1] - obs.tokenized_prompt.shape[1]
                return PrefixInferenceCache(obs, mask, prefix[:, :image_len], kv)

            def actions(self, rng, cache):
                return self.vla.sample_actions_from_prefix_cache(rng, cache, num_steps=args.num_steps)

        with jax.default_device(device):
            stage = "base checkpoint loading"
            params_dir = repo_path(args.checkpoint_dir) / "params"
            if not (params_dir / "_METADATA").is_file():
                raise FileNotFoundError(params_dir)
            start = time.perf_counter()
            config = Pi0Config(pi05=True)
            params = model_lib.restore_params(
                params_dir, dtype=jnp.bfloat16, sharding=jax.sharding.SingleDeviceSharding(device)
            )
            stages = Stages(config.load(params))
            del params
            report["vla_load_seconds"] = time.perf_counter() - start
            batch = args.batch_size
            rng = jax.random.key(args.seed)
            tokens, mask = PaligemmaTokenizer(config.max_token_len).tokenize(
                args.prompt, state=np.zeros(config.action_dim, np.float32)
            )
            cameras = ("base_0_rgb", "left_wrist_0_rgb")
            rng_np = np.random.default_rng(args.seed)
            obs = model_lib.Observation(
                images={k: jnp.asarray(rng_np.uniform(-1, 1, (batch, 224, 224, 3)), jnp.float32) for k in cameras},
                image_masks={k: jnp.ones((batch,), dtype=bool) for k in cameras},
                state=jnp.zeros((batch, config.action_dim)),
                tokenized_prompt=jnp.asarray(np.tile(tokens, (batch, 1)), jnp.int32),
                tokenized_prompt_mask=jnp.asarray(np.tile(mask, (batch, 1))),
            )
            embed = module_jit(stages.embed)
            backbone = module_jit(stages.backbone)
            actions_fn = module_jit(stages.actions)
            # Match rlt_pi05_agilexbag_image: one 2048-dimensional token, two layers.
            rlt_config = RLTokenConfig(num_rl_tokens=1, embed_dim=2048, input_dim=2048)
            rlt = RLTokenModel(rlt_config)
            stage = "RLT and actor initialization"
            embedded = jax.block_until_ready(embed(obs))
            image_len = embedded[0].shape[1] - config.max_token_len
            dummy_prefix = jnp.zeros((batch, image_len, rlt_config.input_dim))
            variables = rlt.init(rng, dummy_prefix, None, train=False)
            encode = jax.jit(lambda p, x: rlt.apply(p, x.astype(jnp.float32), None, method="encode", train=False))
            decode = jax.jit(lambda p, z: rlt.apply(p, z, image_len, method="decode", train=False))
            rl_config = RLTOnlineRLConfig(chunk_len=args.chunk_len)
            actor = ChunkActor(
                rl_config.z_dim,
                rl_config.proprio_dim,
                rl_config.chunk_len,
                rl_config.action_dim,
                rl_config.actor_hidden_dim,
                rl_config.actor_num_layers,
                rl_config.fixed_std,
            )
            actor_params = actor.init_params(jax.random.fold_in(rng, 1))
            actor_fn = jax.jit(
                lambda p, key, z, proprio, ref: actor.sample_action(p, key, z, proprio, ref, deterministic=True)
            )
            jax.block_until_ready((variables, actor_params, obs))
            samples = {}
            warmup = {}

            def timed(name, fn, *inputs, warming=False):
                start = time.perf_counter()
                result = jax.block_until_ready(fn(*inputs))
                ms = (time.perf_counter() - start) * 1000
                if warming:
                    warmup[name] = ms
                else:
                    samples.setdefault(name, []).append(ms)
                return result

            def pipeline(key, *, warming=False):
                begin = time.perf_counter()
                emb = timed("image_and_text_embedding", embed, obs, warming=warming)
                cache = timed("vla_prefix_backbone", backbone, obs, emb, warming=warming)
                z = timed("rlt_encoder", encode, variables, cache.image_prefix_out, warming=warming)
                ref = timed("vla_action_decoder", actions_fn, key, cache, warming=warming)
                # Preserve runtime contract (flattened token, 7D proprio, first
                # chunk_len actions). Values stay in synthetic model space.
                z_flat = z.reshape(batch, -1)
                refined = timed(
                    "rl_actor",
                    actor_fn,
                    actor_params,
                    key,
                    z_flat,
                    obs.state[:, :7],
                    ref[:, : args.chunk_len, :7],
                    warming=warming,
                )
                elapsed = (time.perf_counter() - begin) * 1000
                if warming:
                    warmup["pipeline_total"] = elapsed
                else:
                    samples.setdefault("pipeline_total", []).append(elapsed)
                return z, ref, refined

            stage = "module warmup / compilation"
            logging.info("Warmup: two images, %s image tokens, actor chunk length %s", image_len, args.chunk_len)
            z, ref, refined = pipeline(rng, warming=True)
            for index in range(args.runs):
                stage = f"inference {index + 1}"
                z, ref, refined = pipeline(jax.random.fold_in(rng, index + 2))
                logging.info("Run %d/%d: %.2f ms", index + 1, args.runs, samples["pipeline_total"][-1])
            if args.benchmark_rlt_decoder:
                stage = "optional RLT reconstruction decoder (outside inference pipeline)"
                timed("rlt_decoder_diagnostic", decode, variables, z, warming=True)
                for _ in range(args.runs):
                    reconstruction = timed("rlt_decoder_diagnostic", decode, variables, z)
                if not np.isfinite(np.asarray(reconstruction)).all():
                    raise RuntimeError("Nonfinite RLT decoder output")
            stage = "output validation"
            expected = {
                "z_rl": (batch, 1, 2048),
                "ref_chunk": (batch, 50, 32),
                "actor_chunk": (batch, args.chunk_len, 7),
            }
            arrays = {name: np.asarray(value) for name, value in zip(expected, (z, ref, refined), strict=True)}
            for name, value in arrays.items():
                if value.shape != expected[name] or not np.isfinite(value).all():
                    raise RuntimeError(f"Invalid {name}: shape={value.shape}, expected={expected[name]}")
            np.savez(output / "outputs.npz", **arrays)
            stats = {
                name: {
                    "mean_ms": float(np.mean(values)),
                    "p50_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "samples_ms": values,
                }
                for name, values in samples.items()
            }
            report.update(
                status="passed",
                cameras=list(cameras),
                image_token_count=image_len,
                output_shapes={k: list(v.shape) for k, v in arrays.items()},
                latency=stats,
                warmup_ms=warmup,
                memory_stats=device.memory_stats(),
                timing_method="Synchronized sequential module wall time; warmup excluded. pipeline_total includes dispatch/reshape overhead; decoder diagnostic excluded.",
            )
            for name, values in stats.items():
                logging.info("%-28s mean=%8.3f ms p95=%8.3f ms", name, values["mean_ms"], values["p95_ms"])
            logging.info("PASS (untrained RLT/actor). Artifacts: %s", output)
    except Exception as exc:
        report.update(failed_stage=stage, error=str(exc), traceback=traceback.format_exc())
        logging.exception("FAIL at %s", stage)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
