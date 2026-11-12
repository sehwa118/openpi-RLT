"""Smoke-test the local JAX pi0.5 base checkpoint with synthetic observations.

This checks model execution, not robot behavior or task success. Output actions
remain in model space; robot-specific normalization/transforms are not applied.
All relative paths resolve against the repository root.
"""

"""
terminal scripts 

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync python scripts/verify_pi05_inference.py \
  --device gpu \
  --checkpoint-dir checkpoints/openpi-assets/checkpoints/pi05_base \
  --runs 3 \
  --num-steps 10 \
  --prompt "pick up the object
  
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--output-dir", default="experiments/pi05_inference")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--runs", type=positive_int, default=3, help="Timed runs after one warmup/compile run")
    parser.add_argument("--num-steps", type=positive_int, default=10, help="Flow-matching denoising steps")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="pick up the object")
    args = parser.parse_args()

    # Configure the backend before importing JAX or any project module.
    # GPU mode fails if CUDA is unavailable instead of silently using CPU.
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / "checkpoints"))

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = repo_path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log")],
    )
    report = {"status": "failed", "arguments": vars(args), "output_dir": str(output_dir)}
    stage = "checkpoint validation"
    try:
        checkpoint_dir = repo_path(args.checkpoint_dir)
        params_dir = checkpoint_dir / "params"
        if not (params_dir / "_METADATA").is_file():
            raise FileNotFoundError(f"Expected an Orbax JAX checkpoint at {params_dir}")
        report["checkpoint_dir"] = str(checkpoint_dir)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            report["nvidia_smi_before"] = result.stdout.strip()
            logging.info("GPU inventory:\n%s", result.stdout.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            report["nvidia_smi_error"] = str(exc)

        stage = "imports and backend initialization"
        import jax
        import jax.numpy as jnp
        import numpy as np

        from openpi.models import model as model_lib
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared.nnx_utils import module_jit

        devices = jax.devices()
        if args.device == "gpu" and any(device.platform != "gpu" for device in devices):
            raise RuntimeError(f"Expected GPU devices, got {devices}")
        report["jax_version"] = jax.__version__
        report["devices"] = [str(device) for device in devices]
        report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        logging.info("JAX %s; devices: %s", jax.__version__, devices)
        device = devices[0]
        with jax.default_device(device):
            config = Pi0Config(pi05=True)
            stage = "tokenizer and synthetic observations"
            state = np.zeros(config.action_dim, dtype=np.float32)
            tokens, mask = PaligemmaTokenizer(config.max_token_len).tokenize(args.prompt, state=state)
            rng_np = np.random.default_rng(args.seed)
            batch = args.batch_size
            observation = model_lib.Observation(
                images={
                    key: jnp.asarray(rng_np.uniform(-1, 1, (batch, 224, 224, 3)), dtype=jnp.float32)
                    for key in model_lib.IMAGE_KEYS
                },
                image_masks={key: jnp.ones((batch,), dtype=jnp.bool_) for key in model_lib.IMAGE_KEYS},
                state=jnp.asarray(np.tile(state, (batch, 1))),
                tokenized_prompt=jnp.asarray(np.tile(tokens, (batch, 1)), dtype=jnp.int32),
                tokenized_prompt_mask=jnp.asarray(np.tile(mask, (batch, 1))),
            )
            stage = "checkpoint restore and model loading"
            logging.info("Loading %s on %s", params_dir, device)
            start = time.perf_counter()
            # Pin restore to one GPU; do not replicate weights over other visible GPUs.
            params = model_lib.restore_params(
                params_dir, dtype=jnp.bfloat16, sharding=jax.sharding.SingleDeviceSharding(device)
            )
            model = config.load(params)
            del params
            sample = module_jit(model.sample_actions)
            report["load_seconds"] = time.perf_counter() - start
            rng = jax.random.key(args.seed)
            durations = []
            expected_shape = (batch, config.action_horizon, config.action_dim)
            for index in range(args.runs + 1):
                stage = "warmup / JIT compilation" if index == 0 else f"inference {index}"
                logging.info("Starting %s", stage)
                rng, sample_rng = jax.random.split(rng)
                start = time.perf_counter()
                actions = sample(sample_rng, observation, num_steps=args.num_steps)
                actions.block_until_ready()  # GPU work is asynchronous; synchronize for timing.
                elapsed = time.perf_counter() - start
                host_actions = np.asarray(actions)
                if host_actions.shape != expected_shape:
                    raise RuntimeError(f"Unexpected action shape {host_actions.shape}; expected {expected_shape}")
                if not np.isfinite(host_actions).all():
                    raise RuntimeError("Actions contain NaN or infinity")
                logging.info("%s: %.3f s; shape=%s; all finite", stage, elapsed, host_actions.shape)
                if index == 0:
                    report["warmup_compile_seconds"] = elapsed
                else:
                    durations.append(elapsed * 1000)
            np.save(output_dir / "actions.npy", host_actions)
            report.update(
                status="passed",
                action_shape=list(host_actions.shape),
                action_min=float(host_actions.min()),
                action_max=float(host_actions.max()),
                inference_ms=durations,
                mean_inference_ms=float(np.mean(durations)),
                device_memory_stats=device.memory_stats(),
                scope="JAX pi05_base model inference with synthetic images and tokenized prompt; no robot transforms",
            )
            logging.info("PASS: mean inference %.1f ms; artifacts: %s", np.mean(durations), output_dir)
    except Exception as exc:
        report.update(failed_stage=stage, error=str(exc), traceback=traceback.format_exc())
        logging.exception("FAIL at %s; report: %s", stage, output_dir / "report.json")
        raise
    finally:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
