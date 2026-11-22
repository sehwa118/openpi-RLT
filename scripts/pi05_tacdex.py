"""Local baseline tools: pi05_tacdex.py {stats,inspect,infer} <train-config> [overrides]."""

import json
import sys
import time

import jax
import numpy as np
import pandas as pd

from openpi.policies import policy_config
from openpi.shared import normalize
import openpi.training.config as config_lib
import openpi.training.data_loader as data_loader


def main():
    mode = sys.argv.pop(1)
    config = config_lib.cli()
    data = config.data.create(config.assets_dirs, config.model)
    output = config.assets_dirs.parent.parent / config.exp_name
    output.mkdir(parents=True, exist_ok=True)
    if mode == "stats":
        # Read vectors directly: no video decoding, no dropped final batch, same
        # clamped action chunks as LeRobot delta_timestamps at episode boundaries.
        from pathlib import Path

        root = Path(data.local_root)
        meta = json.loads((root / "meta/info.json").read_text())
        stats = {k: normalize.RunningStats() for k in ("state", "actions")}
        action_stats = [normalize.RunningStats() for _ in range(22)]
        invalid = 0
        frames = 0
        for episode in data.episodes:
            path = root / meta["data_path"].format(episode_chunk=episode // meta["chunks_size"], episode_index=episode)
            episode_data = pd.read_parquet(path)
            mask = np.stack(episode_data["action_valid_mask"])
            invalid += int((~mask).sum())
            state = np.stack(episode_data["observation.state"]).astype(np.float32)
            action = np.stack(episode_data["action"]).astype(np.float32)
            if not (np.isfinite(state).all() and np.isfinite(action).all()):
                raise ValueError(f"Episode {episode} has nonfinite vectors")
            indices = np.minimum(
                np.arange(len(episode_data))[:, None] + np.arange(config.model.action_horizon), len(episode_data) - 1
            )
            stats["state"].update(state)
            chunks = action[indices]
            chunk_masks = mask[indices]
            for dim, accumulator in enumerate(action_stats):
                accumulator.update(chunks[..., dim][chunk_masks[..., dim]][:, None])
            frames += len(episode_data)
        per_dim = [v.get_statistics() for v in action_stats]
        action_norm = normalize.NormStats(
            **{key: np.concatenate([getattr(v, key) for v in per_dim]) for key in ("mean", "std", "q01", "q99")}
        )
        normalize.save(
            config.assets_dirs / data.asset_id, {"state": stats["state"].get_statistics(), "actions": action_norm}
        )
        report = {
            "episodes": list(data.episodes),
            "frames": frames,
            "invalid_action_elements": invalid,
            "action_horizon": config.model.action_horizon,
            "dataset_root": str(root),
            "stats_path": str(config.assets_dirs / data.asset_id),
        }
    elif mode == "inspect":
        loader = data_loader.create_data_loader(config, num_batches=1)
        obs, actions = next(iter(loader))
        report = {
            "images": {k: list(v.shape) for k, v in obs.images.items()},
            "state": list(obs.state.shape),
            "actions": list(actions.shape),
            "prompt_tokens_padded": obs.tokenized_prompt.shape[-1],
            "prompt_tokens_valid": np.asarray(obs.tokenized_prompt_mask.sum(-1)).tolist(),
            "visual_tokens": len(obs.images) * 256,
            "prefix_tokens_padded": len(obs.images) * 256 + obs.tokenized_prompt.shape[-1],
            "action_suffix_tokens": config.model.action_horizon,
            "finite": bool(np.isfinite(np.asarray(actions)).all()),
        }
    elif mode == "infer":
        # exp_name selects the trained run. Checkpoint folders use zero-based loop
        # indices, so a run of N optimizer steps ends in N-1.
        checkpoint = config.checkpoint_dir / str(config.num_train_steps - 1)
        policy = policy_config.create_trained_policy(config, checkpoint)
        raw_dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
        sample = raw_dataset[0]
        inputs = {
            "state": np.asarray(sample["observation.state"]),
            "main": np.asarray(sample["observation.images.main"]),
            "secondary": np.asarray(sample["observation.images.secondary"]),
            "prompt": sample["prompt"],
        }
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            result = policy.infer(inputs)
            jax.block_until_ready(result)
            timings.append(time.perf_counter() - start)
        actions = np.asarray(result["actions"])
        if actions.shape != (config.model.action_horizon, 22) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid inference output: {actions.shape}")
        np.save(output / "inference_actions.npy", actions)
        report = {
            "checkpoint": str(checkpoint),
            "actions_shape": list(actions.shape),
            "finite": True,
            "inference_seconds": timings,
        }
    else:
        raise ValueError(f"Unknown mode {mode}")
    report_name = "inference" if mode == "infer" else mode
    (output / f"{report_name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
