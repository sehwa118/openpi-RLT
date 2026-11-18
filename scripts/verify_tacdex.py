#!/usr/bin/env python3
"""Bounded CPU Assembly train/validation reader check; no backbone or test samples."""

import argparse
from datetime import UTC
from datetime import datetime
import os


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="experiments/trlt/3dtacdex_assembly/implementation/real_data.json")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import jax
    import numpy as np

    from openpi.models.model import Observation
    from openpi.models.model import preprocess_observation
    from openpi.trlt.cache import write_json
    from openpi.trlt.config import root_path
    from openpi.trlt.loading import OrderedBatchSampler
    from openpi.trlt.loading import training_loader
    from openpi.trlt.tacdex_data import AssemblyDataset

    report = {
        "date": datetime.now(UTC).isoformat(),
        "device": jax.default_backend(),
        "training_steps": 0,
        "backbone_initialized": False,
        "test_samples_read": 0,
        "splits": {},
    }
    for split in ("train", "validation"):
        dataset = AssemblyDataset(split=split)
        sampler = OrderedBatchSampler(3)
        # Episode boundaries plus interior, across multiple source trajectories.
        indices = sorted({0, 1, 2, len(dataset) - 1, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4})
        sampler.indices = indices
        # Use list collation to inspect all tensors and host lineage equally.
        with training_loader(dataset, sampler, list, num_workers=0) as loader:
            expected = [sample for batch in loader for sample in batch]
        with training_loader(dataset, sampler, list, num_workers=2, prefetch_factor=2) as loader:
            actual = [sample for batch in loader for sample in batch]
        for left, right in zip(expected, actual, strict=True):
            assert left.keys() == right.keys()
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
            assert left["tactile"].shape == (3, 8, 15, 1, 3)
            assert left["action_valid_mask"].shape == (3, 22)
            assert np.isfinite(left["action_target"]).all()
            mask = left["tactile_valid_mask"]
            assert np.all(left["tactile_timestamps"][mask] <= left["query_timestamp"])
        episode, query = dataset.queries[indices[-2]]
        anchor_index = query // dataset.temporal.visual_refresh_frames * dataset.temporal.visual_refresh_frames
        anchor = dataset.read_anchor(episode, anchor_index)
        # CPU real two-camera pixels through the same preprocessing used by Pi0.
        obs = Observation.from_dict(
            {
                "image": {k: v[None] for k, v in anchor["images"].items()},
                "image_mask": {k: np.ones(1, bool) for k in anchor["images"]},
                "state": np.zeros((1, 32), np.float32),
            }
        )
        processed = preprocess_observation(None, obs, image_keys=dataset.camera_keys, train=False)
        assert tuple(processed.images) == dataset.camera_keys
        assert all(image.shape == (1, 224, 224, 3) for image in processed.images.values())
        report["splits"][split] = {
            "query_count": len(dataset),
            "decoded_anchor": [episode, anchor_index],
            "indices": indices,
            "exact_worker_matches": len(actual),
            "rgb_shapes": {k: list(v.shape) for k, v in anchor["images"].items()},
            "preprocessed_shapes": {k: list(v.shape) for k, v in processed.images.items()},
            "provenance": dataset.provenance,
        }
    report["status"] = "pass"
    write_json(root_path(args.output), report)
    print(f"CPU real-data validation passed: {args.output}")


if __name__ == "__main__":
    main()
