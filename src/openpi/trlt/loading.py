"""Ordered, bounded CPU prefetch using the same Torch DataLoader pattern as openpi."""

from contextlib import contextmanager
from dataclasses import dataclass
import os

import numpy as np


class OrderedBatchSampler:
    """Only the training loop advances the saved cursor, never prefetch workers."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.indices = []

    def __iter__(self):
        indices = self.indices
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


@dataclass
class TrainingCollator:
    fields: tuple
    micro_size: int
    probe: bool

    def __call__(self, samples):
        # Imported after worker initialization, with the CPU backend selected.
        from openpi.trlt.runner import _counts
        from openpi.trlt.runner import collate

        counts = _counts(samples, self.probe)
        denominator = np.asarray([counts.get("action", counts.get("visual", 0)), counts.get("tactile", 0)], np.float32)
        parts = [samples[start : start + self.micro_size] for start in range(0, len(samples), self.micro_size)]
        return (
            tuple(collate(part, self.fields) for part in parts),
            denominator,
            [_counts(part, self.probe) for part in parts],
            len(samples),
        )


def _cpu_worker_init(worker_id):
    del worker_id
    # Spawn rather than fork a parent with a live CUDA/JAX runtime. Workers must
    # never initialize CUDA, including when launched from the GPU training CLI.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    import jax
    import torch

    # Dataset unpickling may import JAX before worker_init_fn runs. Updating
    # environment variables alone would leave its cached configuration on CUDA.
    jax.config.update("jax_platforms", "cpu")
    torch.set_num_threads(1)


@contextmanager
def training_loader(dataset, sampler, collator, *, num_workers=2, prefetch_factor=2, seed=0):
    import torch

    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be nonnegative and prefetch_factor must be positive")
    options = {}
    if num_workers:
        options = {
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": prefetch_factor,
            "worker_init_fn": _cpu_worker_init,
        }
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
        **options,
    )
    try:
        yield loader
    finally:
        # Persistent workers also need cleanup on bounded runs and exceptions.
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()  # noqa: SLF001
