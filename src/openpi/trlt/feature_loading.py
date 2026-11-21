"""Bounded CPU anchor prefetch; whole episode tasks preserve sequential decoding."""

from collections import defaultdict
from contextlib import contextmanager

import torch

from openpi.trlt.loading import _cpu_worker_init


class AnchorStream(torch.utils.data.IterableDataset):
    def __init__(self, dataset, keys):
        self.dataset = dataset
        grouped = defaultdict(list)
        for episode, anchor in keys:
            grouped[int(episode)].append(int(anchor))
        self.tasks = [(episode, sorted(anchors)) for episode, anchors in sorted(grouped.items())]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        tasks = self.tasks if worker is None else self.tasks[worker.id :: worker.num_workers]
        for episode, anchors in tasks:
            if hasattr(self.dataset, "iter_anchors"):
                yield from self.dataset.iter_anchors(episode, anchors)
            else:
                for anchor in anchors:
                    yield (episode, anchor), self.dataset.read_anchor(episode, anchor)


@contextmanager
def anchor_loader(dataset, keys, *, batch_size=1, num_workers=0, prefetch_factor=2):
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("Invalid feature batch/worker/prefetch options")
    options = {}
    if num_workers:
        options = {
            "multiprocessing_context": "spawn",
            "prefetch_factor": prefetch_factor,
            "worker_init_fn": _cpu_worker_init,
        }
    loader = torch.utils.data.DataLoader(
        AnchorStream(dataset, keys),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list,
        drop_last=False,
        **options,
    )
    iterator = iter(loader)
    try:
        yield iterator
    finally:
        if num_workers:
            iterator._shutdown_workers()  # noqa: SLF001 -- deterministic cleanup on bounded/error runs
