"""Optional experiment tracking, independent of model/optimizer identity."""

import json
from pathlib import Path

import numpy as np


def scalar_metrics(values, prefix):
    """Expand sensor/history arrays into individual chartable scalar series."""
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update(scalar_metrics(value, f"{prefix}/{key}"))
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            continue
        if array.ndim == 0:
            result[f"{prefix}/{key}"] = array.item()
        else:
            for index in np.ndindex(array.shape):
                suffix = "/".join(map(str, index))
                result[f"{prefix}/{key}/{suffix}"] = array[index].item()
    return result


RECONSTRUCTION_METRICS = frozenset(
    {
        "total_loss",
        "visual_loss",
        "visual_weighted",
        "tactile_loss",
        "tactile_weighted",
        "tactile_latest",
        "tactile_history",
        "tactile_per_sensor",
        "tactile_per_history",
        "mse",
    }
)


def grouped_metrics(event, phase, stage):
    """Keep display metrics separate from local aggregation/masking diagnostics."""
    group = "stage_b" if stage == "probe" else "stage_a"
    result = {"optimizer_step": event["step"]}
    result.update(
        scalar_metrics(
            {key: event[key] for key in ("step", "epoch", "examples", "improved") if key in event},
            f"progress/{phase}",
        )
    )
    result.update(
        scalar_metrics(
            {key: event[key] for key in ("data_seconds", "update_seconds") if key in event},
            f"timing/{phase}",
        )
    )
    result.update(
        scalar_metrics(
            {key: event[key] for key in ("loss", "value", "best_value") if key in event},
            f"{group}/{phase}",
        )
    )
    allowed = {"action_loss"} if stage == "probe" else RECONSTRUCTION_METRICS
    result.update(
        scalar_metrics(
            {key: value for key, value in event.get("metrics", {}).items() if key in allowed},
            f"{group}/{phase}",
        )
    )
    return result


class TrainingTracker:
    def __init__(self, directory, manifest, *, options=None, resume=False):
        self.directory = Path(directory)
        self.manifest = manifest
        self.options = options or {}
        self.resume = resume
        self.run = None
        self.bar = None

    def __enter__(self):
        if self.options.get("mode", "disabled") == "disabled":
            return self
        import wandb

        self.directory.mkdir(parents=True, exist_ok=True)
        identity_path = self.directory / "wandb_run.json"
        previous = json.loads(identity_path.read_text()) if self.resume and identity_path.exists() else {}
        project = self.options.get("project") or previous.get("project") or "trlt"
        entity = self.options.get("entity") or previous.get("entity")
        if previous and (project != previous["project"] or entity != previous.get("entity")):
            raise ValueError("W&B resume project/entity differs from wandb_run.json")
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=self.options.get("name"),
            group=self.options.get("group"),
            mode=self.options["mode"],
            id=previous.get("id"),
            resume="allow" if previous else None,
            dir=str(self.directory),
            config=self.manifest,
            allow_val_change=True,
        )
        try:
            self.run.define_metric("optimizer_step")
            for group in ("progress", "timing", "stage_a", "stage_b"):
                self.run.define_metric(f"{group}/*", step_metric="optimizer_step")
            identity_path.write_text(
                json.dumps(
                    {
                        "id": self.run.id,
                        "project": self.run.project,
                        "entity": self.run.entity,
                        "url": self.run.url,
                    },
                    indent=2,
                )
            )
        except BaseException:
            self.run.finish(exit_code=1)
            raise
        return self

    def log(self, prefix, event):
        if self.run is not None:
            self.run.log(grouped_metrics(event, prefix, self.manifest.get("stage", "representation")))

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.bar is not None:
                self.bar.close()
        finally:
            if self.run is not None:
                self.run.finish(exit_code=0 if exc_type is None else 1)
