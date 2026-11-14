"""Streaming action metrics; episodes, rather than overlapping windows, are units."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

GROUPS = {
    "left_arm": list(range(7)),
    "left_hand": list(range(7, 29)),
    "right_arm": list(range(29, 36)),
    "right_hand": list(range(36, 58)),
    "arms": list(range(7)) + list(range(29, 36)),
    "hands": list(range(7, 29)) + list(range(36, 58)),
}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class ActionMetrics:
    """Accumulate unique query/offset observations, including overlapping labels.

    Each episode's normalized MSE averages valid query x horizon x joint values;
    primary averages these episode MSEs equally. Micro weights every valid element.
    """

    def __init__(self, action_mean, action_std):
        self.mean = np.asarray(action_mean, dtype=np.float64)
        self.std = np.asarray(action_std, dtype=np.float64)
        if self.mean.shape != (58,) or self.std.shape != (58,) or np.any(self.std <= 0):
            raise ValueError("Expected separate finite 58-dimensional action mean/std")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise ValueError("Non-finite normalization")
        self.episodes = {}
        self.joint_sum = np.zeros(58, np.float64)
        self.offset_sum = None
        self.offset_count = None
        self.joint_count = 0
        self.normalized_sum = 0.0
        self.raw_sum = 0.0
        self.count = 0

    def update(self, predicted, target, valid_mask, episode_ids, objects):
        predicted, target = np.asarray(predicted, np.float64), np.asarray(target, np.float64)
        valid = np.asarray(valid_mask, bool)
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 58:
            raise ValueError("Action tensors must have identical [B,Ha,58] shapes")
        if valid.shape != predicted.shape[:2]:
            raise ValueError("Horizon mask shape mismatch")
        if np.any(~np.isfinite(predicted[valid])) or np.any(~np.isfinite(target[valid])):
            raise ValueError("Non-finite valid action prediction/target")
        error = np.where(valid[..., None], predicted - target, 0.0)
        squared = error**2
        raw_squared = squared * self.std**2
        if self.offset_sum is None:
            self.offset_sum = np.zeros(predicted.shape[1], np.float64)
            self.offset_count = np.zeros(predicted.shape[1], np.int64)
        if len(self.offset_sum) != predicted.shape[1]:
            raise ValueError("Cannot mix action horizons")
        self.joint_sum += raw_squared.sum(axis=(0, 1))
        self.joint_count += int(valid.sum())
        self.offset_sum += raw_squared.sum(axis=(0, 2))
        self.offset_count += valid.sum(axis=0) * 58
        self.normalized_sum += float(squared.sum())
        self.raw_sum += float(raw_squared.sum())
        self.count += int(valid.sum()) * 58
        for i, ep in enumerate(episode_ids):
            key = str(int(ep))
            obj = str(objects[i])
            row = self.episodes.setdefault(key, {"object": obj, "normalized_sum": 0.0, "raw_sum": 0.0, "count": 0})
            if row["object"] != obj:
                raise ValueError(f"Episode {key} has conflicting object labels")
            row["normalized_sum"] += float(squared[i].sum())
            row["raw_sum"] += float(raw_squared[i].sum())
            row["count"] += int(valid[i].sum()) * 58

    def result(self):
        episodes = {
            key: {
                "object": row["object"],
                "valid_elements": row["count"],
                "normalized_mse": _ratio(row["normalized_sum"], row["count"]),
                "mse": _ratio(row["raw_sum"], row["count"]),
            }
            for key, row in sorted(self.episodes.items(), key=lambda item: int(item[0]))
        }
        by_object = defaultdict(lambda: {"normalized_sum": 0.0, "raw_sum": 0.0, "count": 0})
        for row in self.episodes.values():
            for name in ("normalized_sum", "raw_sum", "count"):
                by_object[row["object"]][name] += row[name]
        objects = {
            name: {
                "normalized_mse": _ratio(row["normalized_sum"], row["count"]),
                "mse": _ratio(row["raw_sum"], row["count"]),
                "valid_elements": row["count"],
            }
            for name, row in sorted(by_object.items())
        }
        ep_values = [row["normalized_mse"] for row in episodes.values() if row["normalized_mse"] is not None]
        raw_ep_values = [row["mse"] for row in episodes.values() if row["mse"] is not None]
        obj_values = [row["normalized_mse"] for row in objects.values() if row["normalized_mse"] is not None]
        obj_raw = [row["mse"] for row in objects.values() if row["mse"] is not None]
        raw = _ratio(self.raw_sum, self.count)
        offsets = (
            []
            if self.offset_sum is None
            else [_ratio(value, count) for value, count in zip(self.offset_sum, self.offset_count, strict=True)]
        )
        return {
            "primary_normalized_episode_macro_mse": float(np.mean(ep_values)) if ep_values else None,
            "normalized_micro_mse": _ratio(self.normalized_sum, self.count),
            "mse": raw,
            "rmse": float(np.sqrt(raw)) if raw is not None else None,
            "episode_macro_mse": float(np.mean(raw_ep_values)) if raw_ep_values else None,
            "normalized_object_macro_mse": float(np.mean(obj_values)) if obj_values else None,
            "object_macro_mse": float(np.mean(obj_raw)) if obj_raw else None,
            "group_mse": {
                name: _ratio(self.joint_sum[indices].sum(), self.joint_count * len(indices))
                for name, indices in GROUPS.items()
            },
            "joint_mse": [_ratio(value, self.joint_count) for value in self.joint_sum],
            "horizon_mse": offsets,
            "first_action_mse": offsets[0] if offsets else None,
            "episodes": episodes,
            "objects": objects,
            "valid_elements": self.count,
            "aggregation": "Episode macro averages each episode's valid query*horizon*58 MSE equally; micro includes overlapping query/offset labels.",
        }


def _summarize_seed_values(values):
    """Summarize scalar leaves while retaining each seed, including missing ones."""
    present = [value for value in values.values() if value is not None]
    if any(isinstance(value, dict) for value in present):
        keys = sorted({key for value in present if isinstance(value, dict) for key in value})
        return {
            key: _summarize_seed_values(
                {seed: value.get(key) if isinstance(value, dict) else None for seed, value in values.items()}
            )
            for key in keys
        }
    if any(isinstance(value, list) for value in present):
        length = max(len(value) for value in present if isinstance(value, list))
        return [
            _summarize_seed_values(
                {
                    seed: value[index] if isinstance(value, list) and index < len(value) else None
                    for seed, value in values.items()
                }
            )
            for index in range(length)
        ]
    clean = {
        seed: float(value)
        if isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value)
        else None
        for seed, value in values.items()
    }
    finite = [value for value in clean.values() if value is not None]
    return {
        "seeds": clean,
        "mean": float(np.mean(finite)) if finite else None,
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        "valid_seed_count": len(finite),
        "missing_expected_seeds": sorted(
            {"0", "1", "2"} - {seed for seed, value in clean.items() if value is not None}
        ),
    }


def _aggregate_metrics(metrics):
    result = {
        key: metrics[key]
        for key in (
            "primary_normalized_episode_macro_mse",
            "normalized_micro_mse",
            "mse",
            "rmse",
            "episode_macro_mse",
            "normalized_object_macro_mse",
            "object_macro_mse",
            "group_mse",
            "joint_mse",
            "horizon_mse",
            "first_action_mse",
        )
        if key in metrics
    }
    for group in ("objects", "episodes"):
        if group in metrics:
            result[group] = {
                name: {key: row.get(key) for key in ("normalized_mse", "mse")} for name, row in metrics[group].items()
            }
    return result


def summarize_results(paths, output_path=None, *, bootstrap_samples=2000, seed=42):
    """Summarize all seeds without selecting a favorable test seed/checkpoint."""
    records = [json.loads(Path(path).read_text()) for path in paths]
    by_model = defaultdict(dict)
    for row in records:
        model, training_seed = row["model"], str(row["seed"])
        if training_seed in by_model[model]:
            raise ValueError(f"Duplicate result {model}/{training_seed}")
        by_model[model][training_seed] = row
    summary = {"models": {}, "paired_vs_rlt": {}, "std_ddof": 1, "bootstrap_unit": "episode"}
    for model, rows in sorted(by_model.items()):
        values = {
            seed: row.get("metrics", {}).get("primary_normalized_episode_macro_mse") for seed, row in rows.items()
        }
        finite = [value for value in values.values() if value is not None and np.isfinite(value)]
        summary["models"][model] = {
            "seeds": values,
            "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "missing_expected_seeds": sorted({"0", "1", "2"} - set(values)),
            "metric_summary": _summarize_seed_values(
                {seed: _aggregate_metrics(row.get("metrics", {})) for seed, row in rows.items()}
            ),
        }
    baseline = by_model.get("rlt_original", {})
    rng = np.random.default_rng(seed)
    for model, rows in sorted(by_model.items()):
        if model == "rlt_original":
            continue
        pairs = {}
        for training_seed in sorted(set(rows) & set(baseline)):
            current, ref = rows[training_seed].get("metrics", {}), baseline[training_seed].get("metrics", {})
            if not current.get("episodes") or not ref.get("episodes"):
                pairs[training_seed] = {"status": "missing episode metrics"}
                continue
            if set(current["episodes"]) != set(ref["episodes"]):
                raise ValueError("Paired comparison requires identical episode IDs")
            # Manifest compatibility is checked when provenance fields are present.
            for field in (
                "split_manifest_hash",
                "query_manifest_hash",
                "normalization_hash",
                "revision",
                "dataset_revision",
                "temporal",
                "camera_order",
                "sensor_order",
                "feature_cache_hash",
            ):
                a = rows[training_seed].get("provenance", {}).get(field)
                b = baseline[training_seed].get("provenance", {}).get(field)
                if a != b:
                    raise ValueError(f"Paired comparison mismatched {field}")
            differences = []
            for ep in current["episodes"]:
                a, b = current["episodes"][ep], ref["episodes"][ep]
                if a["valid_elements"] != b["valid_elements"]:
                    raise ValueError("Paired comparison requires identical valid target counts")
                if a.get("normalized_mse") is not None and b.get("normalized_mse") is not None:
                    differences.append(a["normalized_mse"] - b["normalized_mse"])
            delta = float(np.mean(differences)) if differences else None
            denominator = ref.get("primary_normalized_episode_macro_mse")
            ci = None
            if differences and bootstrap_samples:
                values = np.asarray(differences)
                boot = values[rng.integers(len(values), size=(bootstrap_samples, len(values)))].mean(axis=1)
                ci = np.percentile(boot, (2.5, 97.5)).tolist()
            pairs[training_seed] = {
                "delta_normalized_episode_macro_mse": delta,
                "relative_change_percent": 100 * delta / denominator
                if delta is not None and denominator is not None and abs(denominator) > 1e-12
                else None,
                "paired_episode_bootstrap_95_ci": ci,
                "episode_count": len(differences),
            }
        summary["paired_vs_rlt"][model] = pairs
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
