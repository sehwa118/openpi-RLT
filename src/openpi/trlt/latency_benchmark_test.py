"""CPU tests for scripts/benchmark_attention_actor_latency.py."""

import importlib.util
import json

from openpi.trlt.config import root_path


def _load_benchmark_module():
    path = root_path("scripts/benchmark_attention_actor_latency.py")
    spec = importlib.util.spec_from_file_location("benchmark_attention_actor_latency", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_latency_benchmark_dry_run_presets(capsys):
    bench = _load_benchmark_module()
    for dataset in ("3dtacdex_assembly", "trex_squeeze"):
        rc = bench.main(["--dataset", dataset, "--device", "cpu", "--dry-run"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "plan-only"
        assert payload["dataset"] == dataset
        assert len(payload["configs"]) >= 5


def test_latency_benchmark_cpu_smoke_with_stage_a(tmp_path, capsys):
    bench = _load_benchmark_module()
    json_out = tmp_path / "latency.json"
    md_out = tmp_path / "latency.md"
    configs = [
        str(root_path("configs/trlt/3dtacdex_assembly/attention_actor/probe_mlp.json")),
        str(root_path("configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_hierarchical_d64.json")),
    ]
    rc = bench.main(
        [
            "--dataset",
            "3dtacdex_assembly",
            "--configs",
            *configs,
            "--device",
            "cpu",
            "--batch-sizes",
            "1",
            "--warmup",
            "1",
            "--iters",
            "3",
            "--include-stage-a",
            "--output",
            str(json_out),
            "--markdown-output",
            str(md_out),
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    report = json.loads(json_out.read_text())
    assert report["platform"] == "cpu"
    assert report["stage_a"] is not None
    assert "1" in report["stage_a"]["by_batch"]
    sa = report["stage_a"]["by_batch"]["1"]
    assert sa["compile_ms"] >= 0.0
    assert 0.0 < sa["device_resident"]["p50_ms"] <= sa["device_resident"]["p95_ms"] <= sa["device_resident"]["p99_ms"]

    by_variant = {row["variant"]: row for row in report["variants"]}
    assert by_variant["mlp"]["parameter_count"] == 219_970
    assert by_variant["part_attention_hierarchical_d64"]["parameter_count"] == 201_090
    for row in by_variant.values():
        b1 = row["by_batch"]["1"]
        assert b1["output_shape"] == [1, 3, 22]
        assert b1["compile_ms"] >= 0.0
        for key in ("device_resident", "host_roundtrip"):
            assert 0.0 < b1[key]["p50_ms"] <= b1[key]["p95_ms"] <= b1[key]["p99_ms"]
        assert b1["e2e_device_p50_ms"] > b1["device_resident"]["p50_ms"]
        assert 0.0 < b1["actor_share_device_percent"] < 100.0

    md_text = md_out.read_text()
    assert "`mlp`" in md_text
    assert "`part_attention_hierarchical_d64`" in md_text
    assert "Actor Share (%)" in md_text
