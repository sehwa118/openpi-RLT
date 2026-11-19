"""Selection seal rejects incomplete models, mismatched lineage and later edits."""

import json
import shutil

import pytest

from openpi.trlt.cache import write_json
from openpi.trlt.config import MODELS
from openpi.trlt.tacdex_data import seal
from openpi.trlt.tacdex_experiment import seal_selections
from openpi.trlt.tacdex_experiment import sha256
from openpi.trlt.tacdex_experiment import validate_selection_seal


def test_checkpoint_selection_seal_and_mutation(tmp_path):
    protocol = seal({"split_hash": "split", "normalization_hash": "norm", "query_manifest_hash": "queries"})
    directory = tmp_path / "protocol"
    write_json(directory / "protocol.json", protocol)
    checkpoints = tmp_path / "checkpoints"
    for mode in MODELS:
        for stage in ("representation", "probe"):
            output = checkpoints / mode / "seed0" / stage / "best"
            output.mkdir(parents=True)
            (output / "params.msgpack").write_bytes(f"{mode}_{stage}".encode())
            write_json(
                output / "metadata.json",
                {
                    "stage": stage,
                    "seed": 0,
                    "step": 1,
                    "model_config": {"mode": mode},
                    "provenance": dict(
                        {k: protocol[k] for k in ("split_hash", "normalization_hash", "query_manifest_hash")},
                        representation_checkpoint_hash=f"{mode}_representation",
                    ),
                    "params_file_hash": sha256(output / "params.msgpack"),
                    "parameter_hash": f"{mode}_{stage}",
                },
            )
    path = tmp_path / "seal.json"
    seal_selections(checkpoints, path, protocol_dir=directory)
    validate_selection_seal(path, protocol_dir=directory, checkpoint=checkpoints / MODELS[0] / "seed0/probe/best")
    assert seal_selections(checkpoints, path, protocol_dir=directory)["hash"] == json.loads(path.read_text())["hash"]
    # A depth follow-up trains just one model; other records must remain exact.
    replacement = tmp_path / "replacement"
    mode = "trlt_self_only"
    shutil.copytree(checkpoints / mode, replacement / mode)
    for stage in ("representation", "probe"):
        folder = replacement / mode / "seed0" / stage / "best"
        (folder / "params.msgpack").write_bytes(f"new_{stage}".encode())
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata["params_file_hash"] = sha256(folder / "params.msgpack")
        metadata["parameter_hash"] = f"new_{stage}"
        if stage == "probe":
            metadata["provenance"]["representation_checkpoint_hash"] = "new_representation"
        write_json(folder / "metadata.json", metadata)
    followup_path = tmp_path / "followup_seal.json"
    options = {"protocol_dir": directory, "base_selection_seal": path, "replace_models": [mode]}
    followup = seal_selections(replacement, followup_path, **options)
    base = json.loads(path.read_text())
    assert followup["base_selection_seal_hash"] == base["hash"]
    assert followup["replaced_models"] == [mode]
    for other in set(MODELS) - {mode}:
        assert followup["selections"][other] == base["selections"][other]
    validate_selection_seal(followup_path, checkpoint=replacement / mode / "seed0/probe/best")
    assert seal_selections(replacement, followup_path, **options)["hash"] == followup["hash"]
    with pytest.raises(ValueError, match="supplied together"):
        seal_selections(replacement, tmp_path / "bad.json", protocol_dir=directory, replace_models=[mode])
    with pytest.raises(ValueError, match="nonempty"):
        seal_selections(replacement, tmp_path / "bad.json", **{**options, "replace_models": []})
    (replacement / mode / "seed0/probe/best/params.msgpack").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_selection_seal(followup_path)
    with pytest.raises(ValueError, match="requires"):
        validate_selection_seal(None)
    with pytest.raises(ValueError, match="not one"):
        validate_selection_seal(path, checkpoint=tmp_path / "unselected")
    (checkpoints / MODELS[0] / "seed0/probe/best/params.msgpack").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_selection_seal(path)


def test_two_camera_extraction_cache_and_native_mapping(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from openpi.trlt import features
    from openpi.trlt.cache import FeatureCache
    from openpi.trlt.tacdex_data import CAMERA_KEYS
    from openpi.trlt.tacdex_data import CAMERAS
    from openpi.trlt.tacdex_data import DEFAULT_TEMPORAL

    class Dataset:
        queries = ((0, 0), (0, 1), (0, 3))
        camera_keys = CAMERA_KEYS
        camera_order = CAMERAS
        @property
        def provenance(self):
            return {"caption_hash": "caption"}

        def read_anchor(self, episode, anchor):
            assert episode == 0
            return {
                "images": {k: np.zeros((2, 2, 3), np.uint8) for k in CAMERA_KEYS},
                "caption": "fixture",
                "visual_timestamp": anchor / 5,
                "camera_timestamps": [anchor / 5] * 2,
            }

    class Extractor:
        calls = 0

        def __init__(self, checkpoint, *, camera_keys):
            assert camera_keys == CAMERA_KEYS
            self.tokenizer_hash = "fixture"
            self.config = SimpleNamespace(max_token_len=200)
            self.placement = {"backend": "cpu_mock"}

        def prepare_visual_context(self, anchor):
            assert tuple(anchor["images"]) == CAMERA_KEYS
            Extractor.calls += 1
            return {
                "visual_features": np.zeros((512, 2048), np.float32),
                "visual_valid_mask": np.ones(512, bool),
                "caption_tokenization_hash": "caption",
                "elapsed_seconds": 0.0,
                "input_devices": ["cpu_mock"],
                "feature_devices": ["cpu_mock"],
            }

    monkeypatch.setattr(features, "FrozenPrefixExtractor", Extractor)
    checkpoint = tmp_path / "backbone"
    checkpoint.write_bytes(b"fixture")
    options = {
        "backbone_checkpoint": checkpoint,
        "output_dir": tmp_path / "cache",
        "revision": "fixture",
        "episode_manifest_hash": "manifest",
        "temporal": DEFAULT_TEMPORAL,
    }
    plan = features.extract_visual_cache(Dataset(), **options, plan_only=True)
    assert plan["estimated_shape"] == [512, 2048]
    assert plan["anchors_requested"] == 2
    assert Extractor.calls == 0
    result = features.extract_visual_cache(Dataset(), **options)
    assert result["written"] == 2
    cache = FeatureCache(tmp_path / "cache")
    assert cache.manifest["metadata"]["camera_mapping"] == dict(zip(CAMERAS, CAMERA_KEYS, strict=True))
    assert features.extract_visual_cache(Dataset(), **options)["written"] == 0
