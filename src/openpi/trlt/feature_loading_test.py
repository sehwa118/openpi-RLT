"""CPU coverage for exact native rows, worker ownership and padded feature batches."""

from fractions import Fraction
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.trlt import features
from openpi.trlt.cache import FeatureCache
from openpi.trlt.feature_loading import anchor_loader
from openpi.trlt.tacdex_data import CAMERA_KEYS
from openpi.trlt.tacdex_data import CAMERAS
from openpi.trlt.tacdex_data import DEFAULT_TEMPORAL
from openpi.trlt.tacdex_data import AssemblyDataset
from openpi.trlt.tacdex_data import video_path


class FixtureAnchors:
    camera_keys = CAMERA_KEYS
    camera_order = CAMERAS
    queries = ((0, 0), (0, 3), (0, 6), (1, 0), (2, 0), (2, 3), (2, 6), (2, 9), (2, 12))
    provenance = None

    def read_anchor(self, episode, anchor):
        return {
            "images": {key: np.full((4, 6, 3), episode * 20 + anchor, np.uint8) for key in CAMERA_KEYS},
            "caption": f"{episode}:{anchor}",
            "visual_timestamp": anchor / 5,
            "camera_timestamps": [anchor / 5] * 2,
            "worker_environment": (os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("JAX_PLATFORMS")),
        }


def test_worker_keys_cpu_isolation_and_partial_batches(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    keys = list(FixtureAnchors.queries)
    with anchor_loader(FixtureAnchors(), keys, batch_size=4, num_workers=2, prefetch_factor=2) as loader:
        batches = list(loader)
    assert sum(map(len, batches)) == len(keys)
    actual = [tuple(key) for batch in batches for key, _ in batch]
    assert sorted(actual) == sorted(keys)
    assert len(set(actual)) == len(keys)
    assert any(len(batch) < 4 for batch in batches)
    for batch in batches:
        for _, anchor in batch:
            assert anchor["worker_environment"] == ("", "cpu")


def test_sequential_two_camera_pixels_match_seek_reader(tmp_path, monkeypatch):
    import av

    rng = np.random.default_rng(2)
    frames = {}
    for camera, shape in zip(CAMERAS, ((12, 18, 3), (10, 14, 3)), strict=True):
        values = rng.integers(0, 256, size=(9, *shape), dtype=np.uint8)
        frames[camera] = values
        path = video_path(tmp_path, 0, camera)
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as out:
            stream = out.add_stream("libx264rgb", rate=5, options={"crf": "0", "preset": "ultrafast"})
            stream.width = shape[1]
            stream.height = shape[0]
            stream.pix_fmt = "rgb24"
            for index, value in enumerate(values):
                frame = av.VideoFrame.from_ndarray(value, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 5)
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
    stamps = np.arange(9) * 0.201
    stub = SimpleNamespace(
        root=tmp_path,
        _episode=lambda episode: {"source.elapsed_seconds": stamps, "source.instruction": ["fixture"] * 9},
    )
    selected = [0, 3, 8]
    expected = [AssemblyDataset.read_anchor(stub, 0, index) for index in selected]
    original = av.open
    opens = []

    def counted(*args, **kwargs):
        opens.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(av, "open", counted)
    actual = list(AssemblyDataset.iter_anchors(stub, 0, selected))
    assert len(opens) == 2
    for (key, anchor), reference, target in zip(actual, expected, selected, strict=True):
        assert key == (0, target)
        assert anchor["visual_timestamp"] == stamps[key[1]]
        assert anchor["camera_timestamps"] == reference["camera_timestamps"]
        for camera, camera_key in zip(CAMERAS, CAMERA_KEYS, strict=True):
            np.testing.assert_array_equal(anchor["images"][camera_key], frames[camera][key[1]])
            np.testing.assert_array_equal(anchor["images"][camera_key], reference["images"][camera_key])
    with pytest.raises(ValueError, match="sorted"):
        list(AssemblyDataset.iter_anchors(stub, 0, [3, 0]))


class Tokenizer:
    def tokenize(self, caption, state=None):
        del caption, state
        return np.array([1, 2], np.int32), np.array([True, True])


def test_batch_pack_padding_and_single_anchor_equivalence():
    extractor = object.__new__(features.FrozenPrefixExtractor)
    extractor.camera_keys = CAMERA_KEYS
    extractor.config = SimpleNamespace(action_dim=32)
    extractor.tokenizer = Tokenizer()
    sizes = []

    def forward(model, observation):
        del model
        size = observation.state.shape[0]
        sizes.append(size)
        assert np.all(np.asarray(observation.state) == 0)
        assert observation.images[CAMERA_KEYS[0]].dtype == jnp.float32
        value = observation.images[CAMERA_KEYS[0]][:, 0, 0, 0]
        return jnp.broadcast_to(value[:, None, None], (size, 512, 8)), jnp.ones((size, 512), bool)

    extractor.model = None
    extractor.forward = forward
    anchors = [FixtureAnchors().read_anchor(0, index) for index in (0, 3, 6)]
    result = extractor.prepare_visual_batch(anchors, pad_to=4)
    assert sizes == [4]
    assert result["visual_features"].shape == (3, 512, 8)
    assert len(result["caption_tokenization_hashes"]) == 3
    reference = extractor.prepare_visual_context(anchors[0])
    np.testing.assert_array_equal(result["visual_features"][0], reference["visual_features"])
    assert result["caption_tokenization_hashes"][0] == reference["caption_tokenization_hash"]


@pytest.mark.parametrize("workers", [0, 2])
def test_batched_cache_counts_keys_reuse_and_identity(tmp_path, monkeypatch, workers):
    class Extractor:
        camera_keys = CAMERA_KEYS
        placement = None
        tokenizer_hash = "fixture"
        config = SimpleNamespace(max_token_len=200)

        def __init__(self, *args, **kwargs):
            pass

        def prepare_visual_batch(self, anchors, *, pad_to):
            assert pad_to == 4
            return {
                "visual_features": np.stack([np.full((512, 8), a["visual_timestamp"], np.float32) for a in anchors]),
                "visual_valid_mask": np.ones((len(anchors), 512), bool),
                "elapsed_seconds": 0.001,
                "caption_tokenization_hashes": ["fixture"] * len(anchors),
                "input_devices": ["cpu"],
                "feature_devices": ["cpu"],
            }

    monkeypatch.setattr(features, "FrozenPrefixExtractor", Extractor)
    data = FixtureAnchors()
    data.provenance = {"caption_hash": "fixture"}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    options = {
        "backbone_checkpoint": checkpoint,
        "output_dir": tmp_path / "cache",
        "revision": "fixture",
        "episode_manifest_hash": "fixture",
        "temporal": DEFAULT_TEMPORAL,
        "batch_size": 4,
        "num_workers": workers,
    }
    result = features.extract_visual_cache(data, **options)
    assert result["written"] == 9
    cache = FeatureCache(tmp_path / "cache")
    assert cache.keys() == sorted(FixtureAnchors.queries)
    assert cache.manifest["metadata"]["extraction"]["fixed_batch_size"] == 4
    assert features.extract_visual_cache(data, **options)["written"] == 0
    with pytest.raises(ValueError, match="Stale"):
        features.extract_visual_cache(data, **dict(options, batch_size=8))


def test_stable_prefix_preserves_per_example_batch1_llm_calls():
    from openpi.models.model import Observation
    from openpi.models.pi0 import Pi0

    sizes = []

    def llm(embedded, *, mask, positions):
        del mask
        sizes.append(embedded[0].shape[0])
        return [embedded[0] + positions[..., None], None], None

    def embed(observation):
        batch = observation.state.shape[0]
        return (
            jnp.arange(batch * 8 * 4, dtype=jnp.float32).reshape(batch, 8, 4),
            jnp.ones((batch, 8), bool),
            jnp.zeros(8, bool),
        )

    model = SimpleNamespace(embed_prefix=embed, PaliGemma=SimpleNamespace(llm=llm))
    observation = Observation(
        images={key: jnp.zeros((4, 224, 224, 3)) for key in CAMERA_KEYS},
        image_masks={key: jnp.ones(4, bool) for key in CAMERA_KEYS},
        state=jnp.zeros((4, 32)),
        tokenized_prompt=jnp.ones((4, 2), jnp.int32),
        tokenized_prompt_mask=jnp.ones((4, 2), bool),
    )
    reference, _ = Pi0.extract_prefix_embeddings(
        model, jax.random.key(0), observation, image_only=True, image_keys=CAMERA_KEYS
    )
    sizes.clear()
    actual, mask = Pi0.extract_prefix_embeddings(
        model, jax.random.key(0), observation, image_only=True, image_keys=CAMERA_KEYS, prefix_microbatch_size=1
    )
    np.testing.assert_array_equal(actual, reference)
    assert mask.shape == (4, 6)
    assert sizes == [1]
