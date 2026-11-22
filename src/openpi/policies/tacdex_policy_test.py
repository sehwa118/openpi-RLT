import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.policies.tacdex_policy import TacdexInputs
from openpi.transforms import PadStatesAndActions


def test_mask_survives_training_preprocessing():
    """A missing hand target must survive repacking, padding and model preprocessing."""
    mask = np.ones((2, 22), dtype=bool)
    mask[1, 6:] = False
    data = TacdexInputs()(
        {
            "state": np.zeros(22),
            "actions": np.ones((2, 22)),
            "action_valid_mask": mask,
            "main": np.zeros((3, 224, 224), np.float32),
            "secondary": np.zeros((3, 224, 224), np.float32),
        }
    )
    data = PadStatesAndActions(32)(data)
    obs = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], data))
    processed = model_lib.preprocess_observation(None, obs, image_keys=tuple(obs.images))
    np.testing.assert_array_equal(processed.action_loss_mask[0, :, :22], mask)
    assert not processed.action_loss_mask[..., 22:].any()
    assert len(processed.images) == 2


def test_two_camera_spec_keeps_default_compatible():
    assert len(Pi0Config().inputs_spec()[0].images) == 3
    config = Pi0Config(pi05=True, image_keys=("base_0_rgb", "left_wrist_0_rgb"))
    assert tuple(config.inputs_spec()[0].images) == config.image_keys
    assert config.inputs_spec()[1].shape == (1, 50, 32)
