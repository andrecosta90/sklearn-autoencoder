"""Seam 3: invalid configurations and inputs fail loudly at fit time."""

import numpy as np
import pytest

from sklearn_autoencoder import DenoisingAutoencoder

X = np.random.default_rng(0).random((20, 4))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"activation": "swish"}, "activation"),
        ({"corruption": "salt"}, "corruption"),
        ({"loss": "mae"}, "loss"),
        ({"corruption": "masking", "noise_level": 1.0}, "noise_level"),
        ({"corruption": "gaussian", "noise_level": -0.1}, "noise_level"),
    ],
)
def test_invalid_parameters_raise_a_clear_error(params, message):
    with pytest.raises(ValueError, match=message):
        DenoisingAutoencoder(**params).fit(X)


def test_bce_loss_rejects_data_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DenoisingAutoencoder(loss="bce").fit(X - 0.5)


def test_early_stopping_rejects_a_validation_set_too_small_to_score():
    with pytest.raises(ValueError, match="validation_fraction"):
        DenoisingAutoencoder(early_stopping=True, validation_fraction=0.05).fit(X)
