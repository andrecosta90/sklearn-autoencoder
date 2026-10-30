"""Seam 2: the autoencoder actually learns, observed through the public API."""

import numpy as np
import pytest

from sklearn_autoencoder import DenoisingAutoencoder


@pytest.fixture
def low_rank_data():
    """500 samples in 20 dimensions that really live on a 3-d subspace."""
    rng = np.random.default_rng(0)
    codes = rng.normal(size=(500, 3))
    mixing = rng.normal(size=(3, 20))
    return codes @ mixing + 0.01 * rng.normal(size=(500, 20))


def mean_squared_error_vs_mean_baseline(X, reconstruction):
    """Reconstruction MSE as a fraction of the MSE of predicting the mean."""
    return ((X - reconstruction) ** 2).mean() / X.var(axis=0).mean()


@pytest.mark.parametrize("hidden_layer_sizes", [(8,), (16, 8)])
def test_reconstructs_low_rank_data_far_better_than_the_mean(
    low_rank_data, hidden_layer_sizes
):
    ae = DenoisingAutoencoder(
        hidden_layer_sizes=hidden_layer_sizes, random_state=0, device="cpu"
    ).fit(low_rank_data)

    reconstruction = ae.inverse_transform(ae.transform(low_rank_data))

    assert reconstruction.shape == low_rank_data.shape
    assert mean_squared_error_vs_mean_baseline(low_rank_data, reconstruction) < 0.25
