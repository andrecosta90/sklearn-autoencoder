"""Seam 2: the autoencoder actually learns, observed through the public API."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

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


def test_score_is_higher_for_a_better_trained_model(low_rank_data):
    barely_trained = DenoisingAutoencoder(max_iter=1, random_state=0, device="cpu")
    well_trained = DenoisingAutoencoder(max_iter=200, random_state=0, device="cpu")

    barely_trained.fit(low_rank_data)
    well_trained.fit(low_rank_data)

    assert well_trained.score(low_rank_data) > barely_trained.score(low_rank_data)


def test_grid_search_selects_hyperparameters_without_targets(low_rank_data):
    search = GridSearchCV(
        DenoisingAutoencoder(random_state=0, device="cpu"),
        param_grid={"max_iter": [1, 100]},
        cv=3,
    ).fit(low_rank_data)

    assert search.best_params_ == {"max_iter": 100}


def test_masking_autoencoder_repairs_masked_inputs(low_rank_data):
    rng = np.random.default_rng(1)
    masked = low_rank_data * (rng.random(low_rank_data.shape) >= 0.3)
    ae = DenoisingAutoencoder(
        corruption="masking", noise_level=0.3, random_state=0, device="cpu"
    ).fit(low_rank_data)

    repaired = ae.inverse_transform(ae.transform(masked))

    assert ((repaired - low_rank_data) ** 2).mean() < 0.5 * (
        (masked - low_rank_data) ** 2
    ).mean()


def test_gaussian_autoencoder_repairs_noisy_inputs(low_rank_data):
    rng = np.random.default_rng(1)
    noisy = low_rank_data + rng.normal(scale=0.5, size=low_rank_data.shape)
    ae = DenoisingAutoencoder(
        corruption="gaussian", noise_level=0.5, random_state=0, device="cpu"
    ).fit(low_rank_data)

    repaired = ae.inverse_transform(ae.transform(noisy))

    assert ((repaired - low_rank_data) ** 2).mean() < 0.5 * (
        (noisy - low_rank_data) ** 2
    ).mean()


def test_bce_loss_reconstructs_unit_interval_data_within_bounds(low_rank_data):
    probabilities = 1 / (1 + np.exp(-low_rank_data))
    ae = DenoisingAutoencoder(loss="bce", random_state=0, device="cpu")

    reconstruction = ae.fit(probabilities).inverse_transform(
        ae.transform(probabilities)
    )

    assert reconstruction.min() >= 0 and reconstruction.max() <= 1
    assert mean_squared_error_vs_mean_baseline(probabilities, reconstruction) < 0.25


def test_decodes_arbitrary_codes_back_to_input_space(low_rank_data):
    """Issue #1: map a hidden representation obtained elsewhere back to X."""
    ae = DenoisingAutoencoder(
        hidden_layer_sizes=(16, 8), random_state=0, device="cpu"
    ).fit(low_rank_data)
    codes_from_elsewhere = np.random.default_rng(1).random((5, 8))

    decoded = ae.inverse_transform(codes_from_elsewhere)

    assert decoded.shape == (5, low_rank_data.shape[1])
    with pytest.raises(ValueError, match="expects codes with 8 features"):
        ae.inverse_transform(codes_from_elsewhere[:, :4])
