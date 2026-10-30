"""Seam 2 (continued): incremental training on data streamed in chunks."""

import numpy as np
import pytest

from sklearn_autoencoder import DenoisingAutoencoder


@pytest.fixture
def low_rank_data():
    """500 samples in 20 dimensions that really live on a 3-d subspace."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(500, 3)) @ rng.normal(size=(3, 20))


def test_learns_from_data_streamed_in_chunks(low_rank_data):
    ae = DenoisingAutoencoder(random_state=0, device="cpu")
    chunks = np.array_split(low_rank_data, 5)

    for _ in range(40):
        for chunk in chunks:
            ae.partial_fit(chunk)

    reconstruction = ae.inverse_transform(ae.transform(low_rank_data))
    error = ((low_rank_data - reconstruction) ** 2).mean()
    assert error < 0.25 * low_rank_data.var(axis=0).mean()
    assert ae.n_iter_ == 200  # one pass per call


def test_partial_fit_continues_training_from_fit(low_rank_data):
    ae = DenoisingAutoencoder(max_iter=5, random_state=0, device="cpu")
    ae.fit(low_rank_data)
    codes_after_fit = ae.transform(low_rank_data)

    ae.partial_fit(low_rank_data)

    assert ae.n_iter_ == 6
    assert not np.allclose(ae.transform(low_rank_data), codes_after_fit)


def test_partial_fit_rejects_early_stopping(low_rank_data):
    with pytest.raises(ValueError, match="early_stopping"):
        DenoisingAutoencoder(early_stopping=True).partial_fit(low_rank_data)
