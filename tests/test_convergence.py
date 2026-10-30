"""Seam 2 (continued): when training stops, mirroring MLPRegressor semantics."""

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from sklearn_autoencoder import DenoisingAutoencoder

X = np.random.default_rng(0).normal(size=(100, 5))


def test_stops_once_training_loss_fails_to_improve_by_tol():
    # An unreachable tol means no epoch after the first counts as an improvement,
    # so training stops when that count first exceeds n_iter_no_change.
    ae = DenoisingAutoencoder(tol=1e9, n_iter_no_change=3, random_state=0, device="cpu")

    ae.fit(X)

    assert ae.n_iter_ == 5
    assert len(ae.loss_curve_) == 5
    assert ae.best_loss_ == min(ae.loss_curve_)


def test_warns_when_max_iter_is_reached_without_converging():
    ae = DenoisingAutoencoder(
        max_iter=3, n_iter_no_change=10, random_state=0, device="cpu"
    )

    with pytest.warns(ConvergenceWarning, match="max_iter=3"):
        ae.fit(X)

    assert ae.n_iter_ == 3


def test_early_stopping_restores_the_weights_of_the_best_validation_epoch():
    params = dict(
        early_stopping=True,
        learning_rate=0.1,  # large enough for the validation score to worsen
        tol=1e9,
        n_iter_no_change=3,
        random_state=0,
        device="cpu",
    )
    ae = DenoisingAutoencoder(**params).fit(X)
    best_epoch = int(np.argmax(ae.validation_scores_)) + 1
    assert best_epoch < ae.n_iter_, "restoring the last epoch would prove nothing"
    assert ae.best_validation_score_ == max(ae.validation_scores_)

    stopped_at_best_epoch = DenoisingAutoencoder(
        **{**params, "max_iter": best_epoch}
    ).fit(X)

    np.testing.assert_allclose(ae.transform(X), stopped_at_best_epoch.transform(X))
