"""Seam 1: the estimator honours the scikit-learn estimator contract."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from sklearn_autoencoder import DenoisingAutoencoder

# float32 matmuls reorder sums with batch size, so outputs differ at ~1e-7.
FLOAT32_BATCHING = "float32 results differ at ~1e-7 across batch sizes"
EXPECTED_FAILED_CHECKS = {
    "check_methods_sample_order_invariance": FLOAT32_BATCHING,
    "check_methods_subset_invariance": FLOAT32_BATCHING,
}


@parametrize_with_checks(
    [
        DenoisingAutoencoder(max_iter=5, random_state=0, device="cpu"),
        DenoisingAutoencoder(
            hidden_layer_sizes=(8, 4), max_iter=5, random_state=0, device="cpu"
        ),
    ],
    expected_failed_checks=lambda _: EXPECTED_FAILED_CHECKS,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
