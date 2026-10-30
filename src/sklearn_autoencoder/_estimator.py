"""scikit-learn adapter around :class:`~sklearn_autoencoder._network.Autoencoder`."""

from collections.abc import Callable

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_is_fitted,
    validate_data,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._network import Autoencoder, GaussianNoise, MaskingNoise

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
CORRUPTIONS = {"masking": MaskingNoise, "gaussian": GaussianNoise}
# loss name -> (criterion on raw decoder output, activation giving reconstructions)
LOSSES = {
    "mse": (nn.MSELoss, nn.Identity),
    "bce": (nn.BCEWithLogitsLoss, nn.Sigmoid),
}


class DenoisingAutoencoder(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Denoising autoencoder as a scikit-learn transformer.

    ``transform`` maps samples to the latent code (the last hidden layer) and
    ``inverse_transform`` maps codes back to input space, as in ``PCA``.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(32,)
        Encoder layer sizes; the decoder mirrors them. A single entry is the
        classic denoising autoencoder, several entries a stacked one.
    activation : {"relu", "tanh", "sigmoid"}, default="relu"
        Non-linearity after every hidden layer, including the code.
    corruption : {"masking", "gaussian"}, default="masking"
        How inputs are corrupted during training.
    noise_level : float, default=0.3
        Corruption strength: masking probability, or Gaussian noise std.
    loss : {"mse", "bce"}, default="mse"
        Reconstruction loss. ``"bce"`` (binary cross-entropy, as in Vincent et
        al., 2008) requires ``X`` in [0, 1] and yields reconstructions in [0, 1].
    learning_rate : float, default=1e-3
        Adam learning rate.
    max_iter : int, default=200
        Number of training epochs.
    batch_size : int, default=64
        Minibatch size (clipped to the number of samples).
    random_state : int, RandomState instance or None, default=None
        Controls weight initialisation, shuffling and corruption.
    device : str, default="auto"
        Torch device; ``"auto"`` picks CUDA when available.
    verbose : bool, default=False
        Print the training loss after every epoch.

    Attributes
    ----------
    module_ : torch.nn.Module
        The trained network.
    loss_curve_ : list of float
        Mean training loss per epoch.
    n_iter_ : int
        Number of epochs run.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(
        self,
        hidden_layer_sizes=(32,),
        activation="relu",
        corruption="masking",
        noise_level=0.3,
        loss="mse",
        learning_rate=1e-3,
        max_iter=200,
        batch_size=64,
        random_state=None,
        device="auto",
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.corruption = corruption
        self.noise_level = noise_level
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=FLOAT_DTYPES)
        activation = _lookup(ACTIVATIONS, "activation", self.activation)
        corruption = _lookup(CORRUPTIONS, "corruption", self.corruption)
        criterion, output_activation = _lookup(LOSSES, "loss", self.loss)
        self._check_noise_level()
        if self.loss == "bce" and (X.min() < 0 or X.max() > 1):
            raise ValueError("loss='bce' requires X to lie in [0, 1].")

        self.device_ = _resolve_device(self.device)
        seed = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)

        # Seed torch locally so fitting never touches the caller's global RNG.
        forked = [self.device_] if self.device_.type == "cuda" else []
        with torch.random.fork_rng(devices=forked):
            torch.manual_seed(seed)
            self.module_ = Autoencoder(
                n_features=X.shape[1],
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=activation,
                corruption=corruption(self.noise_level),
                output_activation=output_activation(),
            ).to(self.device_)
            self.loss_curve_ = self._train(self._to_tensor(X), criterion())
        self.n_iter_ = len(self.loss_curve_)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=FLOAT_DTYPES, reset=False)
        return self._predict(self.module_.encoder, X)

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        if X.shape[1] != self._n_features_out:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"expects codes with {self._n_features_out} features."
            )
        return self._predict(self.module_.decode, X)

    def score(self, X, y=None):
        """Negative mean squared reconstruction error (higher is better)."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=FLOAT_DTYPES, reset=False)
        return -float(((X - self._predict(self.module_.reconstruct, X)) ** 2).mean())

    def _check_noise_level(self):
        if self.corruption == "masking" and not 0 <= self.noise_level < 1:
            raise ValueError(
                "noise_level is a masking probability and must lie in [0, 1), "
                f"got {self.noise_level}."
            )
        if self.corruption == "gaussian" and self.noise_level < 0:
            raise ValueError(
                "noise_level is a noise std and must be non-negative, "
                f"got {self.noise_level}."
            )

    def _train(self, X: torch.Tensor, criterion: nn.Module) -> list[float]:
        loader = DataLoader(
            TensorDataset(X), batch_size=min(self.batch_size, len(X)), shuffle=True
        )
        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.learning_rate)
        loss_curve = []

        self.module_.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = criterion(self.module_(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            loss_curve.append(epoch_loss / len(X))
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_iter}, loss: {loss_curve[-1]:.6f}")
        self.module_.eval()
        return loss_curve

    def _predict(self, network: Callable[[torch.Tensor], torch.Tensor], X):
        with torch.inference_mode():
            out = network(self._to_tensor(X)).cpu().numpy()
        return out.astype(X.dtype, copy=False)

    def _to_tensor(self, X) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32, device=self.device_)

    @property
    def _n_features_out(self):
        return self.hidden_layer_sizes[-1]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


def _lookup(options: dict, name: str, value):
    try:
        return options[value]
    except KeyError:
        raise ValueError(
            f"{name} must be one of {sorted(options)}, got {value!r}."
        ) from None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
