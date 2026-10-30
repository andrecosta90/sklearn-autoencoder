"""PyTorch building blocks. Nothing here knows about scikit-learn."""

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


class MaskingNoise(nn.Module):
    """Zero each entry with probability ``p`` while training (Vincent et al., 2008).

    Unlike ``nn.Dropout``, surviving entries are not rescaled: the network must
    learn to reconstruct the clean input from a partially destroyed one.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        return x * (torch.rand_like(x) >= self.p)


class GaussianNoise(nn.Module):
    """Add isotropic Gaussian noise with standard deviation ``std`` while training."""

    def __init__(self, std: float) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.std == 0:
            return x
        return x + self.std * torch.randn_like(x)


def mlp(sizes: Sequence[int], activation: type[nn.Module], *, activate_last: bool):
    """Stack of ``Linear`` layers through ``sizes`` with ``activation`` in between."""
    layers: list[nn.Module] = []
    for n_in, n_out in pairwise(sizes):
        layers += [nn.Linear(n_in, n_out), activation()]
    return nn.Sequential(*(layers if activate_last else layers[:-1]))


class Autoencoder(nn.Module):
    """Corrupt -> encode -> decode. The decoder mirrors the encoder.

    One hidden size gives the classic denoising autoencoder; several give a
    stacked (deep) one, trained end to end. ``forward`` returns the raw decoder
    output (e.g. logits) for numerically stable losses; ``decode`` applies
    ``output_activation`` on top to yield reconstructions.
    """

    def __init__(
        self,
        n_features: int,
        hidden_layer_sizes: Sequence[int],
        activation: type[nn.Module],
        corruption: nn.Module,
        output_activation: nn.Module,
    ) -> None:
        super().__init__()
        sizes = [n_features, *hidden_layer_sizes]
        self.corruption = corruption
        self.encoder = mlp(sizes, activation, activate_last=True)
        self.decoder = mlp(sizes[::-1], activation, activate_last=False)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(self.corruption(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.decoder(z))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encoder(self.corruption(x)))
