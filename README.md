# sklearn-autoencoder

Denoising autoencoders — classic single-layer and stacked (deep) — as
scikit-learn transformers, powered by PyTorch.

The estimator passes scikit-learn's own `check_estimator` suite, so it works
anywhere a transformer does: `Pipeline`, `GridSearchCV`, `clone`, pickling,
`get_feature_names_out`.

## Install

```bash
uv sync            # dev environment (CPU-only torch)
uv run pytest      # run the test suite
```

## Usage

```python
from sklearn_autoencoder import DenoisingAutoencoder

ae = DenoisingAutoencoder(hidden_layer_sizes=(10,), random_state=0).fit(X)

codes = ae.transform(X)                       # (n_samples, 10) latent codes
reconstruction = ae.inverse_transform(codes)  # back to input space
ae.score(X)                                   # -MSE, higher is better
```

A stacked autoencoder is just more layers; the decoder mirrors the encoder and
the whole network is trained end to end:

```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

X, y = load_digits(return_X_y=True)
model = make_pipeline(
    MinMaxScaler(),  # loss="bce" needs X in [0, 1]
    DenoisingAutoencoder(hidden_layer_sizes=(128, 32), loss="bce"),
    LogisticRegression(max_iter=1000),
).fit(X, y)
```

Because `score` needs no targets, `GridSearchCV` can tune the autoencoder on
its own reconstruction error.

### Parameters

| Parameter            | Default     | Meaning                                                     |
| -------------------- | ----------- | ----------------------------------------------------------- |
| `hidden_layer_sizes` | `(32,)`     | Encoder sizes; the last one is the code size.               |
| `activation`         | `"relu"`    | `"relu"`, `"tanh"` or `"sigmoid"`.                          |
| `corruption`         | `"masking"` | `"masking"` (zero entries) or `"gaussian"` (additive noise). |
| `noise_level`        | `0.3`       | Masking probability, or Gaussian std.                       |
| `loss`               | `"mse"`     | `"mse"`, or `"bce"` for data in [0, 1].                     |
| `learning_rate`      | `1e-3`      | Adam learning rate.                                         |
| `max_iter`           | `200`       | Training epochs.                                            |
| `batch_size`         | `64`        | Minibatch size.                                             |
| `random_state`       | `None`      | Seed for init, shuffling and corruption.                    |
| `device`             | `"auto"`    | Torch device; `"auto"` uses CUDA when available.            |

## Migrating from the Theano version

| Before (Theano)                       | Now                                    |
| ------------------------------------- | -------------------------------------- |
| `DenoisingAutoencoder(n_hidden=10)`   | `DenoisingAutoencoder(hidden_layer_sizes=(10,))` |
| `training_epochs`                     | `max_iter`                             |
| `corruption_level`                    | `noise_level`                          |
| `da.transform_latent_representation(X)` | `da.transform(X)`                    |
| `da.transform(X)` (reconstruction)    | `da.inverse_transform(da.transform(X))` |

`transform` now follows the scikit-learn convention (as in `PCA`): it maps to
the latent space, and `inverse_transform` maps back.

## References

- P. Vincent, H. Larochelle, Y. Bengio, P.-A. Manzagol. *Extracting and
  Composing Robust Features with Denoising Autoencoders*. ICML 2008.
- P. Vincent et al. *Stacked Denoising Autoencoders*. JMLR 11, 2010.
