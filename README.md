# sklearn-autoencoder
Denoising Autoencoder wrapper (from Theano) to sklearn (scikit-learn)

### Example


<code>da = DenoisingAutoencoder(n_hidden=10)</code>

<code>da.fit(X)</code>

<code>new_X = da.transform(X)</code>
