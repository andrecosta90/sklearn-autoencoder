# sklearn-autoencoder
Denoising Autoencoder wrapper (from Theano) to sklearn (scikit-learn)

### Example


<code>da = DenoisingAutoencoder(n_hidden=10)</code>

<code>da.fit(X)</code>

<code>new_X = da.transform(X)</code>

##### To reduce the dimensionality of X (in this case, short_X will have 10 features, because n_hidden=10)

<code> short_X = da.transform_latent_representation(X)
