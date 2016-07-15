# sklearn-autoencoder
Denoising Autoencoder wrapper (from Theano) to sklearn (scikit-learn)

### Example


<code>da = DenoisingAutoencoder(n_hidden=10)</code>

<code>da.fit(X)</code>

<code>new_X = da.transform(X)</code>

##### To change the dimensionality of X (in this case, changed_X will have "n_hidden" features)

<code> changed_X = da.transform_latent_representation(X)
