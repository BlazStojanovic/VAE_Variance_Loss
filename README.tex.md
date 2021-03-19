Variational Autoencoder with Variance Loss
==========================================

Implementation of variance loss 
$$
\underset{z \sim r(z)}{\operatorname{Var}}\left[\log \left(\frac{p_{\theta}(x \mid z) p(z)}{q_{\phi}(z \mid x)}\right)\right]
$$
for Variational Autoencoders. Tested on standard continuous datasets (Fashion MNIST, Faces, etc.) and discrete datasets (discretized MNIST). 