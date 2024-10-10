# Multifidelity modeling

We will discuss multifidelity modeling, which is a technique for constructing surrogate models that combine information from multiple sources of varying fidelity. The goal is to leverage the strengths of each fidelity level to build a more accurate and efficient surrogate model.

## Examples of Multifidelity Information

+ You could have a finite element model of a structure that is very accurate but computationally expensive. This would be your high fidelity model. You could also have a simplified analytical model that is much faster to evaluate but less accurate. This would be your low fidelity model.

+ You could have a finite element model with a fine mesh (high fidelity) and a coarse mesh (low fidelity).

+ You could have experimental data (high fidelity) and a simplified analytical model (low fidelity).

+ and so on

## Notation

Let $f_l$ denote the low-fidelity and $f_h$ denote the high-fidelity model.
The expectation is that you have a lot of data from the low-fidelity model and only a few data points from the high-fidelity model.
Suppose that our dataset is as follows:

+ $\mathbf{X}_l = \{x_{l,i}\}_{i=1}^{n_l}$ and $\mathbf{y}_l = \{y_{l,i}\}_{i=1}^{n_l}$ are the training data from the low-fidelity model.

+ $\mathbf{X}_h = \{x_{h,i}\}_{i=1}^{n_h}$ and $\mathbf{y}_h = \{y_{h,i}\}_{i=1}^{n_h}$ are the training data from the high-fidelity model.

To keep the notation simple, we will not be showing $\mathbf{X}_l$ and $\mathbf{X}_h$ in the equations below.
But whenever $\mathbf{y}_l$ or $\mathbf{y}_h$ appears, it is assumed that the corresponding $\mathbf{X}_l$ or $\mathbf{X}_h$ is also present.

## What we are after
We are after the best thing we can say about the high-fidelity model given all the data.
That is, formally we are after this:

$$
p(f_h|\mathbf{X}_l, \mathbf{y}_l, \mathbf{X}_h, \mathbf{y}_h) = \int p(f_h|f_l, \mathbf{X}_l, \mathbf{y}_l, \mathbf{X}_h, \mathbf{y}_h) p(f_l|\mathbf{X}_l, \mathbf{y}_l) df_l.
$$

To be abl to pull this off, we need two ingredients:

1. Regression for the low-fidelity model: $p(f_l|\mathbf{X}_l, \mathbf{y}_l)$.

2. A model that relates the low-fidelity and high-fidelity models: $p(f_h|f_l, \mathbf{X}_l, \mathbf{y}_l, \mathbf{X}_h, \mathbf{y}_h)$.

## Regression for the low-fidelity model

We are just going to do Gaussian process regression for the low-fidelity model.
This is the easy part.
We start with a prior:

$$
f_l \sim \operatorname{GP}(m_l, k_l).
$$

Assume that the observations are noisy:

$$
p(y_{l,i}|f_{l,i}) = \mathcal{N}(y_{l,i}|f_{l,i}, \sigma_l^2).
$$

We then condition on the data to get the posterior over $f_l$, which is also a Gaussian process:

$$
f_l|\mathbf{X}_l, \mathbf{y}_l = \operatorname{GP}(\tilde{m}_l, \tilde{k}_l).
$$

Here the posterior mean function is:

$$
\tilde{m}_l(x) = m_l(x) + k_l(x, \mathbf{X}_l) [k_l(\mathbf{X}_l, \mathbf{X}_l) + \sigma_l^2 I]^{-1} (\mathbf{y}_l - m_l(\mathbf{X}_l)),
$$

and the posterior covariance function is:

$$
\tilde{k}_l(x, x') = k_l(x, x') - k_l(x, \mathbf{X}_l) [k_l(\mathbf{X}_l, \mathbf{X}_l) + \sigma_l^2 I]^{-1} k_l(\mathbf{X}_l, x'),
$$

where as usual $I$ is the identity matrix, and $k_l(x, \mathbf{X}_l)$ is the covariance between $x$ and the training data, and $k_l(\mathbf{X}_l, \mathbf{X}_l)$ is the covariance between the training data.

## Relating the low-fidelity and high-fidelity models

The simplest way to relate the low-fidelity and the high-fidelity models is to assume that the high-fidelity model is just the low-fidelity model plus some discrepancy:

$$
f_h(x) = \rho f_l(x) + \epsilon(x),
$$

where $\rho$ is a scaling factor and $\epsilon$ is a discrepancy term.
The scaling factor is typically assumed to be a scalar, but it could also be a function of the input, e.g.,

$$
\rho(x) = \sum_{i=1}^p w_i \phi_i(x),
$$

where $w_i$ are weights and $\phi_i(x)$ are basis functions.

[Kennedy and O'Hagan, 2000](https://academic.oup.com/biomet/article/87/1/1/221217?login=true) suggest that the discrepancy term is a Gaussian process:

$$
\epsilon \sim \operatorname{GP}(m_{\epsilon}, k_{\epsilon}).
$$

They also assume that this discrepancy term is independent of the low-fidelity model, formally:

$$
p(\epsilon|f_l) = p(\epsilon).
$$

Let's work out the posterior over the high-fidelity model given the low-fidelity model.
It is just:

$$
f_h|f_l = \operatorname{GP}(\tilde{m}_h, \tilde{k}_h),
$$

where the mean function is:

$$
\tilde{m}_h(x) = \rho \tilde{m}_l(x),
$$

and the covariance function is:

$$
\tilde{k}_h(x, x') = \rho^2 \tilde{k}_l(x, x') + k_{\epsilon}(x, x').
$$

Now, let's condition on the low-fidelity data and integrate out the low-fidelity model $f_l$.
Let's start with the mean:

$$
m_{h|\mathbf{y}_l}(x) = \mathbb{E}[f_h(x)|\mathbf{y}_l] = \mathbb{E}[\rho f_l(x) + \epsilon(x)|\mathbf{y}_l] = \rho \tilde{m}_l(x).
$$

Now, let's look at the covariance:

$$
c_{h|\mathbf{y}_l}(x, x') = \mathbb{C}[f_h(x), f_h(x')|\mathbf{y}_l] = \mathbb{C}[\rho f_l(x) + \epsilon(x), \rho f_l(x') + \epsilon(x')|\mathbf{y}_l] = \rho^2 \tilde{k}_l(x, x') + k_{\epsilon}(x, x').
$$

If $\rho$ was a function of the input, then we would have:

$$
m_{h|\mathbf{y}_l}(x) = \mathbb{E}[f_h(x)|\mathbf{y}_l] = \rho(x) \tilde{m}_l(x),
$$

and

$$
c_{h|\mathbf{y}_l}(x,x') = \mathbb{C}[f_h(x), f_h(x')|\mathbf{y}_l] = \rho(x) \rho(x') \tilde{k}_l(x, x') + k_{\epsilon}(x, x').
$$

So, we conclude that the high-fidelity model given the low-fidelity data is also a Gaussian process:

$$
f_h|\mathbf{y}_l \sim \operatorname{GP}(m_{h|\mathbf{y}_l}, c_{h|\mathbf{y}_l}).
$$

We can now condition this on the high-fidelity data to get the posterior over the high-fidelity model given all the data.
We still get a Gaussian process:

$$
f_h|\mathbf{y}_l, \mathbf{y}_h \sim \operatorname{GP}(\tilde{m}_h, \tilde{k}_h).
$$

The posterior mean is:

$$
\tilde{m}_h(x) = m_{h|\mathbf{y}_l}(x) + c_{h|\mathbf{y}_l}(x, \mathbf{X}_h) [c_{h|\mathbf{y}_l}(\mathbf{X}_h, \mathbf{X}_h) + \sigma_h^2 I]^{-1} (\mathbf{y}_h - m_{h|\mathbf{y}_l}(\mathbf{X}_h)),
$$

and the posterior covariance is:

$$
\tilde{k}_h(x, x') = c_{h|\mathbf{y}_l}(x, x') - c_{h|\mathbf{y}_l}(x, \mathbf{X}_h) [c_{h|\mathbf{y}_l}(\mathbf{X}_h, \mathbf{X}_h) + \sigma_h^2 I]^{-1} c_{h|\mathbf{y}_l}(\mathbf{X}_h, x').
$$

We can fit the parameters by maximizing the marginal likelihood:

$$
\log p(\mathbf{y}_l, \mathbf{y}_h) = \log p(\mathbf{y}_l) + \log p(y_h|\mathbf{y}_l),
$$

and each one of these terms is analytically tractable.

In practice, if we have a lot more low-fidelity data, we may first maximize $\log p(\mathbf{y}_l)$ with respect to all low-fidelity model parameters, and then maximize $\log p(y_h|\mathbf{y}_l)$ with respect to the high-fidelity model parameters.

## Another approach to low-fidelity and high-fidelity models

Another approach is what we introduced in [Karumuri et al., 2023](https://www.sciencedirect.com/science/article/pii/S0927025622005626).
In this approach, assume that the high-fidelity model given the low-fidelity model is just a Gaussian process:

$$
f_h|f_l \sim \operatorname{GP}(m_h, k_{h|f_l}),
$$

where the mean function is a standard user choice, e.g., a constant, a linear function, or a polynomial,
but the covariance function is now:

$$
k_h(x, x') = k((x, f_l(x)), (x', f_l(x'))),
$$

where $k$ is a kernel function that operates on the input and the low-fidelity model.

This is essentially a deep Gaussian process with one hidden layer.
The high-fidelity covariance function does not just look at the input, but it also compares the similarity of the output low-fidelity models at the two input points.

Some work on general inference with this type of models has been done in [Damianou et al., 2013](http://proceedings.mlr.press/v31/damianou13a.pdf).
But it is not trivial to train such models.
However, if we have a lot of low-fidelity data, we can simplfiy things quite a bit.
Under the assumption that the posterior of the low-fidelity model collapses to the posterior mean, we can write:

$$
f_l \approx \tilde{m}_l(x),
$$

and then take our kernel to be:

$$
k_h(x, x') = k((x, \tilde{m}_l(x)), (x', \tilde{m}_l(x'))).
$$