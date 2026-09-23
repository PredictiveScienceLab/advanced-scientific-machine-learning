# Sparse Variational Gaussian Processes

Exact Gaussian process regression factors an $n\times n$ covariance matrix.
The resulting $\mathcal{O}(n^3)$ time and $\mathcal{O}(n^2)$ storage costs become
prohibitive as the number of observations $n$ grows. A sparse variational
Gaussian process retains the Gaussian process prior and approximates its
posterior through $m\ll n$ inducing variables. Titsias introduced this
variational inducing-point construction for regression, and Hensman, Fusi, and
Lawrence developed the stochastic formulation used for minibatches
{cite:p}`titsias2009variational,hensman2013gaussian`.

## Inducing variables

Let $\mathcal{X}$ be the input space, let
$X=(\mathbf{x}_1,\ldots,\mathbf{x}_n)$ with
$\mathbf{x}_i\in\mathcal{X}$ be the training inputs, let
$\mathbf{y}=(y_1,\ldots,y_n)^{\mathsf T}$ be the observations, and place the
prior

$$
f\sim\operatorname{GP}(0,k)
$$

on the latent function, where
$k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is the covariance kernel. Define
$\mathbf{f}=(f_1,\ldots,f_n)^{\mathsf T}$, where
$f_i=f(\mathbf{x}_i)$, and assume that observations are conditionally
independent:

$$
p(\mathbf{y}\mid\mathbf{f})
=\prod_{i=1}^n p(y_i\mid f_i).
$$

Gaussian regression, for example, uses
$p(y_i\mid f_i)=\mathcal{N}(y_i\mid f_i,\sigma_n^2)$.
We write $I_s$ for the $s\times s$ identity matrix.

Choose inducing inputs
$Z=(\mathbf{z}_1,\ldots,\mathbf{z}_m)$ in the same input space as $X$ and
define the inducing variables

$$
\mathbf{u}
=
\begin{bmatrix}
f(\mathbf{z}_1)&\cdots&f(\mathbf{z}_m)
\end{bmatrix}^{\mathsf T}.
$$

Write $K_{nn}=k(X,X)$, $K_{nm}=k(X,Z)$,
$K_{mn}=K_{nm}^{\mathsf T}$, and $K_{mm}=k(Z,Z)$; for example,
$(K_{nm})_{ij}=k(\mathbf{x}_i,\mathbf{z}_j)$. These four matrices have sizes
$n\times n$, $n\times m$, $m\times n$, and $m\times m$, respectively.
Assuming that $K_{mm}$ is positive definite, the GP prior gives

$$
\begin{bmatrix}
\mathbf{f}\\
\mathbf{u}
\end{bmatrix}
\sim
\mathcal{N}\!\left(
\mathbf{0},
\begin{bmatrix}
K_{nn} & K_{nm}\\
K_{mn} & K_{mm}
\end{bmatrix}
\right).
$$

Gaussian conditioning then gives

$$
p(\mathbf{f}\mid\mathbf{u})
=\mathcal{N}\!\left(
K_{nm}K_{mm}^{-1}\mathbf{u},
K_{nn}-K_{nm}K_{mm}^{-1}K_{mn}
\right).
$$

In particular, the marginal prior of the inducing variables is
$p(\mathbf{u})=\mathcal{N}(\mathbf{0},K_{mm})$.

The inducing variables describe the part of the latent function represented by
the selected locations. The conditional covariance retains the uncertainty
that remains after their values are known. The formulas use inverse notation;
implementations use a Cholesky factorization and linear solves, usually with a
small diagonal jitter for numerical stability.

## Variational posterior

We approximate the posterior over the inducing variables with

$$
q(\mathbf{u})
=\mathcal{N}(\mathbf{u}\mid\boldsymbol{\mu}_u,\Sigma_u),
$$

where $\boldsymbol{\mu}_u\in\mathbb{R}^m$ and the symmetric
positive-definite matrix $\Sigma_u\in\mathbb{R}^{m\times m}$ are variational
parameters. The joint approximation preserves the exact GP conditional:

$$
q(\mathbf{f},\mathbf{u})
=p(\mathbf{f}\mid\mathbf{u})q(\mathbf{u}).
$$

Marginalizing $\mathbf{u}$ gives a Gaussian distribution over the training
values with

$$
\begin{aligned}
\mathbb{E}_{q}[\mathbf{f}]
&=K_{nm}K_{mm}^{-1}\boldsymbol{\mu}_u,\\
\operatorname{Cov}_{q}(\mathbf{f})
&=K_{nn}-K_{nm}K_{mm}^{-1}K_{mn}
  +K_{nm}K_{mm}^{-1}\Sigma_u K_{mm}^{-1}K_{mn}.
\end{aligned}
$$

The same construction gives predictions. For test inputs
$X_*=(\mathbf{x}_{*1},\ldots,\mathbf{x}_{*r})$, let
$\mathbf{f}_*=(f(\mathbf{x}_{*1}),\ldots,f(\mathbf{x}_{*r}))^{\mathsf T}$ and
define $K_{*m}=k(X_*,Z)$, $K_{m*}=K_{*m}^{\mathsf T}$, and
$K_{**}=k(X_*,X_*)$. Then

$$
\begin{aligned}
\mathbb{E}_{q}[\mathbf{f}_*]
&=K_{*m}K_{mm}^{-1}\boldsymbol{\mu}_u,\\
\operatorname{Cov}_{q}(\mathbf{f}_*)
&=K_{**}-K_{*m}K_{mm}^{-1}K_{m*}
  +K_{*m}K_{mm}^{-1}\Sigma_u K_{mm}^{-1}K_{m*}.
\end{aligned}
$$

These expressions describe the latent function. For Gaussian regression, the
predictive distribution of a noisy observation adds $\sigma_n^2 I_r$ to the
latent covariance.

## Evidence lower bound

The Kullback--Leibler divergence from a density $q$ to a density $p$ is

$$
\operatorname{KL}(q\,\|\,p)
=\mathbb{E}_{q}\!\left[\log\frac{q}{p}\right]\geq 0.
$$

Using the factorization
$p(\mathbf{y},\mathbf{f},\mathbf{u})
=p(\mathbf{y}\mid\mathbf{f})p(\mathbf{f}\mid\mathbf{u})p(\mathbf{u})$
and applying Jensen's inequality with
$q(\mathbf{f},\mathbf{u})$ gives the evidence lower bound (ELBO)

$$
\begin{aligned}
\log p(\mathbf{y})
&\geq
\mathbb{E}_{q(\mathbf{f},\mathbf{u})}
\!\left[
\log p(\mathbf{y}\mid\mathbf{f})
+\log p(\mathbf{u})
-\log q(\mathbf{u})
\right]\\
&=
\sum_{i=1}^n
\mathbb{E}_{q(f_i)}\!\left[\log p(y_i\mid f_i)\right]
-\operatorname{KL}\!\left(q(\mathbf{u})\,\|\,p(\mathbf{u})\right)
=:\mathcal{L},
\end{aligned}
$$

where $q(f_i)$ is the one-dimensional marginal of $q(\mathbf{f})$. For fixed
model parameters and inducing inputs, the gap satisfies

$$
\log p(\mathbf{y})-\mathcal{L}
=\operatorname{KL}\!\left(
q(\mathbf{f},\mathbf{u})
\,\|\,
p(\mathbf{f},\mathbf{u}\mid\mathbf{y})
\right).
$$

Maximizing the ELBO therefore brings the approximate joint posterior closer to
the exact joint posterior in this direction of Kullback--Leibler divergence.

## Gaussian regression

The Gaussian likelihood permits an analytic optimization over the variational
distribution. Define

$$
H=K_{nm}K_{mm}^{-1},
\qquad
Q_{nn}=K_{nm}K_{mm}^{-1}K_{mn}.
$$

For fixed kernel parameters and inducing inputs, the maximizing Gaussian has

$$
\begin{aligned}
\Sigma_{u,\mathrm{opt}}^{-1}
&=K_{mm}^{-1}+\sigma_n^{-2}H^{\mathsf T}H,\\
\boldsymbol{\mu}_{u,\mathrm{opt}}
&=\sigma_n^{-2}\Sigma_{u,\mathrm{opt}}H^{\mathsf T}\mathbf{y}.
\end{aligned}
$$

Substituting this distribution into the ELBO gives the collapsed Titsias bound
{cite:p}`titsias2009variational`:

$$
\max_{\boldsymbol{\mu}_u,\Sigma_u}\mathcal{L}
=
\log\mathcal{N}\!\left(
\mathbf{y}\mid\mathbf{0},Q_{nn}+\sigma_n^2I_n
\right)
-\frac{1}{2\sigma_n^2}
\operatorname{tr}(K_{nn}-Q_{nn}).
$$

The trace penalty measures the prior variance not represented by the inducing
variables. When $m=n$ and $Z=X$, we have $Q_{nn}=K_{nn}$, the penalty vanishes,
and the bound recovers exact Gaussian process regression without a
computational saving. Evaluating the collapsed bound for all observations
costs $\mathcal{O}(nm^2+m^3)$ time with dense matrices.

For a non-Gaussian likelihood, the optimal Gaussian $q(\mathbf{u})$ is
generally unavailable in closed form, and the expected log likelihood can be
evaluated with one-dimensional numerical quadrature
{cite:p}`hensman2015scalable`. Retaining
$\boldsymbol{\mu}_u$ and $\Sigma_u$ as explicit parameters also produces the
separable objective needed for stochastic optimization
{cite:p}`hensman2013gaussian`.

## Minibatches and computational cost

Let $B$ be a uniformly sampled minibatch of $b$ observations. Replacing the
data term by its scaled minibatch estimate gives

$$
\widehat{\mathcal{L}}_B
=\frac{n}{b}\sum_{i\in B}
\mathbb{E}_{q(f_i)}\!\left[\log p(y_i\mid f_i)\right]
-\operatorname{KL}\!\left(q(\mathbf{u})\,\|\,p(\mathbf{u})\right).
$$

The estimator satisfies
$\mathbb{E}_B[\widehat{\mathcal{L}}_B]=\mathcal{L}$. The factor $n/b$
scales only the sampled data term; the global Kullback--Leibler term appears
once.

With dense matrices, a typical update costs
$\mathcal{O}(m^3+bm^2)$ time and stores
$\mathcal{O}(m^2+bm)$ working quantities, in addition to the data set
{cite:p}`sun2021scalable`. The
cubic term comes from operations on the $m\times m$ inducing covariance or
variational covariance, while the batch term evaluates the required marginal
means and variances. These working costs depend on $n$ only through the chosen
minibatch size. The variational parameters, inducing locations, kernel
parameters, and likelihood parameters can be optimized together.

The number and placement of inducing inputs determine the tradeoff
between approximation and cost. Increasing $m$ generally gives a more flexible
approximation and increases the computational cost. Independent predictive
diagnostics remain necessary because a high ELBO alone does not establish that
the surrogate is accurate for its intended scientific use.

The following example applies the sparse variational construction to the
autoinjector data and evaluates its predictions with the independent
predictive diagnostics introduced in the surrogate-diagnostics section.

## Exercise

Use the laws of total expectation and total covariance to derive the mean and
covariance of $q(\mathbf{f})$. Then verify that the scaled minibatch objective
is unbiased and explain why its Kullback--Leibler term is not multiplied by
$n/b$. For Gaussian regression, examine the limiting choice $m=n$ and $Z=X$.
