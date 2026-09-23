# Multi-Fidelity Modeling

High-fidelity simulations and experiments may be too expensive to evaluate at all the inputs needed for surrogate modeling. A related low-fidelity source can provide additional information at lower cost. Examples include a coarse mesh paired with a fine mesh, a reduced model paired with a full-order simulation, or an inexpensive measurement paired with a more informative experiment. The low-fidelity source need not be uniformly accurate; it is useful when its relationship to the high-fidelity response can be learned and validated {cite:p}`peherstorfer2018multifidelity`.

This section develops the two-fidelity Gaussian process construction. Recursive extensions cover nested designs with more than two fidelity levels {cite:p}`legratiet2014recursive,perdikaris2015multifidelity`.

## Data and prediction target

Let $\mathcal{X}\subseteq\mathbb{R}^d$ be the common input domain, and let

$$
f_\ell,f_h:\mathcal{X}\to\mathbb{R}
$$

denote the latent low- and high-fidelity responses. For each fidelity $s\in\{\ell,h\}$, suppose that

$$
y_{s,i}=f_s(\mathbf{x}_{s,i})+\eta_{s,i},
\qquad
\eta_{s,i}\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma_s^2),
$$

where $\mathbf{x}_{s,i}\in\mathcal{X}$. The noise variables are mutually independent and independent of the latent processes. For deterministic simulations, $\sigma_s^2$ may instead serve as a numerical nugget. Collect the inputs and observations as

$$
\mathbf{X}_s=
\begin{bmatrix}
\mathbf{x}_{s,1}^{\mathsf{T}}\\
\vdots\\
\mathbf{x}_{s,n_s}^{\mathsf{T}}
\end{bmatrix}
\in\mathbb{R}^{n_s\times d},
\qquad
\mathbf{y}_s=
\begin{bmatrix}
y_{s,1}\\
\vdots\\
y_{s,n_s}
\end{bmatrix}
\in\mathbb{R}^{n_s},
$$

and write $\mathcal{D}_s=(\mathbf{X}_s,\mathbf{y}_s)$. The designs $\mathbf{X}_\ell$ and $\mathbf{X}_h$ need not coincide. The usual regime has $n_\ell\gg n_h$ because low-fidelity evaluations are cheaper. Our target is the high-fidelity posterior

$$
p(f_h\mid\mathcal{D}_\ell,\mathcal{D}_h).
$$

All conditional distributions below treat the covariance, noise, and coupling parameters as fixed. We also assume that the covariance matrices being inverted are nonsingular. We discuss parameter estimation after deriving the posterior.

## Low-fidelity Gaussian process

Assign the low-fidelity response the prior

$$
f_\ell\sim\operatorname{GP}(m_\ell,k_\ell).
$$

Define the covariance matrix $\mathbf{K}_\ell\in\mathbb{R}^{n_\ell\times n_\ell}$ and the covariance vector $\mathbf{k}_\ell(\mathbf{x})\in\mathbb{R}^{n_\ell}$ by

$$
[\mathbf{K}_\ell]_{ij}
=k_\ell(\mathbf{x}_{\ell,i},\mathbf{x}_{\ell,j}),
\qquad
[\mathbf{k}_\ell(\mathbf{x})]_i
=k_\ell(\mathbf{x}_{\ell,i},\mathbf{x}).
$$

Let $\mathbf{m}_\ell=[m_\ell(\mathbf{x}_{\ell,i})]_{i=1}^{n_\ell}$ and

$$
\mathbf{A}_\ell=\mathbf{K}_\ell+\sigma_\ell^2\mathbf{I}_{n_\ell}.
$$

Conditioning on $\mathcal{D}_\ell$ gives

$$
f_\ell\mid\mathcal{D}_\ell
\sim\operatorname{GP}(\widetilde{m}_\ell,\widetilde{k}_\ell),
$$

where

$$
\widetilde{m}_\ell(\mathbf{x})
=m_\ell(\mathbf{x})
+\mathbf{k}_\ell(\mathbf{x})^{\mathsf{T}}
\mathbf{A}_\ell^{-1}(\mathbf{y}_\ell-\mathbf{m}_\ell)
$$

and

$$
\widetilde{k}_\ell(\mathbf{x},\mathbf{x}')
=k_\ell(\mathbf{x},\mathbf{x}')
-\mathbf{k}_\ell(\mathbf{x})^{\mathsf{T}}
\mathbf{A}_\ell^{-1}\mathbf{k}_\ell(\mathbf{x}').
$$

The covariance $\widetilde{k}_\ell$ describes the latent low-fidelity response. A future noisy low-fidelity observation has the additional variance $\sigma_\ell^2$.

## Linear autoregressive coupling

The autoregressive model of {cite:t}`kennedy2000fast` writes the high-fidelity response as

$$
f_h(\mathbf{x})=\rho f_\ell(\mathbf{x})+\delta(\mathbf{x}),
$$

where $\rho\in\mathbb{R}$ is a deterministic scale parameter and

$$
\delta\sim\operatorname{GP}(m_\delta,k_\delta)
$$

is a discrepancy process independent of $f_\ell$ and of the observation noise. Conditional on a realized low-fidelity function,

$$
f_h\mid f_\ell
\sim\operatorname{GP}
\left(\rho f_\ell+m_\delta,k_\delta\right).
$$

The low-fidelity posterior uncertainty enters after $f_\ell$ is marginalized. In distributional notation,

$$
p(f_h\mid\mathcal{D}_\ell)
=\int p(f_h\mid f_\ell)\,
p(f_\ell\mid\mathcal{D}_\ell)\,\mathrm{d}f_\ell,
$$

where the integral denotes marginalization over the low-fidelity function. Because the coupling is affine and both processes are Gaussian,

$$
f_h\mid\mathcal{D}_\ell
\sim\operatorname{GP}
\left(m_{h\mid\ell},c_{h\mid\ell}\right),
$$

with

$$
m_{h\mid\ell}(\mathbf{x})
=\rho\widetilde{m}_\ell(\mathbf{x})+m_\delta(\mathbf{x})
$$

and

$$
c_{h\mid\ell}(\mathbf{x},\mathbf{x}')
=\rho^2\widetilde{k}_\ell(\mathbf{x},\mathbf{x}')
+k_\delta(\mathbf{x},\mathbf{x}').
$$

The boundary cases expose the two information sources. If $\rho=0$, the low-fidelity data have no effect; if $m_\delta=0$ and $k_\delta=0$, the high-fidelity response is a scaled copy of the low-fidelity response.

A deterministic input-dependent scale can be written as

$$
\rho(\mathbf{x})=\sum_{j=1}^{p}w_j\phi_j(\mathbf{x}),
$$

where $\phi_1,\ldots,\phi_p$ are specified basis functions and $w_1,\ldots,w_p$ are fitted coefficients. The conditional mean and covariance then become

$$
m_{h\mid\ell}(\mathbf{x})
=\rho(\mathbf{x})\widetilde{m}_\ell(\mathbf{x})+m_\delta(\mathbf{x})
$$

and

$$
c_{h\mid\ell}(\mathbf{x},\mathbf{x}')
=\rho(\mathbf{x})\rho(\mathbf{x}')
\widetilde{k}_\ell(\mathbf{x},\mathbf{x}')
+k_\delta(\mathbf{x},\mathbf{x}').
$$

{cite:t}`legratiet2014recursive` develop input-dependent scale functions and an equivalent recursive construction for nested designs with more than two fidelity levels. {cite:t}`perdikaris2015multifidelity` combine that recursive multi-fidelity co-kriging construction with sparse Gaussian--Markov random fields.

## Conditioning on the high-fidelity data

The second stage conditions the process $f_h\mid\mathcal{D}_\ell$ on $\mathcal{D}_h$. Define

$$
[\mathbf{C}_h]_{ij}
=c_{h\mid\ell}(\mathbf{x}_{h,i},\mathbf{x}_{h,j}),
\qquad
[\mathbf{c}_h(\mathbf{x})]_i
=c_{h\mid\ell}(\mathbf{x}_{h,i},\mathbf{x}),
$$

and let

$$
\mathbf{m}_{h\mid\ell}
=\left[m_{h\mid\ell}(\mathbf{x}_{h,i})\right]_{i=1}^{n_h},
\qquad
\mathbf{A}_h=\mathbf{C}_h+\sigma_h^2\mathbf{I}_{n_h}.
$$

Gaussian conditioning gives

$$
f_h\mid\mathcal{D}_\ell,\mathcal{D}_h
\sim\operatorname{GP}(\widetilde{m}_h,\widetilde{k}_h),
$$

where

$$
\widetilde{m}_h(\mathbf{x})
=m_{h\mid\ell}(\mathbf{x})
+\mathbf{c}_h(\mathbf{x})^{\mathsf{T}}
\mathbf{A}_h^{-1}
(\mathbf{y}_h-\mathbf{m}_{h\mid\ell})
$$

and

$$
\widetilde{k}_h(\mathbf{x},\mathbf{x}')
=c_{h\mid\ell}(\mathbf{x},\mathbf{x}')
-\mathbf{c}_h(\mathbf{x})^{\mathsf{T}}
\mathbf{A}_h^{-1}\mathbf{c}_h(\mathbf{x}').
$$

The covariance $\widetilde{k}_h$ is the latent high-fidelity covariance. The predictive variance of a future noisy high-fidelity observation at $\mathbf{x}$ is $\widetilde{k}_h(\mathbf{x},\mathbf{x})+\sigma_h^2$.

This two-stage calculation is exact under the stated Gaussian model. The first step integrates the low-fidelity posterior conditioned only on $\mathcal{D}_\ell$; the second step then introduces $\mathcal{D}_h$ exactly once through Gaussian conditioning.

## Estimating the model parameters

Let $\boldsymbol{\theta}$ collect the mean, covariance, noise, and coupling parameters. The joint marginal likelihood factors exactly as

$$
\log p(\mathbf{y}_\ell,\mathbf{y}_h\mid\boldsymbol{\theta})
=\log p(\mathbf{y}_\ell\mid\boldsymbol{\theta})
+\log p(\mathbf{y}_h\mid\mathbf{y}_\ell,\boldsymbol{\theta}).
$$

The two factors are

$$
\mathbf{y}_\ell\mid\boldsymbol{\theta}
\sim\mathcal{N}(\mathbf{m}_\ell,\mathbf{A}_\ell)
$$

and

$$
\mathbf{y}_h\mid\mathbf{y}_\ell,\boldsymbol{\theta}
\sim\mathcal{N}(\mathbf{m}_{h\mid\ell},\mathbf{A}_h),
$$

so both terms are analytically tractable. Joint optimization must retain the dependence of the second term on the low-fidelity parameters. A modular alternative first fits the low-fidelity parameters, freezes them, and then fits the coupling and discrepancy parameters {cite:p}`legratiet2014recursive,perdikaris2015multifidelity`. That shortcut is not generally equivalent to joint marginal-likelihood optimization, and it prevents the high-fidelity data from refining the low-fidelity parameters. Low-fidelity data improve high-fidelity prediction only when the cross-fidelity relation can be learned; an overly flexible discrepancy can also make the scale $\rho$ weakly identified.

## Nonlinear coupling

The linear relation can be replaced by an unknown nonlinear map. Let $g$ be a Gaussian process on the augmented input space $\mathcal{X}\times\mathbb{R}$ and define

$$
f_h(\mathbf{x})=g\bigl(\mathbf{x},f_\ell(\mathbf{x})\bigr).
$$

Conditional on a realized $f_\ell$, the high-fidelity response is a Gaussian process whose covariance has the form

$$
k_{h\mid f_\ell}(\mathbf{x},\mathbf{x}')
=k_g\!\left(
(\mathbf{x},f_\ell(\mathbf{x})),
(\mathbf{x}',f_\ell(\mathbf{x}'))
\right).
$$

This nonlinear autoregressive construction was introduced for multi-fidelity information fusion by {cite:t}`perdikaris2017nonlinear`. Marginalizing the uncertain low-fidelity function through the nonlinear map generally produces a non-Gaussian distribution. The resulting deep-Gaussian-process-style composition therefore typically requires approximate inference, such as the variational methods developed by {cite:t}`damianou2013deep`.

When the low-fidelity posterior is sufficiently concentrated, a plug-in approximation replaces $f_\ell(\mathbf{x})$ by $\widetilde{m}_\ell(\mathbf{x})$ in the augmented input. This approximation produces an ordinary GP kernel,

$$
k_h(\mathbf{x},\mathbf{x}')
=k_g\!\left(
(\mathbf{x},\widetilde{m}_\ell(\mathbf{x})),
(\mathbf{x}',\widetilde{m}_\ell(\mathbf{x}'))
\right),
$$

but discards the uncertainty $\widetilde{k}_\ell$ in the low-fidelity prediction.

{cite:t}`karumuri2023hierarchical` use the same augmented-input architecture for experimental data fusion, although their two sources measure different physical quantities rather than different-fidelity approximations of one response. Here $\mathbf{x}$ contains noise-free alloy descriptors; $r(\mathbf{x})$ is latent, de-noised Vickers hardness, modeled as a Gaussian process with mean $m_r$ and covariance $k_r$; and $y(\mathbf{x})$ is latent yield strength. Using the augmented-input Gaussian process $g$ defined above, their hierarchy has the form

$$
r\sim\operatorname{GP}(m_r,k_r),
\qquad
y(\mathbf{x})=g\bigl(\mathbf{x},r(\mathbf{x})\bigr).
$$

This is a two-layer deep Gaussian process with the same compositional form as nonlinear autoregression, not a linear co-kriging model. Their practical hierarchical-GP approximation trains the second process at the first process's posterior means at the training inputs, then propagates latent-hardness uncertainty at a new input by averaging the corresponding conditional yield-strength predictions.

## Exercises

1. Derive the cross-covariance

   $$
   \operatorname{Cov}\!\left(f_\ell(\mathbf{x}),f_h(\mathbf{x}')\right)
   =\rho k_\ell(\mathbf{x},\mathbf{x}')
   $$

   for the scalar autoregressive model. Use it to construct the joint covariance matrix of $(\mathbf{y}_\ell,\mathbf{y}_h)$ and verify that one-step joint Gaussian conditioning gives the same high-fidelity posterior as the two-stage calculation.

2. Derive $m_{h\mid\ell}$ and $c_{h\mid\ell}$ for the input-dependent scale $\rho(\mathbf{x})$. Then explain which uncertainty is omitted by the nonlinear plug-in approximation.

## From coupled models to data allocation

The next example fits the nonlinear multi-fidelity Gaussian process surrogate and compares it with a Gaussian process trained only on high-fidelity observations. The active-learning section then uses coupled predictive distributions to select both the next input and the fidelity level under a computational budget.
