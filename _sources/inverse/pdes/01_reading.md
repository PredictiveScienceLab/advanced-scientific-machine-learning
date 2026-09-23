# PDE Inverse Problems

A PDE inverse problem uses indirect observations of a state field to infer an unknown coefficient, source, boundary condition, or initial condition. The unknown may itself be a function, so the statistical model should be defined before a mesh or basis is chosen. This subsection separates that function-space model from the finite representations used to compute with it.

## Parameter-to-observable map

Let $D\subset\mathbb{R}^d$ be the physical domain, let $m$ denote the unknown parameter field, and let $u$ denote the state. Write the governing equations schematically as

$$
\mathcal{A}(u,m)=f \quad \text{in }D,
\qquad
\mathcal{B}(u,m)=g \quad \text{on }\partial D.
$$

Let $\mathcal{X}$ be a function space for $m$, let $\mathcal{V}$ be a state space for $u$, and let $\mathcal{X}_{\mathrm{ad}}\subseteq\mathcal{X}$ contain the physically admissible parameters. Assume that the forward problem has a unique solution for every $m\in\mathcal{X}_{\mathrm{ad}}$. It then defines the solution map

$$
\mathcal{S}:\mathcal{X}_{\mathrm{ad}}\longrightarrow\mathcal{V},
\qquad
\mathcal{S}(m)=u(m).
$$

Measurements rarely contain the entire state. An observation operator $\mathcal{O}:\mathcal{V}\to\mathbb{R}^q$ may select sensor values, spatial averages, boundary fluxes, or other measured quantities. The parameter-to-observable map is

$$
\mathcal{G}=\mathcal{O}\circ\mathcal{S}:
\mathcal{X}_{\mathrm{ad}}\longrightarrow\mathbb{R}^q.
$$

For a true parameter $m^\dagger$, consider the additive Gaussian observation model

$$
\mathbf{y}=\mathcal{G}(m^\dagger)+\boldsymbol{\eta},
\qquad
\boldsymbol{\eta}\sim
\mathcal{N}(\mathbf{0},\boldsymbol{\Gamma}),
$$

where $\boldsymbol{\Gamma}\in\mathbb{R}^{q\times q}$ is a positive-definite measurement-noise covariance matrix. The corresponding data-misfit potential is

$$
\Phi(m;\mathbf{y})
=\frac{1}{2}
\left\|\boldsymbol{\Gamma}^{-1/2}
\bigl(\mathbf{y}-\mathcal{G}(m)\bigr)\right\|_2^2.
$$

This model treats the PDE and its inputs as exact. Unknown forcing, boundary data, initial conditions, or model discrepancy must be included explicitly if they are scientifically relevant. Even when the forward PDE is well posed, sparse observations and smoothing dynamics can make the inverse map nonunique or unstable.

## Function-space posterior

Suppose that $\mathcal{X}$ is a separable Hilbert space. A Gaussian prior on the unknown field is a probability measure

$$
\mu_0=\mathcal{N}(m_0,\mathcal{C}),
$$

where $m_0\in\mathcal{X}$ is the prior mean and $\mathcal{C}:\mathcal{X}\to\mathcal{X}$ is a self-adjoint, positive, trace-class covariance operator. Trace class means that the covariance eigenvalues have a finite sum, which ensures that prior draws have finite expected squared norm. Choose the prior so that $\mu_0(\mathcal{X}_{\mathrm{ad}})=1$. It should encode physical length scales, amplitudes, smoothness, and boundary behavior before the computational mesh is selected. If a coefficient must be positive, one may instead assign a Gaussian prior to an unconstrained field $z$ and set $m=\exp(z)$, provided that this transformation is well defined and scientifically appropriate.

Under standard conditions ensuring that the forward map and posterior are well defined, the posterior measure $\mu^{\mathbf{y}}$ is defined relative to the prior by

$$
\frac{d\mu^{\mathbf{y}}}{d\mu_0}(m)
=\frac{1}{Z(\mathbf{y})}
\exp\bigl[-\Phi(m;\mathbf{y})\bigr],
$$

with

$$
Z(\mathbf{y})
=\int_{\mathcal{X}}
\exp\bigl[-\Phi(m;\mathbf{y})\bigr]\,d\mu_0(m).
$$

Here $0<Z(\mathbf{y})<\infty$. The Radon--Nikodym derivative is the posterior density relative to the prior: the observations reweight the probabilities already assigned by the prior. This formulation avoids introducing a nonexistent infinite-dimensional Lebesgue density. It also makes clear that the finite-dimensional computations below approximate one fixed function-space Bayesian inverse problem {cite:p}`stuart2010inverse`.

## Karhunen--Loève truncation

Let $(\lambda_j,\varphi_j)$ be the eigenpairs of $\mathcal{C}$, ordered so that $\lambda_1\geq\lambda_2\geq\cdots\geq 0$. A draw from the Gaussian prior has the Karhunen--Loève representation

$$
m=m_0+
\sum_{j=1}^{\infty}
\sqrt{\lambda_j}\,\xi_j\varphi_j,
\qquad
\xi_j\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,1),
$$

where the series converges in mean square in $\mathcal{X}$. Retaining the first $J$ modes gives

$$
m_J(\boldsymbol{\xi})
=m_0+
\sum_{j=1}^{J}
\sqrt{\lambda_j}\,\xi_j\varphi_j,
\qquad
\boldsymbol{\xi}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_J).
$$

The mean-square prior truncation error is

$$
\mathbb{E}_{\mu_0}
\left[\left\|m-m_J\right\|_{\mathcal{X}}^2\right]
=\sum_{j>J}\lambda_j.
$$

This identity quantifies the error under the prior {cite:p}`lord2014computational`. The coefficients $\boldsymbol{\xi}$ therefore provide a finite-dimensional parameterization with a standard normal prior. The corresponding posterior density is proportional to

$$
\exp\bigl[-\Phi(m_J(\boldsymbol{\xi});\mathbf{y})\bigr]
\exp\left(-\frac{1}{2}\|\boldsymbol{\xi}\|_2^2\right).
$$

The truncation order $J$ is an approximation choice, not a physical parameter. A retained-prior-variance criterion can guide its initial selection, but it is not sufficient: a low-variance mode may still affect the observations. Posterior expectations and scientifically relevant predictions should be checked as $J$ increases {cite:p}`stuart2010inverse`.

## Numerical approximation

The exact map $\mathcal{G}$ is replaced in computation by a numerical map $\mathcal{G}_{h,\tau}$, where $h$ denotes the spatial or temporal discretization scale and $\tau$ denotes the algebraic-solver tolerance. Define the observable-space numerical error

$$
\boldsymbol{\delta}_{h,\tau}(m)
=\mathcal{G}(m)-\mathcal{G}_{h,\tau}(m).
$$

At the true parameter,

$$
\mathbf{y}-\mathcal{G}_{h,\tau}(m^\dagger)
=\boldsymbol{\eta}
+\boldsymbol{\delta}_{h,\tau}(m^\dagger).
$$

Consequently, a likelihood built with $\mathcal{G}_{h,\tau}$ can mistake numerical error for measurement noise or parameter information. The mesh should be refined and the solver tolerance tightened until posterior summaries and predictions are stable. If this is not affordable, the approximation error must be modeled and validated; it should not be absorbed into $\boldsymbol{\Gamma}$ without a defensible stochastic model {cite:p}`kaipio2006statistical,kaipio2007statistical`.

Mesh refinement and KLE refinement address different approximations. Refining $h$ improves the PDE solve for a fixed field, while increasing $J$ expands the field representation. A coherent calculation keeps the physical prior fixed as both are refined and checks convergence of posterior expectations or predictions, rather than comparing mesh-dependent parameter vectors alone {cite:p}`cotter2010approximation,stuart2010inverse`. The prior regularizes weakly informed directions, but it does not create information that the experiment did not collect.

## Coefficient and source inversion

The [thermal-conductivity example](02_thermal.ipynb) begins with a two-parameter coefficient field and shows how sparse temperature observations constrain the conductivity through a steady heat equation. The [contaminant-location example](03_contamination.ipynb) infers a low-dimensional source location and shows how sensor symmetry can create a multimodal posterior. Together, they isolate the two principal mechanisms developed here: an unknown that changes the PDE operator and an unknown that enters its forcing.
