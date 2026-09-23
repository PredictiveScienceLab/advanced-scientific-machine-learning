# ODE Inverse Problems

An ODE inverse problem uses time-resolved observations to infer unknown physical parameters, initial conditions, or finite-dimensional forcing coefficients. The ODE solve maps each candidate parameter vector to a trajectory, and an observation model maps that trajectory to predicted measurements. This section formulates the calibration of deterministic ODE models and identifies the conditions that determine what the data can recover.

## Parameter-to-observable map

Let $p$ and $d$ be positive integers, let $T>0$, and let $\boldsymbol{\theta}\in\Theta\subseteq\mathbb{R}^p$ denote the unknown parameter vector. Let $\mathbf{f}:\mathbb{R}^d\times[0,T]\times\Theta\to\mathbb{R}^d$ denote the vector field, and let $\mathbf{x}_0:\Theta\to\mathbb{R}^d$ specify the initial state. On the time interval $[0,T]$, consider

$$
\frac{d\mathbf{x}}{dt}
= \mathbf{f}(\mathbf{x},t;\boldsymbol{\theta}),
\qquad
\mathbf{x}(0)=\mathbf{x}_0(\boldsymbol{\theta}),
$$

where $\mathbf{x}(t;\boldsymbol{\theta})\in\mathbb{R}^d$ is the state. The vector $\boldsymbol{\theta}$ may include coefficients in $\mathbf{f}$, unknown components of the initial state, or coefficients in a prescribed forcing representation. We assume that the initial-value problem has a unique solution for every $\boldsymbol{\theta}\in\Theta$. The solution operator is therefore well defined:

$$
\mathcal{S}:\Theta\longrightarrow C([0,T];\mathbb{R}^d),
\qquad
\mathcal{S}(\boldsymbol{\theta})
=\mathbf{x}(\cdot;\boldsymbol{\theta}).
$$

Here $C([0,T];\mathbb{R}^d)$ is the space of continuous state trajectories on $[0,T]$.

Let $m$ and $n$ be positive integers. Suppose that measurements are taken at times $0\leq t_1<\cdots<t_n\leq T$ through an observation map $\mathbf{h}:\mathbb{R}^d\to\mathbb{R}^m$. The parameter-to-observable map is

$$
\mathcal{G}:\Theta\longrightarrow\mathbb{R}^{nm},
\qquad
\mathcal{G}(\boldsymbol{\theta})
=
\begin{bmatrix}
\mathbf{h}(\mathbf{x}(t_1;\boldsymbol{\theta}))\\
\vdots\\
\mathbf{h}(\mathbf{x}(t_n;\boldsymbol{\theta}))
\end{bmatrix}.
$$

The map $\mathbf{h}$ may select observed state components or represent a nonlinear sensor. The inverse problem seeks to recover $\boldsymbol{\theta}$ from noisy data near $\mathcal{G}(\boldsymbol{\theta})$.

## Deterministic and Bayesian calibration

Let $\mathbf{y}_k\in\mathbb{R}^m$ denote the observation at time $t_k$. A common measurement model is

$$
\mathbf{y}_k
=\mathbf{h}(\mathbf{x}(t_k;\boldsymbol{\theta}))
+\boldsymbol{\varepsilon}_k,
\qquad
\boldsymbol{\varepsilon}_k
\stackrel{\mathrm{iid}}{\sim}\mathcal{N}(\mathbf{0},\boldsymbol{\Gamma}),
$$

where the known covariance matrix $\boldsymbol{\Gamma}\in\mathbb{R}^{m\times m}$ is positive definite. Here $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Gamma})$ denotes a normal distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Gamma}$. The formulas below treat $\boldsymbol{\Gamma}$ as known. If a noise scale is inferred as part of $\boldsymbol{\theta}$, the likelihood must retain its parameter-dependent Gaussian normalization, including the log-determinant term. Write $\mathbf{y}_{1:n}=(\mathbf{y}_1,\ldots,\mathbf{y}_n)$ and define the weighted least-squares objective

$$
\Phi(\boldsymbol{\theta};\mathbf{y}_{1:n})
=\frac{1}{2}\sum_{k=1}^n
\left\|\mathbf{y}_k-
\mathbf{h}(\mathbf{x}(t_k;\boldsymbol{\theta}))
\right\|_{\boldsymbol{\Gamma}^{-1}}^2,
\qquad
\|\mathbf{r}\|_{\boldsymbol{\Gamma}^{-1}}^2
=\mathbf{r}^{\mathsf T}\boldsymbol{\Gamma}^{-1}\mathbf{r}.
$$

Conditional independence then gives

$$
p(\mathbf{y}_{1:n}\mid\boldsymbol{\theta})
\propto
\exp\!\left[-\Phi(\boldsymbol{\theta};\mathbf{y}_{1:n})\right].
$$

Maximum-likelihood calibration minimizes $\Phi$ over $\Theta$. Let $p_0(\boldsymbol{\theta})$ be a prior density and define

$$
Z(\mathbf{y}_{1:n})
=\int_{\Theta}
\exp\!\left[-\Phi(\boldsymbol{\vartheta};\mathbf{y}_{1:n})\right]
p_0(\boldsymbol{\vartheta})\,d\boldsymbol{\vartheta}.
$$

When $Z(\mathbf{y}_{1:n})$ is finite and positive, Bayes' rule gives

$$
p(\boldsymbol{\theta}\mid\mathbf{y}_{1:n})
=\frac{1}{Z(\mathbf{y}_{1:n})}
\exp\!\left[-\Phi(\boldsymbol{\theta};\mathbf{y}_{1:n})\right]
p_0(\boldsymbol{\theta}).
$$

The maximum a posteriori estimate minimizes $\Phi(\boldsymbol{\theta};\mathbf{y}_{1:n})-\log p_0(\boldsymbol{\theta})$. These optimization and Bayesian formulations are two views of the same forward and observation models {cite:p}`kaipio2006statistical,stuart2010inverse`.

This formulation treats the ODE as deterministic and exact. It includes measurement noise but excludes process noise and model discrepancy. Those effects require a stochastic state model or an explicit discrepancy term.

## Parameter sensitivities

The observation Jacobian

$$
\mathbf{J}(\boldsymbol{\theta})
=D_{\boldsymbol{\theta}}\mathcal{G}(\boldsymbol{\theta})
\in\mathbb{R}^{nm\times p}
$$

measures how the predicted data change with the parameters. When the required derivatives exist, define $\mathbf{f}_{\mathbf{x}}=\partial\mathbf{f}/\partial\mathbf{x}$ and $\mathbf{f}_{\boldsymbol{\theta}}=\partial\mathbf{f}/\partial\boldsymbol{\theta}$ along the trajectory. The state sensitivity matrix $\mathbf{S}(t)=\partial\mathbf{x}(t;\boldsymbol{\theta})/\partial\boldsymbol{\theta}\in\mathbb{R}^{d\times p}$ then satisfies

$$
\frac{d\mathbf{S}}{dt}
=\mathbf{f}_{\mathbf{x}}\mathbf{S}
+\mathbf{f}_{\boldsymbol{\theta}},
\qquad
\mathbf{S}(0)
=\frac{\partial\mathbf{x}_0}{\partial\boldsymbol{\theta}}.
$$

The earlier discussion of [ODE sensitivities](../../up/sensitivity_analysis/02_diff_ode.md) develops this calculation in detail.

If $D\mathbf{h}$ denotes the Jacobian of the observation map, the chain rule gives

$$
\mathbf{J}(\boldsymbol{\theta})
=
\begin{bmatrix}
D\mathbf{h}(\mathbf{x}(t_1;\boldsymbol{\theta}))\mathbf{S}(t_1)\\
\vdots\\
D\mathbf{h}(\mathbf{x}(t_n;\boldsymbol{\theta}))\mathbf{S}(t_n)
\end{bmatrix}.
$$

For the Gaussian observation model, let $\mathbf{I}_n$ be the $n\times n$ identity matrix and define the block covariance $\boldsymbol{\Gamma}_n=\mathbf{I}_n\otimes\boldsymbol{\Gamma}$, where $\otimes$ denotes the Kronecker product. The local information matrix is

$$
\mathbf{J}(\boldsymbol{\theta})^{\mathsf T}
\boldsymbol{\Gamma}_n^{-1}
\mathbf{J}(\boldsymbol{\theta}).
$$

Within a fixed parameterization, small eigenvalues indicate parameter combinations that weakly affect the available measurements. Rank deficiency exposes a first-order confounding direction, although it does not by itself prove global nonidentifiability.

## Structural and practical identifiability

Identifiability is relative to the ODE, observation map, known inputs, initial conditions, and experimental design. A parameterization is globally structurally identifiable when two parameter values that produce the same ideal, noise-free observed trajectory must be equal. Local structural identifiability requires this implication only within a neighborhood of the true parameter. Practical identifiability asks whether the finite, noisy observations constrain the parameters tightly enough for the intended inference {cite:p}`raue2009identifiability`.

Exact equality of ideal observed trajectories for one pair of distinct parameters establishes failure of global structural identifiability. Structural symmetries and finite, noisy data can both produce shallow objective directions, posterior ridges or modes, unstable estimates, and sensitivity to the prior. These numerical features reveal difficult inference geometry, but they do not by themselves establish the structural result.

## Information from multiple trajectories

Suppose $R\geq 1$ experiments share the same $\boldsymbol{\theta}$ but use different known initial conditions, forcings, observation maps, or sampling times. For experiment $r$, let $n_r$ be the number of observations, let $\boldsymbol{\Gamma}_r$ be their common measurement covariance, let $\mathbf{I}_{n_r}$ be the $n_r\times n_r$ identity matrix, and define $\boldsymbol{\Gamma}_{n_r,r}=\mathbf{I}_{n_r}\otimes\boldsymbol{\Gamma}_r$. Let $\mathcal{G}_r$ and $\mathbf{J}_r$ denote the observable map and its Jacobian. Conditionally independent experiments give a product likelihood, and their local information contributions add:

$$
\sum_{r=1}^R
\mathbf{J}_r^{\mathsf T}
\boldsymbol{\Gamma}_{n_r,r}^{-1}
\mathbf{J}_r.
$$

New sensitivity directions are required to remove a local rank deficiency. Repeating the same informative experiment can still improve precision along the directions it already probes, but it cannot resolve a structural parameter confounding.

Trajectories generated by systems with different parameter values serve a different purpose: they can train an amortized inference rule, but they do not form a joint likelihood for one shared parameter vector. The later [companion Duffing-system notebook](03_dyn_system_multiple_traj.ipynb) uses this second construction.

## Prior information

A prior can enforce physical constraints, regularize weakly informed directions, and connect related experiments through a hierarchical model. It cannot make the likelihood structurally identifiable. If the likelihood is constant along a parameter direction, the posterior behavior in that direction is determined by the prior, and the resulting inference should be reported as prior-sensitive.

The next [harmonic-oscillator example](02_identifiability.ipynb) makes structural confounding explicit: the ideal trajectory depends on the mass and stiffness only through their ratio. Its posterior ridge is a computational consequence of that analytic fact. The companion Duffing-system notebook then shows how a collection of simulated parameter--trajectory pairs can train an amortized map for rapid inference on a new trajectory.
