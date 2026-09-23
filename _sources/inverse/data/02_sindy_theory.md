# Sparse Identification of Nonlinear Dynamics

Sparse identification of nonlinear dynamics (SINDy) seeks an interpretable differential equation from observed trajectories {cite:p}`brunton2016discovering`. The method is most useful when the state is observed and the dynamics are approximately deterministic and Markovian, meaning that the current state contains the information needed to determine its next change. Scientific knowledge should also suggest a manageable collection of candidate terms.

## From trajectories to sparse regression

Consider an autonomous dynamical system with state $\mathbf{x}(t) \in \mathbb{R}^d$,

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t)).
$$

Suppose that the state and its time derivative are available at times $t_1,\ldots,t_N$. We arrange the samples by row in the matrices

$$
\mathbf{X} =
\begin{bmatrix}
\mathbf{x}(t_1)^T \\
\vdots \\
\mathbf{x}(t_N)^T
\end{bmatrix},
\qquad
\dot{\mathbf{X}} =
\begin{bmatrix}
\dot{\mathbf{x}}(t_1)^T \\
\vdots \\
\dot{\mathbf{x}}(t_N)^T
\end{bmatrix}.
$$

SINDy begins with candidate scalar functions $\phi_1,\ldots,\phi_p$. Typical libraries contain a constant, low-order monomials, trigonometric functions, or terms required by the governing physics. Evaluating the library along the trajectory gives the design matrix

$$
\boldsymbol{\Phi}(\mathbf{X})_{nj} = \phi_j(\mathbf{x}(t_n)).
$$

The model assumes that each component of the vector field uses only a few library terms. With a coefficient matrix $\mathbf{C} \in \mathbb{R}^{p\times d}$, the regression problem is

$$
\dot{\mathbf{X}} \approx \boldsymbol{\Phi}(\mathbf{X})\mathbf{C}.
$$

One possible sparse estimator minimizes

$$
\left\|\dot{\mathbf{X}}-\boldsymbol{\Phi}(\mathbf{X})\mathbf{C}\right\|_F^2
+ \lambda\sum_{j,k}|C_{jk}|,
$$

where $\lambda>0$ controls the tradeoff between derivative fit and sparsity. Sequential thresholded least squares is another common choice. The fitted vector field is

$$
\widehat{\mathbf{f}}(\mathbf{x})
= \mathbf{C}^T\boldsymbol{\phi}(\mathbf{x}),
\qquad
\boldsymbol{\phi}(\mathbf{x})
= \begin{bmatrix}\phi_1(\mathbf{x})&\cdots&\phi_p(\mathbf{x})\end{bmatrix}^T.
$$

The regression residual alone does not validate the discovered dynamics. The fitted differential equation should also be integrated from withheld initial conditions and compared with withheld trajectories or scientifically relevant long-time statistics. The following sections apply this workflow to [linear and polynomial systems](02_sindy_1.ipynb) and the [Lorenz system](03_sindy_2.ipynb).

## Scope of the basic formulation

Plain SINDy relies on four substantive conditions. First, the measurements must determine the state whose dynamics are being modeled. If $\mathbf{y}=\mathbf{g}(\mathbf{x})$ and $\mathbf{g}$ is not invertible, a closed equation $\dot{\mathbf{y}}=\widetilde{\mathbf{f}}(\mathbf{y})$ need not exist. Second, the candidate library must contain a sufficiently accurate representation of the active terms. Third, the observed trajectories must excite those terms well enough to distinguish their effects. Fourth, the time scale and sampling rate must resolve the dynamics of interest.

These conditions separate sparse equation discovery from ordinary curve fitting. A sparse coefficient vector can still describe the wrong vector field when the library is inadequate, the sampled region is too small, or several library columns are nearly indistinguishable along the observed trajectories.

## Derivatives and noisy states

Direct derivative measurements are uncommon. A finite-difference estimate is simple, but differentiation amplifies high-frequency measurement noise. Smoothing the state first and differentiating the smoother can be more stable. Local polynomial fits and Gaussian process regression are two possible smoothers; their length scales must retain the physical time scales that the model should discover.

Noise in the state is more serious than additive noise in the estimated derivative. State noise perturbs both $\boldsymbol{\Phi}(\mathbf{X})$ and $\dot{\mathbf{X}}$, which produces an errors-in-variables problem and can bias sparse regression. Derivative estimation, library construction, and the sparsity level should therefore be selected together. Validation by forward simulation is essential because a small derivative regression error may accumulate into a poor trajectory.

## Known control inputs

For a controlled system,

$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},\mathbf{u}),
$$

the same construction applies after extending the library to functions $\phi_j(\mathbf{x},\mathbf{u})$. The control $\mathbf{u}(t)$ must be measured and synchronized with the state. It must also vary enough to separate autonomous state terms, control terms, and their interactions. A nearly constant control cannot identify its functional effect.

## Process noise lies outside plain SINDy

If the dynamics contain process noise, a common model uses a drift $\mathbf f$ and a noise-amplitude matrix $\mathbf G$:

$$
d\mathbf{X}_t
= \mathbf{f}(\mathbf{X}_t)\,dt
+ \mathbf{G}(\mathbf{X}_t)\,d\mathbf{W}_t
$$

Here $\mathbf W_t$ is a vector of independent Brownian motions: over a time step $\Delta t$, each component receives an independent Gaussian increment with mean zero and variance $\Delta t$. The differential notation expresses the accumulated drift and random increments. Drift increments are proportional to $\Delta t$, while typical noise increments are proportional to $\sqrt{\Delta t}$. With nonzero diffusion, paths generally have no ordinary pointwise time derivative. Regressing finite differences as though they were noisy evaluations of $\dot{\mathbf{x}}$ therefore confounds drift with stochastic increments. Plain SINDy is not a complete identification method for stochastic dynamics. Estimating drift and diffusion requires a stochastic model and likelihood, moment, or transition-density methods.

## Sparsity does not guarantee identifiability

Sparse regression selects a parsimonious representation within the chosen library; it does not prove that the governing terms are identifiable. Exact linear dependence among library columns makes some coefficient combinations observationally equivalent. Strong correlation over the sampled trajectories can create practical nonidentifiability even when the columns are independent in principle. An $L^1$ penalty may then select one of several nearly equivalent models.

Multiple initial conditions, designed control inputs, and trajectories that visit different regions of state space can improve identifiability. Stability across data subsets and regularization strengths is also informative. When uncertainty about the active terms matters, a Bayesian sparse model can expose correlated or multimodal coefficient posteriors that a single Lasso solution cannot reveal.
