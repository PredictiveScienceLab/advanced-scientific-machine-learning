# Expectation-Maximization for State-Space Models

Parameter estimation in a state-space model is difficult because the state
trajectory is unobserved. EM alternates between
smoothing the latent trajectory under the current parameters and optimizing an
average complete-data objective. The target is a maximum-likelihood point
estimate of the model parameters {cite:p}`dempster1977em`.

## State-Space Likelihood

Let $x_t$ be the latent state for $t=0,\ldots,T$, and let $y_t$ and $u_t$ be
the observation and known input for $t=1,\ldots,T$. We write
$x_{0:T}=(x_0,\ldots,x_T)$, $y_{1:T}=(y_1,\ldots,y_T)$, and
$u_{1:T}=(u_1,\ldots,u_T)$. Let the parameter vector $\theta$ belong to a
parameter space $\Theta$. The initial, transition, and observation densities are
$p_{0,\theta}$, $f_{t,\theta}$, and $g_{t,\theta}$. Their joint density is

$$
p_\theta(x_{0:T},y_{1:T}\mid u_{1:T})
=
p_{0,\theta}(x_0)
\prod_{t=1}^T
f_{t,\theta}(x_t\mid x_{t-1},u_t)
g_{t,\theta}(y_t\mid x_t,u_t).
$$

The complete data consist of the states and observations together,
$(x_{0:T},y_{1:T})$. Their log-likelihood is

$$
\ell_c(\theta;x_{0:T},y_{1:T},u_{1:T})
=
\log p_\theta(x_{0:T},y_{1:T}\mid u_{1:T}).
$$

The observed data contain only $y_{1:T}$. Integrating out the latent path gives
the observed-data log-likelihood

$$
\ell(\theta)
=
\log p_\theta(y_{1:T}\mid u_{1:T})
=
\log\int
p_\theta(x_{0:T},y_{1:T}\mid u_{1:T})\,dx_{0:T}.
$$

Maximum-likelihood estimation seeks

$$
\widehat\theta_{\mathrm{ML}}
\in
\operatorname*{arg\,max}_{\theta\in\Theta}\ell(\theta).
$$

The path integral is usually the difficult part. In contrast, the complete-data
log-likelihood separates into local terms:

$$
\begin{aligned}
\ell_c(\theta;x_{0:T},y_{1:T},u_{1:T})
={}&\log p_{0,\theta}(x_0)\\
&+\sum_{t=1}^T
\log f_{t,\theta}(x_t\mid x_{t-1},u_t)\\
&+\sum_{t=1}^T
\log g_{t,\theta}(y_t\mid x_t,u_t).
\end{aligned}
$$

EM exploits this simpler decomposition while accounting for uncertainty in the
missing trajectory.

## EM Iteration

Let $\theta^{(m)}$ be the parameter value at iteration $m$. The E-step uses the
joint smoothing density

$$
q_m(x_{0:T})
=
p_{\theta^{(m)}}(x_{0:T}\mid y_{1:T},u_{1:T}).
$$

Let $X_{0:T}$ denote a random trajectory with this density. The E-step
defines the expected complete-data log-likelihood

$$
Q(\theta\mid\theta^{(m)})
=
\mathbb{E}_{q_m}
\left[
\ell_c(\theta;X_{0:T},y_{1:T},u_{1:T})
\right].
$$

The expectation averages the complete-data objective over plausible latent
paths under the current parameters. Expanding the state-space factorization
shows the information required by the E-step:

$$
\begin{aligned}
Q(\theta\mid\theta^{(m)})
={}&
\mathbb{E}_{q_m}[\log p_{0,\theta}(X_0)]\\
&+\sum_{t=1}^T
\mathbb{E}_{q_m}
[\log f_{t,\theta}(X_t\mid X_{t-1},u_t)]\\
&+\sum_{t=1}^T
\mathbb{E}_{q_m}
[\log g_{t,\theta}(y_t\mid X_t,u_t)].
\end{aligned}
$$

The transition terms depend jointly on $X_{t-1}$ and $X_t$. The E-step
therefore needs two-time smoothing distributions or complete smoothing
trajectories; unrelated draws from separate one-time marginals do not preserve
the required temporal dependence.

The M-step updates the parameters by

$$
\theta^{(m+1)}
\in
\operatorname*{arg\,max}_{\theta\in\Theta}
Q(\theta\mid\theta^{(m)}).
$$

For a state-space model with linear transition and observation maps and
Gaussian noises, a Kalman smoother can evaluate the needed expectations, and
several parameter updates have closed forms
{cite:p}`shumway1982em`. Nonlinear or non-Gaussian models usually require
numerical smoothing and optimization.

## Likelihood Ascent

The EM ascent property follows from the same variational identity used for the
evidence lower bound. Assume that the expectations below are finite and that
$q_m$ assigns no mass where the candidate posterior density is zero. For
densities $q$ and $r$ and a random variable $X\sim q$, the
KL divergence is

$$
\operatorname{KL}(q\|r)
=
\mathbb{E}_q\!\left[\log\frac{q(X)}{r(X)}\right].
$$

Define the entropy

$$
H(q_m)=-\mathbb{E}_{q_m}[\log q_m(X_{0:T})].
$$

Then

$$
\ell(\theta)
=
Q(\theta\mid\theta^{(m)})
+H(q_m)
+\operatorname{KL}\!\left(
q_m\,\middle\|\,
p_\theta(\,\cdot\mid y_{1:T},u_{1:T})
\right).
$$

At $\theta=\theta^{(m)}$, the Kullback--Leibler divergence is zero because
$q_m$ is exactly the current smoothing density. Subtracting the identity at
$\theta^{(m)}$ gives

$$
\begin{aligned}
\ell(\theta)-\ell(\theta^{(m)})
={}&Q(\theta\mid\theta^{(m)})
-Q(\theta^{(m)}\mid\theta^{(m)})\\
&+\operatorname{KL}\!\left(
q_m\,\middle\|\,
p_\theta(\,\cdot\mid y_{1:T},u_{1:T})
\right).
\end{aligned}
$$

The nonnegativity of the KL divergence shows that an exact M-step cannot
decrease the observed-data likelihood. A **generalized EM** update retains this
guarantee when it merely increases the same fixed function:

$$
Q(\theta^{(m+1)}\mid\theta^{(m)})
\geq
Q(\theta^{(m)}\mid\theta^{(m)}).
$$

This guarantee requires the exact smoothing density in the E-step and a
verified increase of its fixed $Q$ function. It does not guarantee convergence
to the global maximum; initialization matters, and different parameter values
may induce the same observed-data distribution. Stronger convergence statements
require additional regularity conditions
{cite:p}`wu1983em`.

## Particle and Stochastic Approximations

Classical Monte Carlo EM assumes that joint smoothing trajectories can be
sampled from the exact density $q_m$. For $M$ such draws,

$$
X_{0:T}^{(j)}\sim q_m,
\qquad j=1,\ldots,M.
$$

Monte Carlo EM replaces $Q$ by

$$
\widehat Q_M(\theta\mid\theta^{(m)})
=
\frac{1}{M}\sum_{j=1}^M
\ell_c(\theta;X_{0:T}^{(j)},y_{1:T},u_{1:T}),
$$

and holds these trajectories fixed while optimizing this sampled objective
{cite:p}`wei1990mcem`.

For a nonlinear state-space model, a particle filter and smoother can instead
construct an $N$-particle approximation $\widetilde q_{m,N}$ to $q_m$
{cite:p}`doucet2001smc`. Drawing the $M$ trajectories from
$\widetilde q_{m,N}$ gives a particle objective denoted by
$\widehat Q_{M,N}$.

For finite $M$, $\widehat Q_M$ is a random function. An increase in one
realization of $\widehat Q_M$ need not imply an increase in the exact $Q$.
Replacing $q_m$ by $\widetilde q_{m,N}$ adds particle approximation error, so
the exact EM ascent identity no longer applies. Numerical M-steps add
optimization error as well.

Redrawing trajectories during every parameter update changes the sampled
objective and gives a stochastic EM-like procedure. Classical fixed-sample
Monte Carlo EM holds the trajectories fixed during an M-step.
Stochastic-approximation EM maintains a running approximation with a prescribed
decreasing step-size schedule and has its own convergence conditions
{cite:p}`delyon1999saem`.

Particle count, smoothing-trajectory count, optimizer step size, initialization,
and repeated runs are therefore part of the numerical assessment. The observed
likelihood or a controlled estimate of it should be monitored; monotonicity
should not be assumed for an approximate implementation.

The Euler--Maruyama model used in the following example introduces one more
caveat. Its EM objective is the likelihood of the selected discrete-time
state-space approximation. Parameter estimates should be checked as the time
step and transition approximation are refined.

## Point Estimates and Uncertainty

EM returns a parameter point estimate and a smoothing distribution conditional
on that estimate. It does not by itself provide the full parameter posterior.
The particle Markov chain Monte Carlo treatment later in this chapter targets
that posterior and represents parameter uncertainty with samples.

Estimating dynamical-model parameters from observations is often called
**system identification**. The [Duffing calibration example](02_em_example.ipynb)
next combines particle smoothing with numerical parameter updates. Its
finite-particle and stochastic updates form an approximation to the ideal EM
iteration developed here. Its particle likelihood estimates are a diagnostic; iteration-wise
monotonicity is not guaranteed.

## Exercises

1. Derive the three-term decomposition of the complete-data log-likelihood from
   the state-space factorization.
2. Derive the likelihood identity involving $Q$, entropy, and KL divergence.
   Use it to prove the generalized-EM ascent statement under the stated
   finiteness and support conditions.
3. Explain why the transition contribution to $Q$ requires the joint smoothing
   law of $(X_{t-1},X_t)$. Give an example of an error caused by pairing
   independent draws from the two marginal smoothing distributions.
4. Distinguish exact EM, generalized EM, fixed-sample Monte Carlo EM, and a
   procedure that redraws a smoothing trajectory at every gradient step. State
   which likelihood-ascent guarantees apply to each case.
