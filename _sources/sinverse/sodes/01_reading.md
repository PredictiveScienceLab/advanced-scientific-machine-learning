# Stochastic differential equations

Ordinary differential equations describe dynamics once the initial state and
parameters are fixed. Random forcing requires a model for fluctuations that
continue to enter as the system evolves. Brownian motion supplies the random
increments in an SDE {cite:p}`oksendal2003sde`.

A **stochastic process** $(X_t)_{t\geq 0}$ is a collection of random variables
indexed by time. For a fixed time $t$, the random variable $X_t$ describes the
possible values of the system at that time. One complete realization
$t\mapsto X_t$ is called a **sample path**. A stochastic process therefore
describes how uncertainty evolves over time and how random values at different
times are related.

## Brownian motion

A standard Brownian motion, or Wiener process, $(W_t)_{t\geq 0}$ is a stochastic
process with four defining properties:

1. $W_0=0$.
2. Its sample paths are continuous with probability one.
3. Increments over disjoint time intervals are independent.
4. For $0\leq s<t$, the increment satisfies

   $$
   W_t-W_s\sim\mathcal{N}(0,t-s).
   $$

Consequently, $\mathbb{E}[W_t]=0$ and
$\operatorname{Cov}(W_s,W_t)=\min(s,t)$. A Brownian increment over an interval
of length $\Delta t$ therefore has a typical size of $\sqrt{\Delta t}$, much
larger than the $\Delta t$ scaling of an ordinary time increment. Brownian
paths are continuous but nowhere differentiable with probability one, so
$dW_t/dt$ is not an ordinary function.

A random event is a yes-or-no statement about the path, such as whether the
Brownian path has crossed a specified level by time $t$. A **sigma-algebra** is
the collection of events whose answers can be resolved from the available
information. We write $\mathcal{F}_t$ for the events that can be resolved by
time $t$. As time advances, information can be gained but not lost, so
$\mathcal{F}_s\subseteq\mathcal{F}_t$ whenever $s\leq t$. The growing family
$(\mathcal{F}_t)_{t\geq 0}$ is called a **filtration**.

The filtration keeps track of causality. A stochastic process $H_t$ is
**adapted** when its value at time $t$ uses only information available by that
time. We assume that future Brownian increments are independent of the events
that can already be resolved. Thus a model may use the observed past, but it
cannot use a fluctuation that has not yet occurred.

We will not define sigma-algebras rigorously because that requires measure
theory. Measure theory is necessary to prove that the stochastic processes and
integrals we construct are well defined. Our objective here is to develop a
formal working understanding that enables engineers and scientists to use SDEs
to model systems with randomness.

## The Itô integral

Consider a partition $0=t_0<t_1<\cdots<t_n=T$. For a process that is constant
on each interval and uses the value $H_{t_k}$ known at its left endpoint, define

$$
\int_0^T H_t\,dW_t
=
\sum_{k=0}^{n-1}H_{t_k}\bigl(W_{t_{k+1}}-W_{t_k}\bigr).
$$

The **Itô integral** extends this construction by a mean-square limit to
integrands whose value over each new time interval is fixed using information
available at its start and that satisfy

$$
\mathbb{E}\!\left[\int_0^T H_t^2\,dt\right]<\infty.
$$

Such integrands are called **predictable**. For the simple process above,
$H_{t_k}$ is fixed before the next Brownian increment is revealed and is
therefore independent of that increment. The construction gives the two
identities used most often in modeling and computation:

$$
\mathbb{E}\!\left[\int_0^T H_t\,dW_t\right]=0,
$$

and the **Itô isometry**

$$
\mathbb{E}\!\left[\left(\int_0^T H_t\,dW_t\right)^2\right]
=
\mathbb{E}\!\left[\int_0^T H_t^2\,dt\right].
$$

The second identity shows how the local diffusion amplitude accumulates into
the variance of a stochastic trajectory.

## Itô stochastic differential equations

A scalar Itô SDE has the form

$$
dX_t=a(X_t,t)\,dt+b(X_t,t)\,dW_t,
$$

where $a$ is the **drift** and $b$ is the **diffusion**. The drift gives the
systematic local trend, while the diffusion controls the local variability.
By definition, the differential notation denotes the integral equation

$$
X_t
=X_0
+\int_0^t a(X_s,s)\,ds
+\int_0^t b(X_s,s)\,dW_s.
$$

Conditions such as global Lipschitz continuity and linear growth of $a$ and
$b$ ensure that this equation has a unique adapted solution for a specified
initial state and Brownian path.

The distinction between drift and diffusion matters in an inverse problem.
Data can carry different information about a drift parameter, which affects
the local mean, and a diffusion parameter, which affects the local variance.

Some physical models use the **Stratonovich integral**, denoted by
$\circ\,dW_t$. It is obtained from symmetric rather than left-endpoint sums and
obeys the ordinary chain rule. For a scalar model,

$$
dX_t=a_{\mathrm{S}}(X_t,t)\,dt+b(X_t,t)\circ dW_t,
$$

the equivalent Itô drift is

$$
a_{\mathrm{I}}(x,t)
=a_{\mathrm{S}}(x,t)+\frac{1}{2}b(x,t)\frac{\partial b}{\partial x}(x,t).
$$

The convention is therefore part of the model specification. The remainder of
this chapter uses the Itô convention.

## Itô's formula

An ordinary chain rule misses the effect of Brownian quadratic variation. Let
$X_t$ solve the scalar Itô SDE above, and let $\phi(x,t)$ have one continuous
time derivative and two continuous state derivatives. **Itô's formula** gives

$$
\begin{aligned}
d\phi(X_t,t)
={}&\left(
\frac{\partial\phi}{\partial t}
+a\frac{\partial\phi}{\partial x}
+\frac{1}{2}b^2\frac{\partial^2\phi}{\partial x^2}
\right)(X_t,t)\,dt \\
&+\left(b\frac{\partial\phi}{\partial x}\right)(X_t,t)\,dW_t.
\end{aligned}
$$

The second-derivative term follows from the quadratic-variation rule
$(dW_t)^2=dt$. As a concrete example, geometric Brownian motion satisfies

$$
dX_t=\mu X_t\,dt+\sigma X_t\,dW_t,
\qquad X_0>0.
$$

Applying Itô's formula to $\phi(x)=\log x$ yields

$$
d\log X_t
=\left(\mu-\frac{\sigma^2}{2}\right)dt+\sigma\,dW_t.
$$

Thus the logarithm has constant diffusion and a drift correction of
$-\sigma^2/2$. The stochastic-exponential-growth notebook uses this
transformation to obtain an explicit likelihood.

For a multidimensional state $X_t\in\mathbb{R}^d$ driven by an
$m$-dimensional Brownian motion, write

$$
dX_t=a(X_t,t)\,dt+B(X_t,t)\,dW_t,
$$

where $a\in\mathbb{R}^d$ and $B\in\mathbb{R}^{d\times m}$. The quadratic
covariation of the state components is

$$
dX_{i,t}\,dX_{j,t}
=\bigl(BB^{\mathsf{T}}\bigr)_{ij}(X_t,t)\,dt.
$$

For a scalar function $\phi:\mathbb{R}^d\times[0,T]\to\mathbb{R}$ with the
same smoothness as above, the multidimensional Itô formula is

$$
\begin{aligned}
d\phi(X_t,t)
={}&\left[
\frac{\partial\phi}{\partial t}
+\nabla\phi^{\mathsf{T}}a
+\frac{1}{2}\operatorname{tr}\!\left(
BB^{\mathsf{T}}\nabla^2\phi
\right)
\right](X_t,t)\,dt \\
&+\left(\nabla\phi^{\mathsf{T}}B\right)(X_t,t)\,dW_t.
\end{aligned}
$$

The covariance factor is $BB^{\mathsf{T}}$ because independent Brownian
components satisfy $dW_{k,t}\,dW_{\ell,t}=\delta_{k\ell}\,dt$.

## Euler--Maruyama simulation

Numerical inference requires a discrete transition model. For the scalar SDE
above and a time grid $t_{n+1}=t_n+\Delta t$, the **Euler--Maruyama** method
takes the following form {cite:p}`kloeden1992sde`:

$$
X_{n+1}
=X_n+a(X_n,t_n)\Delta t
+b(X_n,t_n)\Delta W_n,
$$

where the independent increments satisfy

$$
\Delta W_n\sim\mathcal{N}(0,\Delta t).
$$

Equivalently, $\Delta W_n=\sqrt{\Delta t}\,Z_n$ with
$Z_n\sim\mathcal{N}(0,1)$. The method resembles explicit Euler, but its random
increment scales as $\sqrt{\Delta t}$.

Suppose that the drift and diffusion depend on a parameter vector $\theta$.
Conditional on $X_n=x_n$, one scalar Euler--Maruyama step with
$b_\theta(x_n,t_n)\neq 0$ has the Gaussian distribution

$$
X_{n+1}\mid X_n=x_n,\theta
\sim
\mathcal{N}\!\left(
x_n+a_\theta(x_n,t_n)\Delta t,
b_\theta^2(x_n,t_n)\Delta t
\right).
$$

For an observed state path $x_{0:N}=(x_0,\ldots,x_N)$, the corresponding
approximate conditional likelihood factors as

$$
p_\theta(x_{1:N}\mid x_0)
=\prod_{n=0}^{N-1}p_\theta(x_{n+1}\mid x_n).
$$

Scientific measurements often observe the state only indirectly. Let $t_k$ be
the observation times, $x_k=X_{t_k}$ the latent state, $y_k$ the corresponding
measurement, and $\theta$ the model parameters. Numerical propagation from
$t_{k-1}$ to $t_k$ defines a transition distribution; when it has a density,
denote it by $p_\theta(x_k\mid x_{k-1})$. Together with an initial-state
density $p_\theta(x_0)$ and an observation density
$p_\theta(y_k\mid x_k)$, these distributions define a state-space model.
Filtering and smoothing infer distributions over the latent states, while
calibration methods infer unknown drift, diffusion, and observation
parameters.

The [Brownian-motion notebook](02_bm.ipynb) first constructs and simulates the
driving process. The
[stochastic-exponential-growth notebook](03_stochastic_exponential_growth.ipynb)
applies Itô's formula to a multiplicative-noise model. The
[Ornstein--Uhlenbeck notebook](04_ornstein_uhlenbeck.ipynb) then studies linear
mean reversion before the chapter turns to sequential state inference.
