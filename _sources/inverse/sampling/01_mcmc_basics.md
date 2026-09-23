# Basics of Markov Chain Monte Carlo

Bayesian inversion produces a posterior distribution, but scientific questions
usually require expectations, probabilities, and credible intervals computed
under that distribution. MCMC approximates these
quantities with a dependent sequence whose long-run distribution is the
posterior. This section develops invariant Markov transitions, Gibbs sampling,
Metropolis--Hastings, gradient-informed samplers, and the diagnostics needed to
judge the resulting draws. A broader treatment appears in
{cite:t}`gelman2013bda`.

## The target distribution

Let $x\in\mathcal{X}$ denote all unknown quantities and let $y$ denote the
observed data. Assume that the posterior has a density $\pi$ with respect to a
reference measure $\mu$. The corresponding probability measure is
$\Pi(A)=\int_A\pi(x)\,\mu(dx)$, and its density is

$$
\pi(x) = p(x\mid y)
       = \frac{p(y\mid x)p(x)}{p(y)}.
$$

The evidence $p(y)$ is often an intractable integral. MCMC avoids evaluating it.
Define the unnormalized target

$$
\widetilde{\pi}(x) = p(y\mid x)p(x).
$$

Its normalizing constant is

$$
Z=\int_{\mathcal X}\widetilde{\pi}(x)\,\mu(dx)=p(y),
$$

so that $\pi(x)=\widetilde{\pi}(x)/Z$.
Metropolis--Hastings uses ratios of target densities, and $Z$ cancels from every
ratio. In numerical work, we evaluate $\log\widetilde{\pi}(x)$ to prevent
underflow and to turn products of likelihood and prior terms into sums.

Suppose the scientific quantity of interest is a function $g(x)$. Given retained
draws that adequately represent $\Pi$, a single chain estimates its expectation
by

$$
\mathbb{E}_{\pi}[g(X)]
\approx
\frac{1}{N}\sum_{n=1}^{N} g(X_n).
$$

Independent draws $X_n\sim\Pi$ would make this ordinary Monte Carlo. MCMC
instead constructs the draws sequentially. The validity of this estimator is an
asymptotic result; the diagnostics below assess whether a finite run provides
enough information for the desired accuracy.

## Markov chains and invariant distributions

A Markov chain is a sequence $X_0,X_1,\ldots$ in which the distribution of the
next state depends on the past only through the current state. Its transition
kernel $P$ assigns a probability $P(x,A)$ of moving from state $x$ into a set
$A$:

$$
\mathbb{P}(X_{n+1}\in A\mid X_0,\ldots,X_n)
= P(X_n,A).
$$

A probability distribution $\Pi$ is **invariant** for $P$ when one transition
preserves it:

$$
\int P(x,A)\,\Pi(dx)=\Pi(A)
$$

for every event $A$, where $A$ represents a collection of possible states
whose probability we wish to track. If $X_0\sim\Pi$, invariance implies
$X_n\sim\Pi$ for every $n$; the chain is then **stationary**. Invariance alone
does not guarantee that a chain started elsewhere will approach $\Pi$. The
transition must also be able to explore the relevant support without becoming
trapped in a periodic pattern. Under the usual ergodicity conditions, the
distribution of $X_n$ approaches $\Pi$, and empirical averages converge to
posterior expectations.

A convenient way to establish invariance is **detailed balance**, or
reversibility:

$$
\Pi(dx)P(x,dx')
=
\Pi(dx')P(x',dx).
$$

The two sides describe probability flow in opposite directions at stationarity.
Integrating out the first state shows that detailed balance implies invariance.
When the off-diagonal moves have a transition density $k(x'\mid x)$, this
measure identity reduces away from $x'=x$ to
${\pi(x)k(x'\mid x)=\pi(x')k(x\mid x')}$. A Metropolis--Hastings kernel also has
a point mass at the current state because of rejection, so the measure form is
the general statement. Detailed balance is sufficient, not necessary; some
valid MCMC transitions are nonreversible.

## Gibbs sampling

Suppose $x=(x_1,\ldots,x_d)$, and write $x_{-i}$ for all components except
$x_i$. Gibbs sampling constructs a transition from the full conditional
distributions

$$
\pi(x_i\mid x_{-i})=p(x_i\mid x_{-i},y).
$$

Starting from $x^{(0)}$, one sweep updates $x_1,\ldots,x_d$ in turn. Each update
draws $x_i$ from its full conditional given the most recently available values
of the other components. Groups of variables can be updated together when a
block conditional is easier to sample or mixes more efficiently. Every exact
conditional update preserves the joint posterior, so their composition also
preserves it {cite:p}`gelfand1990sampling`.

Gibbs sampling is especially convenient when the full conditionals belong to
standard families. Its limitation is equally important: a full conditional may
itself be difficult to sample, and strongly dependent coordinates may move only
slowly when updated one at a time. In that case, a conditional update can be
replaced by another valid MCMC transition, or the variables can be blocked.

## The Metropolis--Hastings construction

Metropolis--Hastings converts a proposal distribution into a transition that
satisfies detailed balance {cite:p}`hastings1970monte`. Let
$q(x'\mid x)$ be a proposal density from which we can draw a candidate $x'$ when
the chain is at $x$. One transition proceeds as follows.

1. Draw a candidate $x'\sim q(\cdot\mid x)$.
2. Compute the acceptance probability

   $$
   \alpha(x,x')
   =
   \min\left\{
   1,
   \frac{\widetilde{\pi}(x')q(x\mid x')}
        {\widetilde{\pi}(x)q(x'\mid x)}
   \right\}.
   $$

3. Draw $u\sim\operatorname{Uniform}(0,1)$. Set the next state to $x'$ when
   $u\leq\alpha(x,x')$; otherwise, keep the current state $x$.

The rejection step must be recorded as another draw at the current state.
Deleting repeated states changes the distribution represented by the chain.
When the proposal is symmetric, so that $q(x'\mid x)=q(x\mid x')$, the proposal
terms cancel and only the target-density ratio remains. This symmetric special
case is the original **Metropolis algorithm** {cite:p}`metropolis1953equation`.

The acceptance rule is designed so that the probability flow between two
distinct states is

$$
\pi(x)q(x'\mid x)\alpha(x,x')
=
\min\left\{
\pi(x)q(x'\mid x),
\pi(x')q(x\mid x')
\right\}.
$$

The right-hand side is unchanged when $x$ and $x'$ are exchanged. Accepted
moves therefore satisfy detailed balance, and rejected moves provide the
remaining probability of staying at the current state. The resulting kernel
has $\Pi$ as an invariant distribution even though the normalizing constant of
its density $\pi$ is unknown.

Proposal choice controls efficiency. Very small random-walk steps are accepted
often but move slowly. Very large steps travel farther but are usually rejected.
An acceptance rate by itself does not establish correctness or convergence; it
is one diagnostic of how a particular proposal interacts with the target
geometry.

## The Metropolis-adjusted Langevin algorithm

Now specialize to $\mathcal X=\mathbb R^d$, write the unnormalized target as
$\widetilde{\pi}(x)=\exp[-U(x)]$, and assume that $U$ is differentiable. A
random walk ignores the local geometry of $U$. For a step size $h>0$, the
Metropolis-adjusted Langevin algorithm (MALA) instead proposes
{cite:p}`roberts1996langevin`

$$
q(x'\mid x)
=
\mathcal{N}\!\left(
x'\,\middle|\,
x-\frac{h}{2}\nabla U(x),
hI_d
\right).
$$

Here $I_d$ is the $d\times d$ identity matrix. The gradient acts like a force
toward regions of larger target density. Because this proposal is generally
asymmetric, both proposal densities must remain in the Metropolis--Hastings
ratio. The correction removes the bias introduced by the finite Langevin step.
MALA still requires the proposal scale $h$ to be chosen. More broadly,
practical chains need an initialization and tuning period before draws are
retained.

## Initialization and warmup

A finite chain remembers its initial state. The early iterations are therefore
used to move toward the region carrying most of the target probability and, for
adaptive algorithms, to tune quantities such as proposal scale and step size.
This initial period is called **warmup**. The older term **burn-in** usually refers
only to discarding early iterations; warmup is more informative because it also
names the adaptation performed during that period.

Warmup draws are excluded from posterior summaries. For adaptive MCMC, the
adapted transition parameters should then be fixed before collecting inference
draws unless the adaptation scheme has its own validity guarantee. Discarding
an arbitrary number or a fixed fraction, such as one half, does not prove that
the retained chain has converged. Initialization should therefore be assessed
with multiple chains and diagnostics rather than with one universal burn-in
length.

## Multiple-chain diagnostics

Run several independent chains, typically four, from meaningfully dispersed
initial states. Trace and rank plots should show each chain repeatedly exploring
the same stable region, without long trends or chains confined to different
modes.

The classical Gelman--Rubin calculation makes the comparison between chains
explicit {cite:p}`gelman1992inference`. Choose a scalar quantity of interest
$\psi$. Suppose that $\psi_{ij}$ is draw $i=1,\ldots,n$ from chain
$j=1,\ldots,m$, and define

$$
\bar{\psi}_{\cdot j}
=
\frac{1}{n}\sum_{i=1}^{n}\psi_{ij},
\qquad
\bar{\psi}_{\cdot\cdot}
=
\frac{1}{m}\sum_{j=1}^{m}\bar{\psi}_{\cdot j}.
$$

The average within-chain variance and the between-chain variance are

$$
\begin{aligned}
W
&=
\frac{1}{m}\sum_{j=1}^{m}
\left[
\frac{1}{n-1}\sum_{i=1}^{n}
(\psi_{ij}-\bar{\psi}_{\cdot j})^2
\right], \\
B
&=
\frac{n}{m-1}\sum_{j=1}^{m}
(\bar{\psi}_{\cdot j}-\bar{\psi}_{\cdot\cdot})^2.
\end{aligned}
$$

They give the variance estimate

$$
\widehat{\operatorname{var}}^{+}(\psi)
=
\frac{n-1}{n}W+\frac{1}{n}B
$$

and the potential scale reduction factor

$$
\widehat{R}
=
\sqrt{
\frac{\widehat{\operatorname{var}}^{+}(\psi)}{W}
}.
$$

When chains explore different regions, $B$ is large relative to $W$ and
$\widehat{R}$ exceeds one. As the chains agree, $\widehat{R}$ approaches one.
This classical formula explains the diagnostic, but a final analysis should use
the modern rank-normalized split version. A modern implementation also applies
the calculation to folded draws and takes the larger value so that failures in
both location and scale are easier to detect {cite:p}`vehtari2021rank`.
$\widehat{R}\leq 1.01$ is a useful warning screen, together with adequate bulk
and tail effective sample sizes, but it is not a certificate of convergence.

## Autocorrelation and effective sample size

Successive MCMC states are dependent. Consider one post-warmup stationary chain
and a scalar quantity $g(X_n)$. Its lag-$k$ autocorrelation is

$$
\rho_k
=
\operatorname{Corr}_{\pi}\bigl(g(X_n),g(X_{n+k})\bigr).
$$

Positive autocorrelation means that nearby draws carry overlapping
information. When the autocorrelations are summable, the variance of the
empirical mean is approximately

$$
\operatorname{Var}\left[
\frac{1}{N}\sum_{n=1}^{N}g(X_n)
\right]
\approx
\frac{\operatorname{Var}_{\pi}[g(X)]}{N}
\left(1+2\sum_{k=1}^{\infty}\rho_k\right).
$$

The factor in parentheses is the integrated autocorrelation time $\tau_g$. Under
a Markov chain central limit theorem, it leads to the **effective sample size**
(ESS)

$$
N_{\mathrm{eff}}
=
\frac{N}{\tau_g}
=
\frac{N}{1+2\sum_{k=1}^{\infty}\rho_k}.
$$

Thus, $N_{\mathrm{eff}}$ is the number of independent draws that would provide
roughly the same precision for the chosen quantity $g$. Effective sample size
depends on the estimand: the posterior mean, a tail probability, and a scale
parameter may mix at different rates. Modern software consequently reports
bulk and tail effective sample sizes {cite:p}`vehtari2021rank` as well as the
Monte Carlo standard error (MCSE), with

$$
\operatorname{MCSE}(\bar{g})
\approx
\sqrt{\frac{\operatorname{Var}_{\pi}[g(X)]}{N_{\mathrm{eff}}}}.
$$

With $m$ chains of $n$ retained draws, modern estimators use all $mn$ draws
while also checking agreement between chains. Negative autocorrelation can even
make an estimand-specific ESS exceed the nominal draw count; ESS is a precision
equivalent, not a count of unique states.

The required ESS follows from the desired precision of the scientific quantity
through its MCSE. No fixed ESS threshold is a universal stopping rule.

Routine thinning is usually unnecessary. Keeping every $k$th state reduces
storage and the correlation between retained states, but it normally discards
information without improving the precision available from a fixed amount of
computation. ESS quantifies dependence; it does not determine a thinning
interval. Thin only when storage or the cost of downstream processing is the
binding constraint, and retain the unthinned draws when practical
{cite:p}`link2012thinning`.

No finite diagnostic proves convergence. A defensible analysis combines
multiple chains, trace and rank plots, $\widehat{R}$, effective sample sizes,
Monte Carlo standard errors, and algorithm-specific warnings for every quantity
that matters scientifically. Posterior predictive checks address a different
question: whether the fitted probabilistic model can reproduce relevant
features of the data.

## Hamiltonian Monte Carlo

Random-walk Metropolis--Hastings becomes inefficient when a posterior is
high-dimensional, strongly correlated, or sharply curved. Hamiltonian Monte
Carlo (HMC) uses the gradient
of the log target to propose distant states while avoiding the diffusive motion
of a random walk {cite:p}`neal2011hmc`. Let

$$
U(x)=-\log\widetilde{\pi}(x)
$$

be the potential energy. Introduce a fictitious momentum
$r\sim\mathcal{N}(0,M)$ for a positive-definite mass matrix $M$, and define

$$
T(r)=\frac{1}{2}r^{\mathsf{T}}M^{-1}r,
\qquad
H(x,r)=U(x)+T(r).
$$

The joint density of position and momentum is proportional to
$\exp[-H(x,r)]$. Integrating out $r$ leaves the target density for $x$.
Hamilton's equations for this system are

$$
\dot{x}=M^{-1}r,
\qquad
\dot{r}=-\nabla U(x).
$$

The exact Hamiltonian flow conserves $H$ and phase-space volume; together with
a momentum reversal, it is reversible. These properties allow a trajectory to
be used inside a Metropolis--Hastings proposal. Exact trajectories are rarely
available, so HMC applies $L$ leapfrog steps with step size $\epsilon$:

$$
r_{\ell+1/2}
=r_{\ell}-\frac{\epsilon}{2}\nabla U(x_{\ell}),
$$

$$
x_{\ell+1}
=x_{\ell}+\epsilon M^{-1}r_{\ell+1/2},
$$

$$
r_{\ell+1}
=r_{\ell+1/2}-\frac{\epsilon}{2}\nabla U(x_{\ell+1}).
$$

Leapfrog is reversible and volume-preserving, although it only approximately
conserves $H$. After drawing a fresh momentum $r_0$, the leapfrog endpoint
defines the proposal $(x_L,-r_L)$, which is accepted with probability

$$
\alpha
=
\min\left\{
1,
\exp[-H(x_L,-r_L)+H(x_0,r_0)]
\right\}.
$$

Because $T(-r)=T(r)$, this is the same numerical acceptance probability as the
usual expression with $r_L$. The flip makes the deterministic proposal
reversible; it need not be stored when the momentum is immediately discarded
or fully refreshed. The momentum refresh is a Gibbs update, and the corrected
trajectory is a Metropolis--Hastings update on the augmented state. Their
composition preserves the target. The step size $\epsilon$, number of steps
$L$, and mass matrix $M$ control efficiency: a poorly scaled $M$ or a step size
that is too large can make the numerical trajectories unreliable.

## The No-U-Turn Sampler

A fixed HMC path can be too short to explore efficiently or so long that it
retraces its route. The No-U-Turn Sampler (NUTS) removes the need to choose the
path length $L$ by hand.
It recursively doubles a binary tree of forward and backward leapfrog states
and stops extending a trajectory when its endpoints begin to turn back toward
one another. For endpoints $(x^-,r^-)$ and $(x^+,r^+)$, the elementary
mass-matrix-aware check stops when

$$
(x^+-x^-)^{\mathsf T}M^{-1}r^-<0
\quad\text{or}\quad
(x^+-x^-)^{\mathsf T}M^{-1}r^+<0.
$$

The details of building the tree and selecting a state from it are essential
for preserving the target distribution; this criterion cannot simply be used
as a naive state-dependent stopping rule {cite:p}`hoffman2014nuts`.

NUTS does not eliminate all tuning. During warmup, a dual-averaging procedure
adapts the step size toward a target acceptance rate, and windowed adaptation
can estimate a diagonal or dense mass matrix. These quantities are then fixed
for the inference draws.

For Hamiltonian methods, post-warmup divergences warn that numerical
trajectories may have missed regions of the target and can signal biased
estimates. Reaching the maximum NUTS tree depth is mainly an efficiency warning,
while the energy-based Bayesian fraction of missing information (E-BFMI) checks
whether the chains explore the marginal energy distribution
{cite:p}`betancourt2017conceptual`.

BlackJAX exposes low-level JAX kernels and adaptation routines, while NumPyro
provides a higher-level probabilistic-programming interface for model-based
inference {cite:p}`cabezas2024blackjax,phan2019numpyro`. Both support the
computational workflows developed in the following sections.

The next three sections turn this foundation into computation. The first
implements Metropolis--Hastings with BlackJAX and makes the diagnostics
concrete. The second introduces gradient-based HMC. The third uses NUTS and
warmup adaptation. Together they show how the invariant-distribution principle
remains fixed while the transition mechanism becomes better matched to modern
posterior geometry.
