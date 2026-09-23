# Bayesian Inference in State-Space Models

Expectation-maximization gives a point estimate of the parameters in a state-space model. Sometimes that is enough. But in scientific machine learning we often care about full posterior uncertainty: how much do the data constrain the parameters, how correlated are they, and how much posterior mass lies in physically distinct regimes?

That is a Bayesian calibration problem. The target is

$$
p(\theta \mid y_{1:T}) \propto p(y_{1:T} \mid \theta)\, p(\theta),
$$

where $\theta$ contains the unknown physical parameters and $p(y_{1:T} \mid \theta)$ is the marginal likelihood of the observations under the latent-state model.

## Particle MCMC

The difficulty is the same as before: in a nonlinear or non-Gaussian state-space model, the marginal likelihood

$$
p(y_{1:T} \mid \theta) = \int p(x_{0:T}, y_{1:T} \mid \theta)\, dx_{0:T}
$$

is usually intractable.

Particle Markov chain Monte Carlo (PMCMC) solves this by combining:

- a particle method to handle the latent states,
- an MCMC method to explore parameter space.

The foundational result of {cite:t}`andrieu2010pmcmc` is that certain particle-based likelihood estimators can be embedded inside MCMC while still targeting the **exact** parameter posterior.

## Pseudo-Marginal MCMC

Suppose we cannot evaluate $p(y_{1:T} \mid \theta)$ exactly, but we can compute an unbiased nonnegative estimator

$$
\widehat{p}(y_{1:T} \mid \theta, \zeta),
$$

where $\zeta$ collects the random variables used by the particle filter.

Pseudo-marginal MCMC replaces the exact likelihood in a Metropolis--Hastings ratio with this estimator. If the estimator is unbiased, the Markov chain targets the correct posterior after marginalizing out the auxiliary randomness. This property permits exact Bayesian inference from noisy likelihood estimates.

## Particle Marginal Metropolis--Hastings

The most common PMCMC algorithm is **particle marginal Metropolis--Hastings** (PMMH) {cite:p}`andrieu2010pmcmc`.

At the current parameter value $\theta$, we hold the unbiased particle-filter estimate of the marginal likelihood that was computed when $\theta$ was accepted (it is stored, not recomputed):

$$
\widehat{p}(y_{1:T} \mid \theta).
$$

Then we propose a new parameter value

$$
\theta' \sim q(\theta' \mid \theta),
$$

run a new particle filter at $\theta'$, and form the acceptance probability

$$
a(\theta,\theta') = \min\left(
1,
\frac{
\widehat{p}(y_{1:T} \mid \theta')\, p(\theta')\, q(\theta \mid \theta')
}{
\widehat{p}(y_{1:T} \mid \theta)\, p(\theta)\, q(\theta' \mid \theta)
}
\right).
$$

This looks like ordinary Metropolis--Hastings, except that the exact likelihood is replaced by its particle estimate.

## Likelihood Estimation

A bootstrap particle filter produces an unbiased estimator of the marginal likelihood by multiplying particle estimates of the one-step predictive likelihoods across time {cite:p}`doucet2001smc`.

Schematically,

$$
\widehat{p}(y_{1:T} \mid \theta)
= \prod_{t=1}^T \widehat{p}(y_t \mid y_{1:t-1}, \theta).
$$

Each factor is estimated from the weighted particles at time $t$. The same
filter therefore approximates the latent-state distributions and supplies the
likelihood estimate needed by the parameter sampler.

## PMCMC Mixing

PMCMC targets the exact parameter posterior under the pseudo-marginal
conditions, but a finite chain can explore that posterior poorly. With too
few particles, a likelihood overestimate can hold the chain at one parameter
value for many rejected proposals. This is often called a *sticky chain*.
Increasing the particle count reduces estimator noise and can improve
exploration, at the cost of a more expensive particle filter at every proposal.

## Role of Unbiasedness

PMMH requires a nonnegative estimator that is unbiased for the marginal
likelihood, not for its logarithm. The particle filter may approximate the
latent-state posterior while still supplying such an estimator. A biased
likelihood estimator generally changes the parameter target. As in other
MCMC methods, posterior interpretation also requires a proper target and
adequate exploration; unbiasedness alone does not validate a finite run.

## PMCMC versus EM

The Duffing MCEM calculation estimates one parameter vector by alternating
particle smoothing with a fixed-sample maximization. PMCMC instead uses
particle likelihood estimates to sample a parameter posterior, revealing
uncertainty and correlations that a point estimate cannot describe. Both
methods perform repeated particle calculations; their relative cost depends
on the particle counts, optimization or sampling budget, and accuracy needed.

## Practical Tuning

The main practical considerations are:

1. the number of particles used in the particle filter;
2. the variability of the estimated log-likelihood;
3. the proposal distribution used for the parameters;
4. whether estimator noise prevents the chain from exploring the posterior; and
5. the computational cost relative to the uncertainty information gained.

## Relation to the Calibration Methods

The next [particle MCMC example](04_pmcmc_example.ipynb) applies PMMH to the stochastic Duffing oscillator used in the preceding EM example. It reuses the state-space structure from filtering, but now the goal is posterior inference over the unknown system parameters rather than state estimation alone.
