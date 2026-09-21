# Bayesian Inference in State-space Models

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

## Pseudo-marginal MCMC

Suppose we cannot evaluate $p(y_{1:T} \mid \theta)$ exactly, but we can compute an unbiased nonnegative estimator

$$
\widehat{p}(y_{1:T} \mid \theta, u),
$$

where $u$ collects the random variables used by the particle filter.

Pseudo-marginal MCMC replaces the exact likelihood in a Metropolis-Hastings ratio with this estimator. If the estimator is unbiased, the Markov chain targets the correct posterior after marginalizing out the auxiliary randomness. This property permits exact Bayesian inference from noisy likelihood estimates.

## Particle marginal Metropolis-Hastings

The most common PMCMC algorithm is **particle marginal Metropolis-Hastings** (PMMH) {cite:p}`andrieu2010pmcmc`.

At the current parameter value $\theta$, we run a particle filter and obtain an unbiased estimate of the marginal likelihood:

$$
\widehat{p}(y_{1:T} \mid \theta).
$$

Then we propose a new parameter value

$$
\theta' \sim q(\theta' \mid \theta),
$$

run the particle filter again at $\theta'$, and form the acceptance probability

$$
\alpha = \min\left(
1,
\frac{
\widehat{p}(y_{1:T} \mid \theta')\, p(\theta')\, q(\theta \mid \theta')
}{
\widehat{p}(y_{1:T} \mid \theta)\, p(\theta)\, q(\theta' \mid \theta)
}
\right).
$$

This looks like ordinary Metropolis-Hastings, except that the exact likelihood is replaced by its particle estimate.

## Likelihood estimation

A bootstrap particle filter produces an unbiased estimator of the marginal likelihood by multiplying the predictive normalizing constants across time {cite:p}`doucet2001smc`.

Schematically,

$$
\widehat{p}(y_{1:T} \mid \theta)
= \prod_{k=1}^T \widehat{p}(y_k \mid y_{1:k-1}, \theta).
$$

Each factor is estimated from the weighted particles at time $k$. The estimator's unbiasedness makes PMMH theoretically valid.

The particle filter therefore has two roles:

- approximating latent-state distributions,
- providing an unbiased likelihood estimate for the parameter sampler.

## PMCMC mixing

PMCMC is exact in principle, but its practical efficiency depends strongly on the variance of the log-likelihood estimator.

If the particle filter uses too few particles:

- the likelihood estimate becomes very noisy,
- acceptance probabilities become unstable,
- the MCMC chain becomes sticky.

If the particle filter uses many particles:

- the likelihood estimate becomes more stable,
- acceptance and mixing improve,
- but each MCMC iteration becomes much more expensive.

PMCMC therefore balances **statistical efficiency of the chain** against **computational cost per iteration**.

## Role of unbiasedness

PMMH requires the marginal likelihood estimator to be unbiased and nonnegative. The particle filter need not recover the exact latent posterior.

That distinction matters:

- a noisy but unbiased likelihood estimator can still yield the correct parameter posterior,
- a biased likelihood estimator typically changes the target distribution.

PMCMC is therefore an exact-approximate method whose validity depends on this structural property of the estimator.

## PMCMC versus EM

EM and PMCMC attack the same latent-state calibration problem from two different angles.

- **EM** produces a point estimate by alternating between latent-state reconstruction and parameter maximization.
- **PMCMC** samples from the full parameter posterior by embedding a particle filter inside MCMC.

That means:

- EM is usually faster when a point estimate is enough,
- PMCMC is preferable when uncertainty quantification matters,
- PMCMC is often computationally heavier because it repeatedly runs a particle filter inside a Markov chain.

This is a standard tradeoff in scientific inverse problems: optimization is cheaper, but Bayesian sampling tells us more.

## Practical tuning

The main practical considerations are:

1. the number of particles used in the particle filter;
2. the variability of the estimated log-likelihood;
3. the proposal distribution used for the parameters;
4. whether estimator noise prevents the chain from exploring the posterior; and
5. the computational cost relative to the uncertainty information gained.

These considerations often determine whether PMCMC is effective in practice.

## Relation to the calibration methods

The notebook on [particle MCMC](04_pmcmc_example.ipynb) applies PMMH to the stochastic Duffing oscillator used earlier in the calibration section. It reuses the state-space structure from filtering, but now the goal is posterior inference over the unknown system parameters rather than state estimation alone.

The calibration subsection presents two complementary approaches:

- EM for iterative point estimation,
- PMCMC for fully Bayesian parameter inference.
