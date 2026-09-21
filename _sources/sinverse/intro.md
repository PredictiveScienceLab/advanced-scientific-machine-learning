# Inverse Problems in Stochastic Scientific Models

Deterministic dynamical systems assign one trajectory to an initial condition and a set of parameters. Many scientific systems also experience unresolved forcing, thermal fluctuations, or random environmental inputs. A stochastic model represents these effects through a probability distribution over trajectories, so inference must account for uncertainty in both the evolving state and the observations.

Deterministic inverse methods provide the basic language of forward models, likelihoods, and parameter inference. Stochastic dynamics add a latent state path: the physical state evolves randomly and is observed only indirectly through noisy measurements. Reconstructing that path and learning the model parameters are therefore coupled problems.

The chapter begins with stochastic differential equations and their numerical discretization. It then develops filtering and smoothing for state reconstruction, followed by expectation-maximization and particle Markov chain Monte Carlo for parameter calibration. These constructions provide a common framework for simulating stochastic dynamics, estimating hidden states, and quantifying parameter uncertainty from time-series data.
