# PINNs for Inverse Problems

A parametric PINN treats problem parameters as known inputs and predicts the corresponding states. An inverse problem reverses that relationship: sparse observations are available, while a physical coefficient, source term, or constitutive parameter and the associated state are unknown.

We first train a deterministic inverse PINN to estimate an unknown conductivity together with the state of a diffusion equation. We then define a Bayesian PINN with priors on the network weights and conductivity, a data likelihood, and a finite residual pseudo-likelihood. The equation residual restricts the state--parameter pairs compatible with the sparse observations. The deterministic model produces a single fitted conductivity and state, whereas the Bayesian model represents uncertainty in both conditional on the chosen architecture, priors, noise scales, and collocation points.
