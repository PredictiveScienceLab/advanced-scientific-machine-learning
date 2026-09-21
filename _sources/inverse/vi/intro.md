# Variational Inference

Posterior sampling can be computationally expensive. Variational inference (VI) instead chooses a tractable family of probability distributions and optimizes within that family to approximate the true posterior.

The variational family provides a parametric approximation to the joint posterior distribution of the unknown quantities. Turning Bayesian inference into an optimization problem trades exactness for speed and can make uncertainty quantification practical in larger models.
