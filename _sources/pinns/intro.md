# Physics-Informed Neural Networks (PINNs)

Scientific workflows use solvers as forward maps for uncertainty propagation and inverse inference. When repeated solver calls or labeled solution fields are expensive, a neural approximation can instead be trained with information supplied by the governing differential equation.

Physics-informed neural networks {cite:p}`raissi2019physics` incorporate equation residuals, boundary conditions, and available observations into the training objective. This chapter begins with forward problems, spectral bias, and energy formulations; extends the construction to parametric solution maps and physics-informed neural operators; and concludes with deterministic and Bayesian inverse problems. The resulting framework shows how physical constraints can reduce reliance on labeled data and how boundary enforcement, scaling, spectral bias, and optimization affect the reliability of physics-informed models.
