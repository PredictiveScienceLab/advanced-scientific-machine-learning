# Surrogate Models

Polynomial chaos becomes costly as input dimension grows and can converge slowly for irregular responses. More generally, Monte Carlo, sensitivity analysis, and optimization may all be impractical when one evaluation of the scientific model takes hours. A surrogate replaces that expensive input-output map with a validated statistical approximation.

The reader should be comfortable with multivariate probability and Gaussian conditioning; supervised regression and model validation; feedforward neural networks and gradient-based training; and Gaussian process (GP) regression, including covariance functions, conditioning, and predictive uncertainty. We also use Latin hypercube and Sobol designs for input sampling.

We first define the surrogate-modeling workflow, including data collection, model selection, validation, and diagnostics. Neural-network and exact Gaussian-process surrogates are then built for the same biomechanical simulation. We next derive a sparse variational Gaussian process in terms of inducing variables and its evidence lower bound before applying the approximation to a larger data set. A reliable surrogate makes repeated prediction, uncertainty propagation, sensitivity analysis, and optimization feasible while keeping approximation error visible through independent diagnostics.
