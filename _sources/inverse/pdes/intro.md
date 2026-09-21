# PDE-constrained inverse problems

Inverse problems for PDEs are the natural mathematical language for inferring spatially distributed physics from indirect measurements. The unknown may be a coefficient field, a source term, or a boundary condition, while the data are usually sparse observations of the resulting solution.

This section formulates the calibration of partial differential equations as a Bayesian inverse problem. Compared with finite-dimensional inverse problems, the unknowns are often larger, the forward solves are more expensive, and regularization becomes more important because the data leave much of the field unobserved. The resulting inference must also be interpreted together with the numerical discretization and solver used to evaluate the forward model.
