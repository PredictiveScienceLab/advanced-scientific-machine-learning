# Representing Function-Valued Uncertainty

A coefficient field, initial condition, boundary condition, or geometry is a function, so its discretization may be too large to use as a direct surrogate-model input.

The treatment assumes basic partial differential equations; linear algebra, including orthogonality and eigenvalue problems; multivariate Gaussian distributions and Gaussian processes; Monte Carlo uncertainty propagation; and surrogate modeling.

We first view a scientific solver as an operator, meaning a map between function spaces, and model uncertain input functions as random fields. Singular value decomposition supplies low-rank matrix approximations and the computational basis for principal component analysis. Principal component analysis then provides data-driven coordinates for discretized fields, while the Karhunen--Loève expansion provides covariance-based coordinates for Gaussian random fields. Two heat-equation examples use these finite representations for field-to-scalar and field-to-field uncertainty propagation. The resulting coordinate systems retain dominant variability while replacing full input and output discretizations with manageable vectors.
