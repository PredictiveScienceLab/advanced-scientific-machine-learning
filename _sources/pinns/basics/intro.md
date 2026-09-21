# Basics of Physics-Informed Neural Networks

Supervised surrogate models learn from labeled simulations. A governing differential equation provides an additional training signal: automatic differentiation can evaluate its residual at collocation points even where no solution data are available.

We first represent an unknown PDE solution with a neural network, enforce boundary conditions, and minimize a residual-based loss. We then expose spectral bias, diagnose loss imbalance and other common training pathologies, and replace the squared residual with an energy functional whose minimization requires fewer derivatives. These small forward problems provide the formulation and diagnostics needed before the same ideas are applied to families of equations and inverse problems.
