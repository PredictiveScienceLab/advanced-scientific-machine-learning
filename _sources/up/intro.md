# Uncertainty Propagation through Scientific Models

Scientific predictions depend on uncertain parameters, initial conditions, and inputs, so a deterministic model evaluation must be extended to characterize uncertainty in its outputs and quantities of interest.

The treatment assumes probability and statistics, Monte Carlo sampling, ordinary differential equations (ODEs), partial differential equations (PDEs), JAX array programming, and automatic differentiation. The input uncertainty is represented by a probability distribution, and the scientific model maps each input realization to an output.

We begin with local and global sensitivity analysis, then develop polynomial-chaos expansions for low-dimensional smooth responses. Surrogate models address expensive simulations, multi-fidelity models combine information of different costs, active learning selects informative evaluations, and symmetry-aware models incorporate known structure. Together, these methods support uncertainty propagation when direct Monte Carlo is too expensive and clarify how cost, dimension, smoothness, and physical structure guide the choice of method.
