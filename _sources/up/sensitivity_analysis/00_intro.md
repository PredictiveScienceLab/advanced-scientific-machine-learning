# Sensitivity Analysis of ODEs and PDEs

Uncertain inputs affect a scientific model through both the magnitude of output variation and the relative influence of each input. Derivatives provide an efficient local description for small perturbations, while global sampling methods capture nonlinear behavior over the full input distribution.

The treatment assumes probability, multivariable calculus, ordinary differential equations, Monte Carlo sampling, and automatic differentiation. Familiarity with numerical ODE solvers is used in the local examples.

We first propagate small parameter uncertainty through ODE solutions using sensitivity equations, differentiated solvers, and examples based on the Duffing and Lorenz systems. A Fokker-Planck equation then describes the full evolving density in a simple setting. Latin hypercube designs and Sobol sequences provide global sampling strategies. Functional analysis of variance (functional ANOVA) then decomposes output variation into contributions from individual inputs and their interactions; the theory defines first-order, interaction, and total-effect Sobol indices before the computational example estimates them. The resulting methods quantify output uncertainty and identify influential inputs at a computational cost suited to the model and uncertainty regime.
