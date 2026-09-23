# Sensitivity Analysis of ODEs

Uncertain inputs affect a scientific model through both the magnitude of output variation and the relative influence of each input. Derivatives provide an efficient local description for small perturbations, while global sampling methods capture nonlinear behavior over the full input distribution.

The treatment assumes probability, multivariable calculus, ordinary differential equations, Monte Carlo sampling, and automatic differentiation. Familiarity with numerical ODE solvers is used in the local examples.

We first propagate small parameter uncertainty through ODE solutions using sensitivity equations, differentiated solvers, and the Duffing oscillator. Companion notebooks examine the Lorenz system and density transport through a Fokker--Planck equation. Latin hypercube designs and Sobol sequences distribute model evaluations across the input distribution. Functional analysis of variance (functional ANOVA) then decomposes output variation into contributions from individual inputs and their interactions. Its Sobol sensitivity indices measure input importance; they are distinct from the Sobol sequences used to choose evaluation points. Together, these methods quantify output uncertainty and identify influential inputs at a computational cost suited to the model and uncertainty regime.
