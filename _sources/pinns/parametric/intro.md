# PINNs for Parametric Studies

A standard forward PINN produces one solution for one differential equation. Parameter studies require solutions across many coefficients, boundary conditions, or source terms. Retraining a separate network for every case repeats the training cost instead of sharing it across the family.

We first include the problem parameters among the network inputs and minimize an objective averaged over a chosen training distribution for those parameters. A parametric Poisson problem makes the construction explicit. A physics-informed neural operator then combines operator learning with a differential-equation residual to learn maps from forcing functions to solutions while using only a small set of paired forcing functions and solution values. The trained model approximates an entire family of solutions and can therefore replace many repeated solves.
