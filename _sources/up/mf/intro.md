# Multi-fidelity Surrogates

A surrogate trained only on high-fidelity simulations may still require more high-fidelity evaluations than the available budget permits. Many scientific problems also provide cheaper approximations, such as coarse-mesh solvers or simplified physical models, whose systematic relationship to the high-fidelity response can supply additional information {cite:p}`peherstorfer2018multifidelity`.

The treatment assumes familiarity with surrogate-modeling workflows and Gaussian process regression, including covariance kernels, posterior prediction, and predictive uncertainty. We use low- and high-fidelity observations whose input designs need not coincide.

We first formulate joint inference for low- and high-fidelity functions and develop discrepancy-based and nonlinear Gaussian process couplings. Numerical examples then combine many low-cost evaluations with a small high-fidelity data set, including a flow application. The resulting cross-fidelity predictive distribution can improve high-fidelity prediction within a fixed computational budget and guide the choice of both the next input and the fidelity at which to evaluate it.
