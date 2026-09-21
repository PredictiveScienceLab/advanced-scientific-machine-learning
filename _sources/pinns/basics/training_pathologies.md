# Training pathologies and remedies

Physics-informed neural networks optimize several objectives at once: a differential-equation residual, initial and boundary conditions, and sometimes observed data. These terms can have different units, scales, derivative orders, and convergence rates. A small total loss can therefore conceal a large physical error in one component. Reliable training begins by monitoring each term and its gradients separately.

## Unbalanced loss gradients

Write a representative objective as

$$
\mathcal{L}(\boldsymbol{\theta})
= \lambda_r\mathcal{L}_r(\boldsymbol{\theta})
+ \lambda_{ic}\mathcal{L}_{ic}(\boldsymbol{\theta})
+ \lambda_{bc}\mathcal{L}_{bc}(\boldsymbol{\theta})
+ \lambda_d\mathcal{L}_d(\boldsymbol{\theta}).
$$

The four terms enforce the residual, initial conditions, boundary conditions, and data, respectively. If

$$
\left\|\lambda_j\nabla_{\boldsymbol{\theta}}
\mathcal{L}_j\right\|
$$

is much larger for one term than for the others, its gradient controls the update. Training may then reduce the residual while violating a boundary condition, or fit the data while ignoring the governing equation. Comparing raw loss values is insufficient because equal loss values can produce very different parameter gradients.

Nondimensionalization is the first remedy. It removes unit-induced differences and places the inputs, outputs, and differential terms near comparable scales. The [forward PINN example](forward.ipynb) demonstrates this step. Loss weighting should be considered after the physical scaling has been corrected.

## Stiff gradient flow

The continuous-time idealization of gradient descent is

$$
\frac{d\boldsymbol{\theta}}{d\tau}
= -\nabla_{\boldsymbol{\theta}}
\mathcal{L}(\boldsymbol{\theta}).
$$

In a locally convex neighborhood, the positive eigenvalues of the loss Hessian describe parameter directions with different local time scales. PINN losses can span many such scales because differentiation amplifies some network modes while the constraints act on different subsets of the domain. An explicit optimizer step that is stable for a slowly varying direction may be too large for a rapidly varying one. The discrete loss can then oscillate or increase even though the idealized gradient flow decreases it {cite:p}`wang2021gradient`.

Loss curves, componentwise gradients, and the largest stable learning rate provide practical evidence of stiffness. Smaller learning rates, decay schedules, adaptive optimizers, and quasi-Newton refinement can help, but they do not repair a badly scaled formulation.

## Adaptive loss weighting

Adaptive weighting tries to keep every constraint active. Let

$$
g_j
= \left\|\nabla_{\boldsymbol{\theta}}
\mathcal{L}_j(\boldsymbol{\theta})\right\|,
\qquad
\bar g = \frac{1}{J}\sum_{j=1}^J g_j.
$$

A simple target weight is

$$
\widehat{\lambda}_j
= \frac{\bar g}{g_j+\varepsilon},
$$

followed by an exponential moving average of the current and target weights. Other methods use neural-tangent-kernel statistics or uncertainty-based weights. The purpose is the same: prevent one objective from silencing the others.

Gradient balancing is a training heuristic, not a change to the governing equation. Rapidly varying weights create a moving objective and can amplify noisy gradient estimates. The individual physical errors should therefore remain visible, and the final solution should be checked against every constraint independently. The broader training recommendations are summarized by {cite:t}`wang2023expert`.

## Causal residual weighting

Time-dependent problems have an additional structure: errors at early times propagate forward. Uniformly minimizing residuals over the full time interval can spend effort on late-time states before the earlier trajectory is accurate. Partition the interval into ordered times or windows and write

$$
\mathcal{L}_r(\boldsymbol{\theta})
= \frac{1}{N_t}\sum_{i=1}^{N_t}
w_i\mathcal{L}_r(t_i,\boldsymbol{\theta}).
$$

One causal choice is

$$
w_i
= \exp\!\left(
-\epsilon\sum_{k=1}^{i-1}
\mathcal{L}_r(t_k,\boldsymbol{\theta})
\right),
$$

with $\epsilon>0$ {cite:p}`wang2024causality`. A large early-time residual suppresses the weight of later intervals. As the early intervals become accurate, the optimization advances through time. This construction is appropriate for evolutionary problems; a stationary elliptic problem has no corresponding temporal order.

During a parameter update, the weights $w_i$ are treated as constants: compute
them from the current residuals and stop their gradients before differentiating
the weighted objective. Otherwise, the terms $\nabla_{\boldsymbol{\theta}}w_i$
enter the update, and the optimizer can reduce a late-time weight by increasing
an earlier residual instead of improving the trajectory. Recompute the detached
weights between parameter updates or training stages.

## Curriculum design and collocation sampling

A curriculum changes the training problem gradually. A time-dependent PINN may begin on a short interval and extend the terminal time after reaching an error threshold. A parametric PINN may begin with a narrow parameter range, and a multiscale problem may introduce higher-frequency content after the low-frequency structure has been learned. Each stage should inherit the same physical problem on its restricted domain, so that the curriculum does not silently change the target solution.

Collocation points can also be adapted. Residual-based sampling allocates more points where the current residual is large, while boundary layers and interfaces can be oversampled from the start. Adaptive points are selected using the training residual, so a fixed validation grid remains necessary. Otherwise, the same approximation can choose the points and certify its own accuracy.

## Architectural mitigations

Architecture can improve optimization after the formulation and scaling are sound. Fourier features help represent high-frequency components and are developed in the [spectral-bias notebook](spectral_bias.ipynb). Modified multilayer perceptrons with gated or residual pathways can improve gradient transport through depth. Random weight factorization separates the direction and scale of selected weight vectors, which can make useful scales easier to learn.

These changes alter the optimization geometry and inductive bias; none guarantees convergence to the physical solution. A useful comparison holds the collocation points and evaluation grid fixed, reports every loss component, and evaluates the resulting solution rather than the training objective alone.

## A diagnostic sequence

When a PINN fails to train, inspect the problem in the following order:

1. Verify the differential operator, signs, coefficients, and boundary or initial conditions.
2. Nondimensionalize the variables and balance physical scales.
3. Monitor each loss term and each weighted gradient norm.
4. Check the solution on a fixed grid that was not used to select collocation points.
5. Adjust the optimizer, loss weights, sampling scheme, or curriculum according to the observed failure.
6. Compare architectural changes only after the preceding diagnostics are controlled.

This order distinguishes errors in the mathematical formulation from limitations of the optimizer or neural representation.
