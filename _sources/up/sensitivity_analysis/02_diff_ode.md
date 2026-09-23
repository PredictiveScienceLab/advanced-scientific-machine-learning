# Differentiating the Solution of Ordinary Differential Equations

Local uncertainty propagation requires derivatives of an ODE solution with respect to its parameters. These derivatives can be obtained by differentiating a numerical solver, by solving forward sensitivity equations, or, for a scalar objective, by solving an adjoint equation.

## Automatic differentiation of a numerical solver

A numerical solver defines a map from parameters to an approximate solution. If the solver is implemented with differentiable JAX operations, automatic differentiation can differentiate the executed numerical computation directly. The result is the derivative of the discrete approximation produced by the solver. [Diffrax](https://github.com/patrick-kidger/diffrax) provides differentiable ODE solvers that we will use in the Duffing oscillator example.

## Forward sensitivity equations

Consider the initial value problem of the [local sensitivity section](01_theory.md) with a parameter vector $\boldsymbol{\theta}\in\mathbb{R}^p$, and assume that its solution $\mathbf{x}(t;\boldsymbol{\theta})$ exists uniquely and depends differentiably on the parameters. Define the sensitivity matrix function $S:[0,T]\to\mathbb{R}^{n\times p}$ by

$$
S(t)=\frac{\partial\mathbf{x}(t;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}.
$$

Along the state trajectory, define the two Jacobian matrices

$$
f_{\mathbf{x}}(t)
=\frac{\partial\mathbf{f}}{\partial\mathbf{x}}
(\mathbf{x}(t;\boldsymbol{\theta}),t,\boldsymbol{\theta})
\in\mathbb{R}^{n\times n},
\qquad
f_{\boldsymbol{\theta}}(t)
=\frac{\partial\mathbf{f}}{\partial\boldsymbol{\theta}}
(\mathbf{x}(t;\boldsymbol{\theta}),t,\boldsymbol{\theta})
\in\mathbb{R}^{n\times p}.
$$

Differentiating the state equation with respect to $\boldsymbol{\theta}$ gives

$$
\dot{S}(t)
=f_{\mathbf{x}}(t)S(t)+f_{\boldsymbol{\theta}}(t),
\qquad
S(0)=\frac{\partial\mathbf{x}_0(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}.
$$

This initial value problem is the **forward sensitivity equation**, also called the **tangent equation**. It is not an adjoint equation. In practice, one solves the state and sensitivity equations together as an augmented forward system. The $p$ columns of $S$ give the response to perturbations in the $p$ parameter directions.

## The adjoint equation for a scalar objective

Forward sensitivities construct the derivative of the full state. When the quantity of interest is a scalar, an adjoint can obtain its parameter gradient without evolving one sensitivity column per parameter. Let $\Phi:\mathbb{R}^n\times\mathbb{R}^p\to\mathbb{R}$ be a differentiable terminal contribution, and let $L:\mathbb{R}^n\times[0,T]\times\mathbb{R}^p\to\mathbb{R}$ be a differentiable running contribution. Define the scalar objective $J:\mathbb{R}^p\to\mathbb{R}$ by

$$
J(\boldsymbol{\theta})
=\Phi(\mathbf{x}(T),\boldsymbol{\theta})
+\int_0^T L(\mathbf{x}(t),t,\boldsymbol{\theta})\,dt.
$$

All derivatives of $\Phi$ and $L$ below are evaluated along the state trajectory. Their derivatives with respect to $\boldsymbol{\theta}$ hold the state fixed, and $\nabla_{\mathbf{x}}$ and $\nabla_{\boldsymbol{\theta}}$ denote column gradients. The chain rule gives

$$
\nabla_{\boldsymbol{\theta}}J
=\nabla_{\boldsymbol{\theta}}\Phi
+S(T)^{\mathsf T}\nabla_{\mathbf{x}}\Phi
+\int_0^T
\left(
\nabla_{\boldsymbol{\theta}}L
+S(t)^{\mathsf T}\nabla_{\mathbf{x}}L
\right)dt.
$$

Define the adjoint $\boldsymbol{\lambda}:[0,T]\to\mathbb{R}^n$ by the terminal value problem

$$
-\dot{\boldsymbol{\lambda}}(t)
=f_{\mathbf{x}}(t)^{\mathsf T}\boldsymbol{\lambda}(t)
+\nabla_{\mathbf{x}}L(\mathbf{x}(t),t,\boldsymbol{\theta}),
\qquad
\boldsymbol{\lambda}(T)
=\nabla_{\mathbf{x}}\Phi(\mathbf{x}(T),\boldsymbol{\theta}).
$$

Combining the forward sensitivity and adjoint equations gives

$$
\frac{d}{dt}\left(S(t)^{\mathsf T}\boldsymbol{\lambda}(t)\right)
=f_{\boldsymbol{\theta}}(t)^{\mathsf T}\boldsymbol{\lambda}(t)
-S(t)^{\mathsf T}\nabla_{\mathbf{x}}L.
$$

Integrating this identity removes $S(t)$ from the objective gradient:

$$
\nabla_{\boldsymbol{\theta}}J
=\nabla_{\boldsymbol{\theta}}\Phi
+\left(\frac{\partial\mathbf{x}_0}{\partial\boldsymbol{\theta}}\right)^{\mathsf T}
\boldsymbol{\lambda}(0)
+\int_0^T
\left(
\nabla_{\boldsymbol{\theta}}L
+f_{\boldsymbol{\theta}}(t)^{\mathsf T}\boldsymbol{\lambda}(t)
\right)dt.
$$

Computing the gradient requires a forward state solve followed by a backward adjoint solve.

## Choosing a differentiation method

Forward sensitivities are attractive when the number of parameters is modest or when derivatives of many state outputs are required. Adjoint methods are attractive when many parameters influence a small number of scalar objectives. Automatic differentiation through a solver is often the simplest implementation and computes the derivative of the discrete solve; continuous forward and adjoint equations differentiate the ODE before discretization. Solver tolerances, interpolation, and adaptive-step logic can therefore affect the agreement between the two approaches.
