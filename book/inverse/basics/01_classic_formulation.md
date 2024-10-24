# The Classical Formulation of Inverse Problems

## Definition of inverse problems

Suppose that you have scientific model that predicts a quantity of interest.
Let's assume that this model has parameters that you do not know.
These parameters could be simple scalars (mass, spring constant, dumping coefficients, etc.) or it could be also be functions (initial conditions, boundary values, spatially distributed constitutive relations, etc.)
In the case of the latter, we assume that you have already reduced the dimensionality of the parameterization with, for example, the Karhunen-Lo\`eve expansion.
Let's denote all these parameters with the vector $x$.
Assume that:

$$
x\in X \subset\mathbb{R}^d.
$$

Now, let's say we perform an experiment that measures a *noisy* vector:

$$
y\in Y\subset \mathbb{R}^m.
$$

Assume that, you can use your model *model* to predict $y$.
It does not matter how complicated your model is.
It could be a system of ordinary differential or partial differential equations, or something more complicated.
If it predicts $y$, you can always think of it as a function from the unknown parameter space $X$ to the space of $y$'s, $Y$.
That is, you can think of it as giving rise to a function:

$$
f : X \rightarrow Y.
$$

The **inverse problem**, otherwise known as the **model calibration** problem is to find the ``best`` $x\in X$ so that:

$$
f(x) \approx y.
$$

## Formulation of inverse problems as optimization problems
Saying that $f(x)\approx y$ is not an exact mathematical statement.
What does it really mean for $f(x)$ to be close to $y$?
To quantify this, let us introduce a *loss metric*:

$$
\ell: Y \times Y \rightarrow \mathbb{R}.
$$

such that $\ell(f(x),y)$ is how much our prediction is off if we chose the input $x$.
Equiped with this loss metric, we can formulate the mathematical problem as:

$$
\min_{x\in X} \ell(f(x),y).
$$

### The Square Loss
The choice of the loss metric is somewhat subjective.
However, a very common assumption is that to take the *square loss*:

$$
\ell(f(x), y) = \parallel f(x) - y\parallel_2^2 = \sum_{i=1}^m\left(f_i(x)-y_i\right)^2.
$$

For this case, the inverse problem can be formulated as:

$$
\min_{x\in X}\parallel f(x) - y\parallel_2^2.
$$

### Solution methodologies
We basically have to solve an optimization problem.
For the square loss function, if $f(x)$ is linear, then you get the classic least squares problem which has a known solution.
Otherwise, you get what is known as *generalized least squares*.
Let's discuss two possibilities for the most general case:

#### Case 1: Good for ODEs and simple PDEs

+ Implement your model from scratch in a differential programming framework like JAX.
+ Use automatic differentiation to compute the gradient of the loss function.
+ Use a gradient-based optimization algorithm like L-BFGS-B to solve the optimization problem.

#### Case 2: Good for legacy codes

+ Build a computationally inexpensive surrogate model for $f(x)$.
+ Make sure the surrogate modeling is done in a differentiable programming framework.
+ Use automatic differentiation to compute the gradient of the loss function.
+ Use a gradient-based optimization algorithm like L-BFGS-B to solve the optimization problem.

