# PDE Solvers as Operators

Consider the elliptic PDE

$$
-\nabla \cdot (a(x) \nabla u(x)) = f(x) \quad \text{in } \Omega
$$

with Dirichlet boundary conditions

$$
u(x) = g(x) \quad \text{on } \partial \Omega
$$

where $\Omega$ is a domain in $\mathbb{R}^d$ and $a, f, g$ are scalar functions on $\Omega$.

Let's think a bit about the solver.
What sort of object is it?
Consider the space of scalar functions on $\Omega$:

$$
\mathcal{U} = \left\{ u: \Omega \to \mathbb{R} | \int_\Omega \left(f^2 + |\nabla f|^2\right) < \infty \right\}.
$$

Similarly, let $\mathcal{A}$, $\mathcal{F}$, and $\mathcal{G}$ be the spaces of scalar functions on $\Omega$ for $a$, $f$, and $g$ respectively.
The solver is a map

$$
S: \mathcal{A} \times \mathcal{F} \times \mathcal{G} \to \mathcal{U}
$$

that takes the coefficients $a$, $f$, and $g$ and returns the solution $u$:

$$
(a, f, g) \mapsto u.
$$

We call maps with functional inputs and outputs **operators**.
The solver is an operator that maps the coefficients to the solution.

## Uncertainty in functional inputs

We may be uncertain about the functioanal inputs to a scientific model.
For example, we may not know the exact values of the thermal conductivity $k$.
We can model this uncertainty by considering $k$ as a random field.
Similarly for $f$ and $g$.

The most common choice of random field is a Gaussian process.
Now, the thermal conductivity $k$ is positive, so we can model it as a log-Gaussian process.
We say:

$$
a = \exp\{h\},
$$

and we can set

$$
h \sim \operatorname{GP}(m, k),
$$

where $m$ is a mean function and $k$ is a covariance function.
Recall that you can use the mean function to encode information about trends and the covariance function to encode information about smoothness.

## Uncertainty propagation with Monte Carlo

Recall that you can sample Gaussian random fields wherever you like.
So, you can sample $h$ at a set of points $\{x_i\}_{i=1}^n$.
Then, you can compute the thermal conductivity at these points:

$$
a(x_i) = \exp\{h(x_i)\},
$$

where the function values:

$$
\mathbf{h} = \begin{bmatrix} h(x_1) \\ \vdots \\ h(x_n) \end{bmatrix},
$$

are drawn from a multivariate normal distribution:

$$
\mathbf{h} \sim \mathcal{N}(\mathbf{m}, \mathbf{K}),
$$

with 

$$
\mathbf{m} = \begin{bmatrix} m(x_1) \\ \vdots \\ m(x_n) \end{bmatrix}
$$

and

$$
\mathbf{C} = \begin{bmatrix} k(x_1, x_1) & \cdots & k(x_1, x_n) \\ \vdots & \ddots & \vdots \\ k(x_n, x_1) & \cdots & k(x_n, x_n) \end{bmatrix}.
$$

If you pick the points $\{x_i\}_{i=1}^n$ to be the nodes of a finite element mesh, then you can compute the solution $u$ at these points.
You can then use the finite element solver to interpolate the solution to the entire domain.
You can repeat this process many times to get a Monte Carlo estimate of the statistics of the solution.

## Finite Coordinates for Functional Inputs

Sampling a field at $n$ mesh points replaces one functional input with a vector in $\mathbb{R}^n$. When $n$ is large, a surrogate constructed directly on those values must learn in a high-dimensional input space, which can require many model evaluations. The next step is to approximate the field with a shorter vector of coordinates that retains its dominant variation.

Singular value decomposition provides the low-rank matrix construction used to identify such coordinates. Principal component analysis and the Karhunen--Loève expansion then provide data-driven and covariance-driven representations, respectively. These finite coordinate systems turn a function-valued uncertainty problem into a lower-dimensional uncertainty-propagation problem.
