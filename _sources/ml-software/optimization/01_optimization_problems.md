# Basics of Optimization Problems

Model training and inference in scientific machine learning often ask us to choose adjustable parameters so that a scalar score is as small as possible. Let $n$ be the number of adjustable model parameters and let $\mathbb{R}^n$ denote the $n$-dimensional real vector space. Let $U\subseteq\mathbb{R}^n$ be the set of admissible parameter vectors, and write a parameter vector as $x=(x_1,\ldots,x_n)\in U$. An objective function $f:U\to\mathbb{R}$ assigns the scalar score $f(x)$ to each admissible vector $x$. The task of finding the smallest attainable objective value is written as

$$
\min_{x\in U} f(x).
$$

Any admissible vector that attains this smallest value is called a global minimizer. The objective may measure mismatch between model predictions and data, penalize parameter values that violate prior knowledge, or combine several modeling goals. Scientific machine learning makes these problems difficult because $n$ can be large, $f$ can have many peaks and valleys or regions where it changes very little, and evaluating $f$ may require a costly scientific computation.

## Stationary points and local minima

For $x,y\in\mathbb{R}^n$, the expression $\lVert x-y\rVert_2$ denotes their Euclidean distance. A point $x_*\in U$ is an interior point if there is a radius $\rho>0$ such that every $x\in\mathbb{R}^n$ satisfying $\lVert x-x_*\rVert_2<\rho$ also belongs to $U$.

The following statements concern an interior point $x_*$ and a function $f$ that has continuous first and second derivatives near $x_*$. The gradient $\nabla f(x_*)$ is the vector of first partial derivatives. The point $x_*$ is stationary if the gradient equals the zero vector $\boldsymbol{0}\in\mathbb{R}^n$:

$$
\nabla f(x_*)=\boldsymbol{0}.
$$

The point $x_*$ is a local minimum if there is a radius $r>0$ such that

$$
f(x_*)\leq f(x)
$$

for every $x\in U$ satisfying $\lVert x-x_*\rVert_2<r$. A local maximum is defined by reversing the inequality.

The Hessian $H(x_*)=\nabla^2f(x_*)$ is the matrix of second partial derivatives and describes the local curvature of $f$. For a direction $v\in\mathbb{R}^n$, let $v^\mathsf{T}$ denote its transpose. The Hessian is positive semidefinite if

$$
v^\mathsf{T}H(x_*)v\geq 0
$$

for every $v\in\mathbb{R}^n$. At an interior local minimum, the gradient is zero and the Hessian is positive semidefinite. These conditions are necessary but not sufficient.

The Hessian is positive definite if the inequality is strict for every nonzero $v$. At a stationary point, a positive-definite Hessian guarantees a strict local minimum: there is a radius $r>0$ such that $f(x_*)<f(x)$ whenever $x\in U$ and $0<\lVert x-x_*\rVert_2<r$. A negative-definite Hessian, for which $v^\mathsf{T}H(x_*)v<0$ for every nonzero $v$, guarantees a strict local maximum, defined by reversing this strict inequality. These conclusions apply only under the interior-point and differentiability assumptions stated above.

## Saddle points

A saddle point is a stationary point $x_*$ such that, for every radius $r>0$, there are points $x^+,x^-\in U$ with $\lVert x^+-x_*\rVert_2<r$ and $\lVert x^--x_*\rVert_2<r$ for which $f(x^-)<f(x_*)<f(x^+)$. Thus, arbitrarily close points have objective values both below and above the value at the saddle point.

The Hessian is indefinite if there are directions $v,w\in\mathbb{R}^n$ for which

$$
v^\mathsf{T}H(x_*)v>0
\qquad\text{and}\qquad
w^\mathsf{T}H(x_*)w<0.
$$

At a stationary point, an indefinite Hessian guarantees a saddle point. A positive-semidefinite Hessian that is not positive definite does not by itself determine the type of stationary point. For example, let $n=2$ and consider the function $f:\mathbb{R}^2\to\mathbb{R}$ defined by

$$
f(x_1,x_2)=x_1^2-x_2^4
$$

This function has the positive-semidefinite Hessian

$$
H(0,0)=
\begin{pmatrix}
2 & 0\\
0 & 0
\end{pmatrix},
$$

but the origin is a saddle point because the function increases along the $x_1$-axis and decreases along the $x_2$-axis.

## Geometry of machine-learning objectives

Different parameter vectors can sometimes produce the same model predictions and therefore the same objective value. This explains why distinct points in $U$ may represent equivalent solutions, but it does not imply that every local minimum is satisfactory. A smaller objective value also need not imply more accurate predictions on new data or greater scientific fidelity.

High-dimensional objectives can contain saddle points, directions in which $f$ changes very little, and other directions in which it changes rapidly. These features can make it difficult for a procedure that searches over $U$ to choose its next candidate and the distance to move. Their prevalence and effect depend on the model and data. The following sections introduce specific search procedures one at a time and explain how their behavior depends on this local shape. A broader treatment of optimization for machine learning appears in {cite:t}`goodfellow2016deep`.
