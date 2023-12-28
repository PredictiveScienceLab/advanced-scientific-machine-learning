# Differentiating the Solution of Ordinary Differential Equations

## Automatic differentiation of a numerical solver

The idea here is simple.
Write a standard ODE solver, e.g., Euler, Runga-Kutta, etc., and then use automatic differentiation to compute the gradient of the solution with respect to the parameters.
All you have to do is to write the solver in pure `jax`` and then use `jax.grad` to compute the gradient.
This is exactly what Patrick Kidger did with [Diffrax](https://github.com/patrick-kidger/diffrax).
We will use it in the next section.

## The method of adjoints

The method of adjoints helps us find the gradient of the solution of an ODE with respect to the parameters.
It yields another ODE, called the adjoint ODE, which we can solve to find the gradient.
Here is how it works.
Recall that the IVP is:

$$
\begin{align*}
\dot{x} &= f(x,t,\theta),\\
x(0;\theta) &= x_0(\theta),
\end{align*}
$$

and that we are looking for $\nabla_{\theta}x(t;\theta)$, the gradient of $x(t;\theta)$ with respect to $\theta$.

Start by taking the time derivative of $\nabla_{\theta}x(t;\theta)$:

$$
\begin{align*}
\dot{\nabla_{\theta}x(t;\theta)} &= \nabla_{\theta}\dot{x}(t;\theta)\\
&= \nabla_{\theta}f(x,t,\theta)\\
&= \partial_x f(x,t,\theta)\nabla_{\theta}x(t;\theta) + \nabla_{\theta}f(x,t,\theta).
\end{align*}
$$

We see that $\nabla_{\theta}x(t;\theta)$ satisfies the following IVP:

$$
\begin{align*}
\dot{\nabla_{\theta}x(t;\theta)} &= \partial_x f(x,t,\theta)\nabla_{\theta}x(t;\theta) + \nabla_{\theta}f(x,t,\theta),\\
\nabla_{\theta}x(0;\theta) &= \nabla_{\theta}x_0(\theta).
\end{align*}
$$

How do we solve this IVP?
Well, first use a standard ODE solver to solve the original IVP.
Then use the solution to solve the adjoint IVP with the same solver (or a different one).

## Which method should I use?

Some people claim that the method of adjoints is more efficient than automatic differentiation, see [this paper](https://arxiv.org/abs/2002.08071).
However, some other people claim the opposite.
It looks like the method of adjoints may be more efficient, but less accurate.
So it depends on your application.
For our application, local sensitivity analysis, we any of the two methods will work.