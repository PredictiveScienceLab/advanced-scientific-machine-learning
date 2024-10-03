# Local Sensitivity Analysis for Ordinary Differential Equations

Let us consider an initial value problem (IVP) of the form

$$
\begin{align*}
\dot{x} &= f(x,t,\theta),\\
x(0;\theta) &= x_0(\theta),
\end{align*}
$$

where $x$ is a vector representing the state of a dynamical system, $f(x,t,\theta)$ is a vector field representing the dynamics of the system, and $\theta$ is a vector of parameters. We assume that the initial condition $x_0(\theta)$ is a function of the parameters $\theta$.

Suppose that we are uncertain about the value of the parameters $\theta$. We can represent this uncertainty by a probability distribution $p(\theta)$ over the parameters.
How does this uncertainty affect the state of the system $x(t;\theta)$ at a later time $t$?

Local sensitivity analysis answers this question under certain conditions:

+ The uncertainty in the parameters is small. So, small that we can approximate it by a Gaussian distribution with mean $\theta$ and covariance matrix $\Sigma$:

$$
p(\theta) \approx \mathcal{N}(\theta|\mu,\Sigma).
$$

+ The vector field $f(x,u,\theta)$ is differentiable with respect to the parameters $\theta$.

+ The system is not chaotic. In other words, the state of the system does not diverge exponentially from the initial condition. We will see later how for chaotic system local sensitivity analysis fails.

The first step is to Taylor expand the solution $x(t;\theta)$ around the nominal parameter values $\theta$:

$$
x(t;\theta) = x(t;\mu) + \nabla_{\theta} x(t;\mu)(\theta-\mu) + \mathcal{O}\left(\parallel|\theta-\mu\parallel|^2\right)
\approx x(t;\mu) + \nabla_{\theta} x(t;\mu)(\theta-\mu).
$$

In the equation above, $\nabla_{\theta} x(t;\mu)$ is the gradient of $x(t;\mu)$ with respect to $\theta$.

Now, $\theta$ is a random variable. So, you can think of $X(t) = x(t;\theta)$ as a random variable as well.
As a matter of fact, to first order, $X(t)$ is just an affine transformation of $\theta$.
So, since $\theta$ is Gaussian, $X(t)$ is also Gaussian.
Let's find the expectation and the covariance:

$$
\begin{align*}
\mathbb{E}[x(t)] &= \mathbb{E}[x(t;\mu)] + \mathbb{E}[\nabla_{\theta}x(t;\mu)(\theta-\mu)]\\
&= x(t;\mu) + \nabla_{\theta}x(t;\mu)\mathbb{E}[\theta-\mu]\\
&= x(t;\mu),
\end{align*}
$$

where we have used the fact that $\mathbb{E}[\theta-\mu] = 0$.

The covariance between $X_i(t)$ and $X_j(t')$ is

$$
\begin{align*}
\mathrm{cov}[X_i(t),X_j(t')] &= \mathbb{E}[(X_i(t)-\mathbb{E}[X_i(t)])(X_j(t')-\mathbb{E}[X_j(t')])]\\
&= \mathbb{E}[(X_i(t)-x_i(t;\mu))(X_j(t')-x_j(t';\mu))]\\
&= \mathbb{E}[(\nabla_{\theta}x_i(t;\mu)(\theta-\mu))(\nabla_{\theta}x_j(t';\mu)(\theta-\mu))]\\
&= \nabla_{\theta}x_i(t;\mu)\mathbb{E}[(\theta-\mu)(\theta-\mu)^T]\nabla_{\theta}x_j(t';\mu)^T\\
&= \nabla_{\theta}x_i(t;\mu)\Sigma\nabla_{\theta}x_j(t';\mu)^T.
\end{align*}
$$

Putting everything together, we can write:

$$
X \sim \text{GP}(x(t;\mu),\nabla_{\theta}x(t;\mu)\Sigma\nabla_{\theta}x(t';\mu)^T).
$$

Pay attention to the fact that this is a **vector-valued Gaussian process**.
So **the covariance function is a matrix** giving the covariance between the components of $X(t)$ and $X(t')$.

Having expressed $X(t)$ as a Gaussian process, we can do all sort of things with it.
We can compute the variance, the covariance, the probability of $X(t)$ being in a certain region, etc.

There is only one problem: we need to compute $\nabla_{\theta}x(t;\mu)$.
We will talk about it in the next section.