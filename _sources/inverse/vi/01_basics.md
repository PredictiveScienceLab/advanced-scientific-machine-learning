# Basics of Variational Inference

Our problem is to approximate the posterior distribution of the parameters of a model given some observed data.
As usual, the parameters are $x$ with prior $p(x)$ and the data are $y$ with likelihood $p(y|x)$.

The posterior is given by Bayes' theorem:

$$
p(x|y) = \frac{p(y|x)p(x)}{p(y)} = \frac{p(x,y)}{p(y)}.
$$

Here, for later use, we defined the *joint* distribution $p(x,y) = p(y|x)p(x)$ and the *marginal* distribution, or *evidence*, $p(y) = \int p(x,y) dx$.

The idea in variational inference (VI) is to approximate the posterior $p(x|y)$ with a simpler distribution $q_\phi(x)$ that depends on some parameters $\phi$.
We often call $\phi$ the *variational parameters* and the distribution $q_\phi(x)$ the *variational distribution* or the *guide*.

To identify the best parameters $\phi$ for the guide, we want to minimize some sort of distance between the posterior and the guide.
We use the KL divergence for this purpose:

$$
\text{KL}(q_\phi(x)\parallel p(x|y)) = \int q_\phi(x) \log \frac{q_\phi(x)}{p(x|y)} dx.
$$

We should mention that the KL divergence is not a distance, but a divergence.
It is not symmetric and does not satisfy the triangle inequality.

So, the problem that VI solves is:

$$
\min_\phi \text{KL}(q_\phi(x)\parallel p(x|y)).
$$

Solving this problem is easier said than done.
There are a lot of issues to resolve.
We will need to come up with some good choices for the guide.
We will have to show that the optimization problem does indeed make sense.
That is, that the KL divergence has a minimum and that if we achieve it get do get closer to the posterior.
Finally, we will have to come up with a scalable algorithm to actually converges to the minimum.

## On the choice of the guide

### Guide example 1: Gaussian with diagonal covariance

One of the simplest guides we can use is a Gaussian distribution with a diagonal covariance matrix:

$$q_\phi(x) = \mathcal{N}(x|\mu, \text{diag}(\sigma^2))$$

When implementing this guide, we prefer to work with unconstrained parameters. Since $\sigma$ must be positive, we parameterize it as:

$$\lambda = \log \sigma$$

The variational parameters are then:

$$\phi = (\mu, \lambda)$$

### Guide example 2: Gaussian with low-rank covariance

We can extend the previous example to a Gaussian with a low-rank covariance matrix:

$$q_\phi(x) = \mathcal{N}(x|\mu, \Sigma)$$

where the covariance matrix has a low-rank structure:

$$\Sigma = \sum_{i=1}^k e^{\lambda_i} u_i u_i^T$$

Here, $k$ is much smaller than the dimension of $x$. The variational parameters are:

$$\phi = (\mu, \lambda_1, \ldots, \lambda_k, u_1, \ldots, u_k)$$

### Guide example 3: Gaussian with full covariance

For a more flexible guide, we can use a Gaussian with a full covariance matrix:

$$q_\phi(x) = \mathcal{N}(x|\mu, \Sigma)$$

To ensure that $\Sigma$ is positive definite, we parameterize it using a Cholesky decomposition:

$$\Sigma = L L^T$$

where $L$ is a lower triangular matrix:

$$L = \begin{bmatrix}
    \exp(\lambda_1) & 0 & \cdots & 0 \\
    * & \exp(\lambda_2) & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    * & * & \cdots & \exp(\lambda_d)
\end{bmatrix}$$

The diagonal entries are parameterized as exponentials to ensure they're positive, while the $d(d-1)/2$ entries below the diagonal (marked with *) are unconstrained. We'll denote these unconstrained entries as $u$. The variational parameters are:

$$\phi = (\mu, \lambda_1, \ldots, \lambda_d, u)$$

### Transformed Gaussian guides

Sometimes, the parameter space has constraints that make a direct Gaussian approximation inappropriate. In these cases, we can use a transformation approach:

1. Define a one-to-one transformation $T$ of the parameters
2. Define the transformed parameters: $z = T(x)$
3. Put a Gaussian guide on $z$: $\tilde{q}_\phi(z) = \mathcal{N}(z|\mu, \Sigma)$

The guide for $x$ is then:

$$q_\phi(x) = \tilde{q}_\phi(T(x)) |J_T(x)|$$

where $J_T(x) = \frac{\partial z}{\partial x}$ is the Jacobian of the transformation.

### Example 4: Guide for positive variables

If $x$ is a scalar and must be positive, we can use the transformation:

$$z = \log x$$

The Jacobian is:

$$J_T(x) = \frac{d}{dx} \log x = \frac{1}{x}$$

The guide for $x$ is then:

$$q_\phi(x) = \tilde{q}_\phi(\log x) \frac{1}{x}$$

Many probabilistic programming frameworks like Pyro handle these transformations automatically.

### Example 5: Guide for variables in [0,1]

If $x$ is a scalar constrained to the interval $[0, 1]$, we can use the logit transformation:

$$z = \text{logit}(x) = \log \left( \frac{x}{1-x} \right)$$

The Jacobian is:

$$J_T(x) = \frac{d}{dx} \text{logit}(x) = \frac{1}{x(1-x)}$$

The guide for $x$ is then:

$$q_\phi(x) = \tilde{q}_\phi(\text{logit}(x)) \frac{1}{x(1-x)}$$

### Example 6: Non-Gaussian guides

The guide doesn't have to be Gaussian. We can choose distributions that match the constraints of our parameters:

For a positive scalar $x$, we might use a Gamma distribution:

$$q_\phi(x) = \text{Gamma}(x|\alpha, \beta)$$

with variational parameters $\phi = (\alpha, \beta)$.

For a scalar $x$ between 0 and 1, a Beta distribution might be appropriate:

$$q_\phi(x) = \text{Beta}(x|\alpha, \beta)$$

with variational parameters $\phi = (\alpha, \beta)$.

### Example 7: Composite guides

For multivariate parameters with different constraints, we can combine different distributions:

If $x = (x_1, x_2)$ with $x_1$ positive and $x_2$ between 0 and 1, we might use:

$$q_\phi(x) = \text{Gamma}(x_1|\alpha_1, \beta_1) \text{Beta}(x_2|\alpha_2, \beta_2)$$

with variational parameters $\phi = (\alpha_1, \beta_1, \alpha_2, \beta_2)$.

### Structured guides

The guide can be adapted to match the structure of the model:

- If the model has a hierarchical structure, the guide can have the same structure
- This approach can capture dependencies between parameters more effectively
- We'll explore this further when discussing hierarchical models

## The optimization problem to fit the guide

### The "distance" between the guide and the posterior

We need a distance between the guide and the posterior. We use the Kullback-Leibler (KL) divergence:

$$\text{KL}(q_\phi(x) || p(x|y)) = \int q_\phi(x) \log \frac{q_\phi(x)}{p(x|y)} dx = \mathbb{E}_{q_\phi(x)} \left[ \log \frac{q_\phi(x)}{p(x|y)} \right]$$

It measures how much information is lost when we use $q_\phi(x)$ to approximate $p(x|y)$.

### Some basic properties of the KL divergence

The KL divergence has several important properties:

- It is non-negative:
  $$\text{KL}(q_\phi(x) || p(x|y)) \geq 0$$

- It is zero if and only if $q_\phi(x) = p(x|y)$

- But it is not a proper distance, because it is not symmetric:
  $$\text{KL}(q_\phi(x) || p(x|y)) \neq \text{KL}(p(x|y) || q_\phi(x))$$

### Proof of the KL properties

Recall Jensen's inequality:

$$\mathbb{E}[f(x)] \geq f(\mathbb{E}[x])$$

if $f$ is convex. If $f$ is strictly convex, then equality holds if and only if $x$ is a constant.

Equipped with Jensen's inequality, we can prove that the KL divergence is non-negative. Here is the proof:

$$
\begin{array}{rcl}
    \text{KL}(q_\phi(x) || p(x|y)) &=& \mathbb{E}_{q_\phi(x)} \left[ \log \frac{q_\phi(x)}{p(x|y)} \right] \\
    &=& - \mathbb{E}_{q_\phi(x)} \left[ \log \frac{p(x|y)}{q_\phi(x)} \right] \\
    &\geq& - \log \mathbb{E}_{q_\phi(x)} \left[ \frac{p(x|y)}{q_\phi(x)} \right] \\
    &=& - \log \int q_\phi(x) \frac{p(x|y)}{q_\phi(x)} dx \\
    &=& - \log \int p(x|y) dx\\
    &=& 0.
\end{array}
$$

We used Jensen's inequality with the convex function $f(x) = -\log x$ and the fact that $p(x|y)$ is a probability distribution and thus integrates to 1.

The other property of the KL divergence is that it is zero if and only if $q_\phi(x) = p(x|y)$.
We can see this from the fact that $f(x) = -\log x$ is *strictly* convex and thus Jensen's inequality is an equality if and only if the ratio $\frac{p(x|y)}{q_\phi(x)}$ is constant.
This means that the two terms are equal up to a constant factor.
But the constant factor has to be one because they are both probability distributions.

### Derivation of the evidence lower bound (ELBO)

In practice, we don't minimize the KL divergence directly.
Instead, we maximize the evidence lower bound (ELBO).
Minimizing the KL divergence is equivalent to maximizing the ELBO.

Start with the fact that the KL divergence is non-negative:

$$\text{KL}(q_\phi(x) || p(x|y)) \geq 0$$

Use the definition of the KL divergence and the fact that $p(x|y) = \frac{p(x,y)}{p(y)}$:

$$
\begin{array}{rcl}
    \text{KL}(q_\phi(x) || p(x|y)) &=& \mathbb{E}_{q_\phi(x)} \left[ \log \frac{q_\phi(x)}{p(x|y)} \right] \\
    &=& \mathbb{E}_{q_\phi(x)} \left[ \log \frac{q_\phi(x) p(y)}{p(x,y)} \right] \\
    &=& \mathbb{E}_{q_\phi(x)} \left[ \log q_\phi(x) \right] - \mathbb{E}_{q_\phi(x)} \left[ \log p(x,y) \right] + \log p(y)
\end{array}
$$

Since this is non-negative, we can rearrange it to get:

$$
\log p(y) \ge \mathbb{E}_{q_\phi(x)} \left[ \log p(x,y) \right] - \mathbb{E}_{q_\phi(x)} \left[ \log q_\phi(x) \right]
$$

From the long equation above, we see that the KL divergence and the ELBO are related by:

$$
\text{KL}(q_\phi(x) || p(x|y)) = \log p(y) - \text{ELBO}(\phi)
$$

Now think that you are maximizing the ELBO:

$$\max_\phi \text{ELBO}(\phi)$$

What would this do to the KL divergence?
You are pushing the ELBO up, which closes the gap between the ELBO and the evidence.
This reduces the KL divergence.
So, indeed, maximizing the ELBO minimizes the KL divergence.

## The reparameterization trick

To optimize the ELBO with respect to the variational parameters, we will need gradients like:

$$\nabla_\phi \mathbb{E}_{q_\phi(x)} \left[ f_\phi(x) \right] = \nabla_\phi \int q_\phi(x) f_\phi(x) dx$$

But there is a problem: this is not an expectation over $q_\phi(x)$.
So we cannot use standard Monte Carlo methods to estimate the gradient.

To overcome this, we use the reparameterization trick. 
This is an idea found in [Kingma and Welling, 2014](https://arxiv.org/pdf/1312.6114).
The idea is to express the variable $x$ as a deterministic function of a random variable $\epsilon$ drawn from a fixed distribution (without parameters).
Like this:

  $$x = g_\phi(\epsilon)$$

where $\epsilon \sim p(\epsilon)$, a fixed distribution, and $g_\phi$ is one-to-one transformation.
Then we can express the expectation over $q_\phi(x)$ as an expectation over $p(\epsilon)$:
  
$$\mathbb{E}_{q_\phi(x)} \left[ f_\phi(x) \right] = \mathbb{E}_{p(\epsilon)} \left[ f_\phi(g_\phi(\epsilon)) \right]$$

Now, there is no problem taking the gradient:
  
  $$\nabla_\phi \mathbb{E}_{q_\phi(x)} \left[ f_\phi(x) \right] = \nabla_\phi \mathbb{E}_{p(\epsilon)} \left[ f_\phi(g_\phi(\epsilon)) \right] = \mathbb{E}_{p(\epsilon)} \left[ \nabla_\phi f_\phi(g_\phi(\epsilon)) \right]$$

And we can easily construct a sampling average approximation:

$$
\nabla_\phi \mathbb{E}_{q_\phi(x)} \left[ f_\phi(x) \right] \approx \frac{1}{S} \sum_{s=1}^S \nabla_\phi f_\phi(g_\phi(\epsilon_s))
$$

## ELBO maximization as a stochastic optimization problem

We like minimizing things instead of maximizing things.
So we define a "loss" function:

$$
\mathcal{L}(\phi) = - \text{ELBO}(\phi)
$$

Use the reparameterization trick to approximate the expectations over $q_\phi(x)$ by expectations over $p(\epsilon)$.
Then construct a sampling average approximation of the loss:
  
$$\hat{\mathcal{L}}(\phi) = -\frac{1}{S}\sum_{s=1}^S \left[ \log p(g_\phi(\epsilon_s), y) - \log q_\phi(g_\phi(\epsilon_s)) \right]$$

where $\epsilon_s \sim p(\epsilon)$ independently.
At this point, we can use any standard stochastic optimization method.
We can use Adam, for example.

Let's go over some specific examples of how to apply the reparameterization trick.

### Example 1: The reparameterization trick for a univariate Gaussian guide

Suppose

$$q_\phi(x) = \mathcal{N}(x|\mu, \Sigma)$$
  
  with
  
$$\Sigma = L L^T$$

We can take:

$$x = g_{\phi}(\epsilon) = \mu + L \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$.

### Example 2: The reparameterization trick for a multivariate Gaussian guide


Suppose
  
$$q_\phi(x) = \tilde{q}_\phi(T(x)) |J_T(x)|$$

where
  
$$\tilde{q}_\phi(z) = \mathcal{N}(z|\mu, \Sigma)$$
  $T$ is a one-to-one transformation, and $J_T(x)$ is the Jacobian.

Then we can take:
  
$$z = g_{\phi}(\epsilon) = \mu + L \epsilon$$

and thus:

$$x = T^{-1}(g_{\phi}(\epsilon))$$

where $\epsilon \sim \mathcal{N}(0, I)$.



