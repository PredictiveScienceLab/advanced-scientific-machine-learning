# Bayesian formulation to inverse problems

The Bayesian formulation is the gold standard for inverse problems.
You need the following ingredients:

+ Your scientific model $f$:

$$
f: X \rightarrow Y,
$$

+ Your data $y$.

+ A prior over the parameters $x$, say $p(x)$. The prior encodes your state of knowledge about the parameters before you see the data.

+ A likelihood function $p(y|x)$. The likelihood function encodes the probability of observing the data $y$ given the parameters $x$. It is a model of the measurement process. A very common likelihood is the Gaussian likelihood:

$$
p(y|x) = \mathcal{N}(y| f(x), \Sigma),
$$

where $\Sigma$ is the covariance matrix of the noise.
If $y$ are independent and identically distributed, then $\Sigma = \sigma^2 I$.
We can treat $\sigma^2$ as a hyperparameter and estimate it from the data -- just like the parameters $x$.

The Bayesian formulation of the inverse problem is to find the posterior distribution of the parameters $x$ given the data $y$:

$$
p(x|y) = \frac{p(y|x)p(x)}{p(y)}.
$$

The posterior quantifies your state of knowledge about the parameters after you have seen the data.

The denominator $p(y)$ is the marginal likelihood, which is the probability of observing the data $y$ under the assumption that your model is correct.
Other names for this quantity are the evidence or the marginal likelihood.
It can be used to select the best model among a set of competing models.
It is:

$$
p(y) = \int p(y|x)p(x)dx.
$$

Typically, it is intractable to compute the marginal likelihood.

The Bayesian solution is a posterior distribution rather than a point estimate.
Bayes' rule defines this distribution when the evidence is finite and positive.
In function-space inverse problems, additional assumptions on the prior, forward model, and likelihood are needed to ensure that the posterior is well posed {cite:p}`stuart2010inverse`.
The posterior quantifies uncertainty conditional on the assumed model. Structural model error requires an explicit discrepancy model; otherwise, parameter uncertainty can be confounded with model inadequacy {cite:p}`kennedy2001bayesian,brynjarsdottir2014discrepancy`.

The main computational challenge in Bayesian inversion is obtaining the posterior distribution.
There are three possibilities:

1. Analytical solution: This is possible for very simple models and likelihoods.

2. Sampling: This is the most general approach. You can use variants of Markov chain Monte Carlo (MCMC).

3. Optimization: Variational inference is a popular method. It approximates the posterior with a simpler distribution that is easier to work with.

We begin with the simplest of these techniques: the Laplace approximation.
