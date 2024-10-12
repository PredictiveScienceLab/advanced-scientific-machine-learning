# Active Learning

Active learning is a machine learning paradigm that aims to reduce the amount of labeled data required to train a model. The idea is to train a model on a small set of labeled data, and then iteratively query the user for the labels of the most informative data points. The model is then retrained on the expanded labeled dataset, and the process is repeated until the model reaches a desired level of performance. Active learning is particularly useful in scenarios where labeling data is expensive or time-consuming.
In the context of building surrogate models, active learning can be used to reduce the number of high-fidelity simulations required to train the model. 

## The general setting

We will focus on regression tasks because they are of higher relevance in the context of surrogate modeling.
So, we want to learn a function $f$ from some input space $\mathcal{X}$ to the real numbers $\mathbb{R}$:

$$
    f: \mathcal{X} \to \mathbb{R}.
$$

We will follow a Bayesian approach.
Formally, we put a prior on the space of models:

$$
    f \sim p(f).
$$

If we are working with a generalized linear model or a neural network, this prior will be over the parameters of the model.
If we are working with a Gaussian process, this prior will be over the function space.

We have a likelihood function connecting observations to the model.

$$
    p(y|\mathbf{x}, f),
$$

typically a Gaussian distribution with a mean centered at the model prediction and a variance that accounts for the noise in the observations.
The variance can be fixed or learned from the data.
We do not show the dependence of of the likelihood on the noise variance for simplicity.

Suppose we have data:

$$
    \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n.
$$

We always assume that the observations are independent conditioned on the model, i.e.,

$$
p(\mathcal{D}|f) = \prod_{i=1}^n p(y_i|\mathbf{x}_i, f).
$$

The posterior over the models is:

$$
    p(f|\mathcal{D}) = \frac{p(\mathcal{D}|f) p(f)}{p(\mathcal{D})}.
$$

In the context of a parameteric model, this is just a finite dimensional probability density.
In the context of Gaussian process regression, it is the posterior Gaussian process.

Finally, we make predictions at an arbitrary point $\mathbf{x}$ using the posterior predictive probability density function (PDF)

$$
p(y|\mathbf{x}, \mathcal{D}) = \int p(y|\mathbf{x},f)p(f|\mathcal{D})\;Df.
$$

Here $Df$ denotes integration over all parameters (i.e., the regular type of integration).
For Gaussian process regression, $Df$ is a so-called functional (or path or Feynman) integral.
Think of it as an expectation over the probabilty measure defiend by the posterior Gaussian process.

It will be useful to define the mean and variance of the posterior predictive PDF:

$$
\mu(\mathbf{x}|\mathcal{D}) = \mathbb{E}[y|\mathbf{x},\mathcal{D}] = \int y p(y|\mathbf{x}, \mathcal{D}),
$$

and

$$
\sigma^2(\mathbf{x}|\mathcal{D}) = \mathbb{V}[y|\mathbf{x},\mathcal{D}] = \mathbb{E}[y^2|\mathbf{x},\mathcal{D}] - \left(\mathbb{E}[y|\mathbf{x},\mathcal{D}]\right)^2.
$$

Of course, these are analytically available for generalized linear models and Gaussian processes.
For non-linear models, e.g., neural networks, one would have to resort to the Laplace approximation over the posterior of the parameters or just use samples from the posterior.

## The general active learning paradigm

+ Start with a small set of labeled data (ideally, space filling in the input space):

$$
\mathcal{D}_{n_0} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_0}.
$$

+ Condition your model on $\mathcal{D}_{n_0}$. This means, you should be able to characterize the posterior,

    $$
    p_{n_0}(f) := p(f|\mathcal{D}_{n_0}),
    $$

    the posterior predictive mean:

    $$
    \mu_{n_0}(\mathbf{x}) := mu(\mathbf{x}|\mathcal{D}_{n_0}),
    $$

    the posterior predictive variance:

    $$
    \sigma^2_{n_0}(\mathbf{x}) := \sigma^2(\mathbf{x}|\mathcal{D}_{n_0}),
    $$

    etc.

+ For $t = n_{0}, n_{0}+1, \dots$:

    - Find the input $\mathbf{x}$ that maximizes an *information acquisition function* $\alpha_{t}(\mathbf{x})$:

        $$
            \mathbf{x}_{t+1} = \arg\max_{\mathbf{x}\in \mathcal{X}} \alpha_t(\mathbf{x}).
        $$

        This function captures how much value or information there is in making an observation at a given input. There are many options and we will talk about them shortly.
    
    - If the $\alpha(\mathbf{x}_{t+1})$ is smaller than a threshold. STOP.
    
    - Evaluate your information source to get the output:

        $$
            y_{t+1} = f_{\text{true}}(\mathbf{x}_{t+1}).
        $$

    - Add the new observation to your dataset:

        $$
            \mathcal{D}_{t+1} = \mathcal{D}_t \cup \{(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\}.
        $$
    
    - Condition your model on $\mathcal{D}_{t+1}$.

## Information theoretic acquisition functions

This approach is based on ideas developed by [MacKay, 1992](https://direct.mit.edu/neco/article-abstract/4/4/590/5648/Information-Based-Objective-Functions-for-Active?redirectedFrom=fulltext).
The idea is to pick $\alpha_t(\mathbf{x})$ to be the expected information gain about the model given the data.
We think as follows:

+ Our state of knowledge about the model is $p(f|\mathcal{D}_t)$.

+ Suppose we make a hypothetical observation at $\mathbf{x}$ and get the output $y$. Our state of knowledge about the model would change to:

    $$
        p(f|\mathcal{D}_t, \mathbf{x}, y) = \frac{p(y|\mathbf{x}, f, \mathcal{D}_t)p(f|\mathcal{D}_t)}{p(y|\mathbf{x}, \mathcal{D}_t)}.
    $$

+ The information gain is then the Kullback-Leibler divergence between the posterior before and after the observation:

    $$
        \operatorname{KL}\left[p(f|\mathcal{D}_t,\mathbf{x},y)\parallel p(f|\mathcal{D}_t)\right] = 
        \mathbb{E}\left[\log\left(\frac{p(f|\mathcal{D}_t,\mathbf{x},y)}{p(f|\mathcal{D}_t)}\right)\middle| \mathbf{x},y\right] =
        \int p(f|\mathcal{D}_t,\mathbf{x},y)\log\left(\frac{p(f|\mathcal{D}_t,\mathbf{x},y)}{p(f|\mathcal{D}_t)}\right)\;Df.
    $$

+ Since we do not know $y$, we take the expectation of the information gain over all possible values of $y$:

    $$
        \alpha_t(\mathbf{x}) = \mathbb{E}\left[\operatorname{KL}\left[p(f|\mathcal{D}_t,\mathbf{x},y)\parallel p(f|\mathcal{D}_t)\right]\middle|\mathbf{x}\right].
    $$

This is not analytically tractable.
However, MacKay proves a useful formula that connects the expected information gain to the expecteed difference in the entropy of the model distribution before and after the observation.
The differential entropy is:

$$
S_{t} = -\int p(f|\mathcal{D}_t)\log \frac{p(f|\mathcal{D}_t)}{m(f)}\;Df = -\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t)}{m(f)}\right],
$$

where $m(f)$ is a reference measure.

The differential entropy after the hypothetical observation is:

$$
S_{t+1}(x,y) = -\int p(f|\mathcal{D}_t, \mathbf{x}, y)\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\;Df = -\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\middle| \mathbf{x}, y\right].
$$

Using the properties of the conditional expectation, we have:

$$
\mathbb{E}\left[S_{t+1}(x,y)\middle| x\right] = -\mathbb{E}\left[\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\middle| \mathbf{x}, y\right]\middle| \mathbf{x}\right] = -\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\middle| \mathbf{x}\right].
$$

From this formula, we can see that the expected information gain is:

$$
\alpha_t(\mathbf{x}) = S_{t} - \mathbb{E}\left[S_{t+1}(x,y)\middle| x\right].
$$

Notice that the reference measure cancels out.
From this point on, MacKay approximates the model posterior with a Gaussian shows that the expected information gain can be approximated as:

$$
\alpha_t(\mathbf{x}) \approx \frac{1}{2}\frac{\sigma_t^2(\mathbf{x})}{\sigma^2}.
$$

The resulting approach is also known as *uncertainty sampling*.
Intuitively, we want to pick the next observation at the point where the model is most uncertain.
If the measurement noise $\sigma^2$ was a function of the input, we would have:

$$
\alpha_t(\mathbf{x}) \approx \frac{1}{2}\frac{\sigma_t^2(\mathbf{x})}{\sigma^2(\mathbf{x})}.
$$

In this case, we for the same epistemic uncertainty level, we would prefer to make observations in regions where the measurement noise is smaller.

Uncertainty sampling is known to put more emphasis on the boundaries of the input space.
This is because the model is more uncertain in these regions.
This is not always desirable.
MacKay in the paper cited above develops some other information acquisition functions that attempt to maximize the expected information gain about the model in a specific region of interest.
Another way to construct information acquisition fucntions is to think about the value of information.

## The value of information

Another way to construct a utility function is to think about the value of information.

+ Suppose we make a hypothetical observation at $\mathbf{x}$ and get the output $y$. Suppose that you have a utility function $u_t(x, y)$ that quantifies how much value you get from making the observation.
For example, it could be:

    $$
    u_t(x, y) = v_t(y) - c_t(x),
    $$

where $v_t(y)$ is the value of the output and $c_t(x)$ is the cost of making the observation.

+ The information gain is then the expected value of the utility function:

    $$
    \alpha_t(\mathbf{x}) = \mathbb{E}[u_t(\mathbf{x}, y)|\mathbf{x}] = \int u_t(\mathbf{x}, y)p(y|\mathbf{x}, \mathcal{D}_t)\;Dy.
    $$

The expected improvement information acquisition function is a special case of this utility function.

## Multi-fidelity active learning

In the context of multi-fidelity modeling our decision is not only where to make the next observation, but also at which fidelity level.
So, we have to pick the fidelity level $s$, say in $\{0,1\}$ if we have two levels, and the input $\mathbf{x}$.
The information acquisition function we construct must be of the form $\alpha_t(s, \mathbf{x})$.
The algorithm changes to:

+ Start with a dataset:

    $$
    \mathcal{D}_{n_0} = \{(\mathbf{x}_i, y_i, s_i)\}_{i=1}^{n_0}.
    $$

+ Condition your model on $\mathcal{D}_{n_0}$.

+ For $t = n_{0}, n_{0}+1, \dots$:

    - Find the input $\mathbf{x}$ and fidelity level $s$ that maximizes an *information acquisition function* $\alpha_t(s, \mathbf{x})$:

        $$
            (s_{t+1}, \mathbf{x}_{t+1}) = \arg\max_{s\in\{0,1\}, \mathbf{x}\in \mathcal{X}} \alpha_t(s, \mathbf{x}).
        $$

    - If the $\alpha(s_{t+1}, \mathbf{x}_{t+1})$ is smaller than a threshold. STOP.
    
    - Evaluate your information source to get the output:

        $$
            y_{t+1} = f_{s_{t+1}}(\mathbf{x}_{t+1}).
        $$

    - Add the new observation to your dataset:

        $$
            \mathcal{D}_{t+1} = \mathcal{D}_t \cup \{(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, s_{t+1})\}.
        $$
    
    - Condition your model on $\mathcal{D}_{t+1}$.


A simple example of a multi-fidelity information acquisition function is:

$$
\alpha_t(s, \mathbf{x}) = \lambda\frac{\sigma_{t,s}^2(\mathbf{x})}{\sigma_s^2} - (1-\lambda)c_s(\mathbf{x}),
$$

where $\lambda$ is a parameter that balances the value of information and the cost of making the observation, $\sigma_{t,s}^2(\mathbf{x})$ is the posterior predictive variance of the model at fidelity level $s$ at input $\mathbf{x}$, $\sigma_s^2$ is the measurement noise of the model at fidelity level $s$, and $c_s(\mathbf{x})$ is the cost of making an observation at fidelity level $s$.

## Active learning for multi-objective optimization

In multi-objective optimization, we want to find the Pareto front of a set of objectives.
One possible information acquisition function is the hypervolume improvement.
See [Beume et al., 2007](https://www.sciencedirect.com/science/article/pii/S0377221706005443) and [Pandita et al., 2018](https://www.dl.begellhouse.com/journals/52034eb04b657aea,2a63c994718e44bd,79dac56a2fbc871c.html) for more details.