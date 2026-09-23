# Active Learning Basics

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

We have a likelihood function connecting observations to the model,

$$
    p(y|\mathbf{x}, f),
$$

typically a Gaussian distribution with a mean centered at the model prediction and a variance that accounts for the noise in the observations.
The variance can be fixed or learned from the data.
We do not show the dependence of the likelihood on the noise variance for simplicity.

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

In the context of a parametric model, this is just a finite-dimensional probability density.
In the context of Gaussian process regression, it is the posterior Gaussian process.

Finally, we make predictions at an arbitrary point $\mathbf{x}$ using the posterior predictive density

$$
p(y|\mathbf{x}, \mathcal{D}) = \int p(y|\mathbf{x},f)p(f|\mathcal{D})\;Df.
$$

Here $Df$ means that we average over the uncertain model: over its parameters for a parametric model, or over random functions for a Gaussian process. In either case, the predictive density is an expectation under the model's posterior distribution.

It will be useful to define the mean and variance of the posterior predictive density:

$$
\mu(\mathbf{x}|\mathcal{D}) = \mathbb{E}[y|\mathbf{x},\mathcal{D}] = \int y\,p(y|\mathbf{x}, \mathcal{D})\,dy,
$$

and

$$
\sigma^2(\mathbf{x}|\mathcal{D}) = \operatorname{Var}[y|\mathbf{x},\mathcal{D}] = \mathbb{E}[y^2|\mathbf{x},\mathcal{D}] - \left(\mathbb{E}[y|\mathbf{x},\mathcal{D}]\right)^2.
$$

These moments have closed forms for Gaussian linear regression with a Gaussian prior and for Gaussian process regression with Gaussian observations and fixed hyperparameters.
Other likelihoods, nonlinear models such as neural networks, or integration over uncertain hyperparameters generally require an approximation or posterior samples.

## The general active learning paradigm

+ Start with a small set of labeled data (ideally, space filling in the input space):

    $$
    \mathcal{D}_{n_0} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_0}.
    $$

+ Condition your model on $\mathcal{D}_{n_0}$. This means you should be able to characterize the posterior,

    $$
    p_{n_0}(f) := p(f|\mathcal{D}_{n_0}),
    $$

    the posterior mean of the latent model response:

    $$
    \mu_{n_0}(\mathbf{x}) := \mathbb{E}[f(\mathbf{x})|\mathcal{D}_{n_0}],
    $$

    and its posterior variance:

    $$
    \sigma^2_{n_0}(\mathbf{x}) := \operatorname{Var}[f(\mathbf{x})|\mathcal{D}_{n_0}],
    $$

    which excludes the measurement-noise variance.

+ For $t = n_{0}, n_{0}+1, \dots$:

    - Find the input $\mathbf{x}$ that maximizes an *information acquisition function* $\alpha_{t}(\mathbf{x})$:

        $$
            \mathbf{x}_{t+1} = \arg\max_{\mathbf{x}\in \mathcal{X}} \alpha_t(\mathbf{x}).
        $$

        This function captures how much value or information there is in making an observation at a given input. There are many options and we will talk about them shortly.
    
    - If $\alpha_t(\mathbf{x}_{t+1})$ is smaller than a threshold, STOP.
    
    - Evaluate your information source to get the output:

        $$
            y_{t+1} = f_{\text{true}}(\mathbf{x}_{t+1}).
        $$

    - Add the new observation to your dataset:

        $$
            \mathcal{D}_{t+1} = \mathcal{D}_t \cup \{(\mathbf{x}_{t+1}, y_{t+1})\}.
        $$
    
    - Condition your model on $\mathcal{D}_{t+1}$.

## Information-theoretic acquisition functions

This construction follows the information-based criterion of {cite:t}`mackay1992information`.
The idea is to pick $\alpha_t(\mathbf{x})$ to be the expected information gain about the model given the data.
We think as follows:

+ Our state of knowledge about the model is $p(f|\mathcal{D}_t)$.

+ Suppose we make a hypothetical observation at $\mathbf{x}$ and get the output $y$. Our state of knowledge about the model would change to:

    $$
        p(f|\mathcal{D}_t, \mathbf{x}, y) = \frac{p(y|\mathbf{x}, f, \mathcal{D}_t)p(f|\mathcal{D}_t)}{p(y|\mathbf{x}, \mathcal{D}_t)}.
    $$

+ The information gain is then the Kullback--Leibler divergence between the posterior before and after the observation:

    $$
    \begin{aligned}
        &\operatorname{KL}\left[p(f|\mathcal{D}_t,\mathbf{x},y)\parallel p(f|\mathcal{D}_t)\right] \\
        &\quad= \mathbb{E}\left[\log\left(\frac{p(f|\mathcal{D}_t,\mathbf{x},y)}{p(f|\mathcal{D}_t)}\right)\middle| \mathbf{x},y\right] \\
        &\quad= \int p(f|\mathcal{D}_t,\mathbf{x},y)\log\left(\frac{p(f|\mathcal{D}_t,\mathbf{x},y)}{p(f|\mathcal{D}_t)}\right)\;Df.
    \end{aligned}
    $$

+ Since we do not know $y$, we take the expectation of the information gain over all possible values of $y$:

    $$
        \alpha_t(\mathbf{x}) = \mathbb{E}\left[\operatorname{KL}\left[p(f|\mathcal{D}_t,\mathbf{x},y)\parallel p(f|\mathcal{D}_t)\right]\middle|\mathbf{x}\right].
    $$

In general, this acquisition function is not analytically tractable.
However, MacKay proves a useful formula that connects the expected information gain to the expected difference in the entropy of the model distribution before and after the observation.
The differential entropy is:

$$
S_{t} = -\int p(f|\mathcal{D}_t)\log \frac{p(f|\mathcal{D}_t)}{m(f)}\;Df = -\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t)}{m(f)}\right],
$$

where $m(f)$ is a reference measure.

The differential entropy after the hypothetical observation is:

$$
\begin{aligned}
S_{t+1}(\mathbf{x},y)
&= -\int p(f|\mathcal{D}_t, \mathbf{x}, y)
\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\;Df \\
&= -\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\middle| \mathbf{x}, y\right].
\end{aligned}
$$

Using the properties of the conditional expectation, we have:

$$
\begin{aligned}
\mathbb{E}\left[S_{t+1}(\mathbf{x},y)\middle| \mathbf{x}\right]
&= -\mathbb{E}\left[\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\middle| \mathbf{x}, y\right]\middle| \mathbf{x}\right] \\
&= -\mathbb{E}\left[\log \frac{p(f|\mathcal{D}_t, \mathbf{x}, y)}{m(f)}\middle| \mathbf{x}\right].
\end{aligned}
$$

From this formula, we can see that the expected information gain is:

$$
\alpha_t(\mathbf{x}) = S_{t} - \mathbb{E}\left[S_{t+1}(\mathbf{x},y)\middle| \mathbf{x}\right].
$$

Notice that the reference measure cancels out.
For a Gaussian predictive model with independent Gaussian observation noise of variance $\sigma_n^2$, the expected information gain is:

$$
\alpha_t(\mathbf{x}) = \frac{1}{2}\log\left(1 + \frac{\sigma_t^2(\mathbf{x})}{\sigma_n^2}\right).
$$

With constant noise, this criterion has the same maximizer as the predictive variance and therefore reduces to *uncertainty sampling*.
If the measurement noise varies with the input, the expected information gain becomes:

$$
\alpha_t(\mathbf{x}) = \frac{1}{2}\log\left(1 + \frac{\sigma_t^2(\mathbf{x})}{\sigma_n^2(\mathbf{x})}\right).
$$

The acquisition then balances epistemic uncertainty against local measurement noise.

Uncertainty sampling is known to put more emphasis on the boundaries of the input space.
This is because the model is more uncertain in these regions.
This is not always desirable.
MacKay in the paper cited above develops some other information acquisition functions that attempt to maximize the expected information gain about the model in a specific region of interest.

## The value of information

Another way to construct an information acquisition function is to think about the value of information.

+ Suppose we make a hypothetical observation at $\mathbf{x}$ and get the output $y$. Suppose that you have a utility function $u_t(\mathbf{x}, y)$ that quantifies how much value you get from making the observation.
For example, it could be:

    $$
    u_t(\mathbf{x}, y) = v_t(y) - c_t(\mathbf{x}),
    $$

    where $v_t(y)$ is the value of the output and $c_t(\mathbf{x})$ is the cost of making the observation.

+ The acquisition function is then the expected value of the utility function:

    $$
    \alpha_t(\mathbf{x}) = \mathbb{E}[u_t(\mathbf{x}, y)|\mathbf{x}] = \int u_t(\mathbf{x}, y)p(y|\mathbf{x}, \mathcal{D}_t)\,dy.
    $$

For maximization, expected improvement takes the utility to be the positive increase above the best value observed so far {cite:p}`jones1998efficient`.

The knowledge gradient uses the same decision-theoretic idea but values what can be done *after* the new observation. It scores a candidate by the expected increase in the largest posterior mean attainable after updating the model with that observation {cite:p}`frazier2009knowledge`.

## Multi-fidelity active learning

In the context of multi-fidelity modeling, our decision is not only where to make the next observation, but also at which fidelity level.
So, we have to pick the fidelity level $s$, in $\{\ell,h\}$ for the two-level model of the multi-fidelity section, and the input $\mathbf{x}$.
The information acquisition function we construct must be of the form $\alpha_t(s, \mathbf{x})$.
The algorithm changes to:

+ Start with a dataset:

    $$
    \mathcal{D}_{n_0} = \{(\mathbf{x}_i, y_i, s_i)\}_{i=1}^{n_0}.
    $$

+ Condition your model on $\mathcal{D}_{n_0}$.

+ For $t = n_{0}, n_{0}+1, \dots$:

    - Find the input $\mathbf{x}$ and fidelity level $s$ that maximize an *information acquisition function* $\alpha_t(s, \mathbf{x})$:

        $$
            (s_{t+1}, \mathbf{x}_{t+1}) = \arg\max_{s\in\{\ell,h\}, \mathbf{x}\in \mathcal{X}} \alpha_t(s, \mathbf{x}).
        $$

    - If $\alpha_t(s_{t+1}, \mathbf{x}_{t+1})$ is smaller than a threshold, STOP.
    
    - Evaluate your information source to get the output:

        $$
            y_{t+1} = f_{s_{t+1}}(\mathbf{x}_{t+1}).
        $$

    - Add the new observation to your dataset:

        $$
            \mathcal{D}_{t+1} = \mathcal{D}_t \cup \{(\mathbf{x}_{t+1}, y_{t+1}, s_{t+1})\}.
        $$
    
    - Condition your model on $\mathcal{D}_{t+1}$.


When our target is the high-fidelity response, a query is valuable only to the extent that it reduces uncertainty about that response. For a joint Gaussian posterior, let $k_{t,hs}(\mathbf{z},\mathbf{x})$ denote the posterior covariance between the high-fidelity value at a target input $\mathbf{z}$ and the response queried at fidelity $s$ and input $\mathbf{x}$. Let $\sigma_{t,s}^2(\mathbf{x})$ be the latent posterior variance at the queried point and $\sigma_s^2$ its independent observation-noise variance. The reduction in the high-fidelity variance at $\mathbf{z}$ after this observation is

$$
\Delta_t(\mathbf{z};s,\mathbf{x})
=\frac{k_{t,hs}(\mathbf{z},\mathbf{x})^2}
{\sigma_{t,s}^2(\mathbf{x})+\sigma_s^2}.
$$

We can average this reduction over target inputs drawn from a chosen distribution $\pi$ and divide by the positive query cost $c_s(\mathbf{x})$:

$$
\alpha_t(s,\mathbf{x})
=\frac{\mathbb{E}_{\mathbf{z}\sim\pi}[\Delta_t(\mathbf{z};s,\mathbf{x})]}
{c_s(\mathbf{x})}.
$$

This is a cost-normalized variance-reduction criterion based on Gaussian conditioning {cite:p}`rasmussen2006gaussian`. An uncorrelated low-fidelity source has zero cross-covariance and therefore zero value for this target. A noiseless simulator is allowed by setting $\sigma_s^2=0$, provided the queried latent variance is positive; an already known value has zero acquisition. The Gaussian covariance update assumes fixed hyperparameters.

## Active learning for multi-objective optimization

In multi-objective optimization, the Pareto front contains the attainable outcomes for which no objective can be improved without worsening at least one other objective.
A common acquisition function is expected hypervolume improvement, the expected increase in the objective-space volume dominated by the Pareto front {cite:p}`emmerich2011hypervolume,pandita2018stochastic`.
The related S-metric selection evolutionary multiobjective optimization algorithm (SMS-EMOA) uses dominated hypervolume directly for selection {cite:p}`beume2007sms`.

## Data selection and symmetry-aware models

The companion notebook summarized in the next subsection instantiates the fit--score--acquire--update loop with uncertainty sampling and shows how the selected inputs change as data accumulate. Active learning reduces cost by choosing model evaluations carefully. Physical symmetries provide a complementary source of efficiency by restricting the relationships that a surrogate may learn. The symmetry-aware models developed in the next section encode these restrictions through group actions, invariance, and equivariance.
