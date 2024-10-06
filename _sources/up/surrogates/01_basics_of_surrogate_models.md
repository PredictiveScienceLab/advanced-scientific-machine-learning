# Basic Elements of Surrogate Modeling

## The idea of a surrogate model

Let $f:\Omega \to \mathbb{R}$ be a scientific model that is expensive to evaluate. 
We use $x$ in $\Omega$ to denote the input to the model and $y = f(x)$ to denote the output of the model.
Here $x$ could be random inputs, which we denoted by $\xi$ earlier, but it could also be design variables that we can control.

The idea of a surrogate model is to approximate $f$ with a simpler function $\hat{f}:\Omega \to \mathbb{R}$ that is cheaper to evaluate. The surrogate model is constructed using a set of training points $\{x_i, y_i = f(x_i)\}_{i=1}^n$.
Once we have constructed the surrogate model, we can use it to make predictions at new points in $\Omega$ without having to evaluate the expensive model $f$.
We could also use the surrogate model to perform optimization, sensitivity analysis, uncertainty propagation with Monte Carlo, and (later in this course) solve inverse problems.

## Surrogate modeling workflow

The workflow for constructing a surrogate model is as follows:

1. **Collect training data**: Evaluate the expensive model $f$ at a set of training points $\{x_i\}_{i=1}^n$ to obtain the corresponding outputs $\{y_i = f(x_i)\}_{i=1}^n$. Typically, we want to use a space-filling design to ensure that the training points are well-distributed in the input space.
The most common choices are Latin hypercube sampling and Sobol sequences.

2. **Collect test data**: Evaluate the expensive model $f$ at a set of test points $\{x_i\}_{i=1}^m$ to evaluate the accuracy of the surrogate model. The test points should be different from the training points. Also use a space-filling design for the test points.

3. **Select a surrogate model**: Choose a surrogate model that can approximate the expensive model $f$. The choice of surrogate model depends on the characteristics of the expensive model and the training data. Common choices include polynomial regression, Gaussian process regression, and neural networks.

4. **Train the surrogate model**: Use the training data to train the surrogate model. This involves fitting the parameters of the surrogate model to the training data.

5. **Validate the surrogate model**: Use the test data to evaluate the accuracy of the surrogate model. This involves comparing the predictions of the surrogate model to the true values of the expensive model at the test points. If the accuracy is not satisfactory, collect more training data and and go to step 4.
Otherwise go to step 6.

6. **Use the surrogate model**: Once the surrogate model has been validated, use it to for prediction, optimization, sensitivity analysis, uncertainty propagation, and other tasks.


## Surrogate models

### Generalized linear models
Generalized linear models are of the form:

$$
    \hat{f}(x) = \sum_{i=1}^n w_i \phi_i(x),
$$

where $w_i$ are weights to be determined and $\phi_i(x)$ are basis functions.
If $x$ just includes a low number of random variables, we could use polynomial regression.
Another popular choice is radial basis functions.

Typically, we train the models by either minimizing the mean squared error, maximizing the likelihood of the data, or characterizing the posterior distribution of the weights by sampling or variational inference.
In this course, we prefer the last three options.

Some examples using polynomial chaos:

+ [Liu et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0951832020305093)

If you need to freshen up your knowledge, recall [Lecture 13](https://predictivesciencelab.github.io/data-analytics-se/lecture13/intro.html), [Lecture 14](https://predictivesciencelab.github.io/data-analytics-se/lecture14/intro.html), and [Lecture 15](https://predictivesciencelab.github.io/data-analytics-se/lecture15/intro.html) of ME 539.

### Gaussian process regression
In Gaussian process regression, one starts with a prior

$$
    f\sim \GP(m, k),
$$

where $m$ is the mean function and $k$ is the covariance function.
The mean function is typically set to zero.
But it could also be a constant, a linear function, or a polynomial.
In the latter cases, it will attempt to capture the trend in the data.
The covariance function models our beliefs about the smoothness of the function.
The most common choice is the squared exponential kernel, which assumes that the function is infinitely differentiable.
Both the mean and covariance functions have hyperparameters that need to be optimized.
We typically train the model by maximizing the marginal likelihood of the data.

Here are some examples of papers that use Gaussian process regression:

+ [Sree et al. 2023](https://www.sciencedirect.com/science/article/abs/pii/S1751616123000486)
+ [Sahu et al. 2020](https://ieeexplore.ieee.org/abstract/document/9103068)

The problem with GP regression is that it scales cubically with the number of training points.
This makes it impractical for large datasets (more than 5,000 points).
But there are some ways around this:

+ **Sparse GP regression**: This is a method that approximates the GP by using a small number of inducing points. See [Hensman et al. 2015](https://proceedings.mlr.press/v38/hensman15.pdf).

+ **Inputs on a Grid**: If the inputs are on a regular grid, and you have a separable kernel, you can use the Kronecker product to speed up computations. See [Bilionis et al. 2013](https://www.sciencedirect.com/science/article/abs/pii/S0021999113000417).

+ **GPyTorch list**: Andrew Gordon Wilson has a list of resources on scalable GP regression [here](https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/index.html#exact-gps-with-gpu-acceleration).

If you need to freshen up your knowledge of the basics, recall [Lecture 21](https://predictivesciencelab.github.io/data-analytics-se/lecture21/intro.html) and [Lecture 22](https://predictivesciencelab.github.io/data-analytics-se/lecture22/intro.html) of ME 539.

### Neural networks
Neural networks are also commonly used as surrogate models.
The best neural network for the task depends on the characteristics of the data.

Here are some examples of papers that use neural networks:

+ [Zhong et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S0378517322001429)
+ [Casey et al. 2020](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00259)

If you need to freshen up your knowledge, recall [Lecture 24](https://predictivesciencelab.github.io/data-analytics-se/lecture24/intro.html) and [Lecture 25](https://predictivesciencelab.github.io/data-analytics-se/lecture25/intro.html) of ME 539.

## Surrogate diagnostics

Once we have trained the surrogate model, we need to evaluate its accuracy.
We can use any of the diagnostics we discussed [here](https://predictivesciencelab.github.io/data-analytics-se/lecture15/hands-on-15.3.html).
In reality, we are primarily interested in reducing the mean squared error.