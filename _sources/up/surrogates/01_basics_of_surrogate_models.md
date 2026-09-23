# Basic Elements of Surrogate Modeling

## The idea of a surrogate model

Let $f:\Omega \to \mathbb{R}$ be a scientific model that is expensive to evaluate. 
We use $x$ in $\Omega$ to denote the input to the model and $y = f(x)$ to denote the output of the model.
Here $x$ could be random inputs, which we denoted by $\xi$ earlier, but it could also be design variables that we can control.

The idea of a surrogate model is to approximate $f$ with a simpler function $\hat{f}:\Omega \to \mathbb{R}$ that is cheaper to evaluate. The surrogate model is constructed using a set of training points $\{x_i, y_i = f(x_i)\}_{i=1}^n$.
Once we have constructed the surrogate model, we can use it to make predictions at new points in $\Omega$ without having to evaluate the expensive model $f$.
We could also use the surrogate model to perform optimization, sensitivity analysis, uncertainty propagation with Monte Carlo, and solve inverse problems.

## Surrogate modeling workflow

The workflow for constructing a surrogate model is as follows:

1. **Collect training data**: Evaluate the expensive model $f$ at a set of training points $\{x_i\}_{i=1}^n$ to obtain the corresponding outputs $\{y_i = f(x_i)\}_{i=1}^n$. Typically, we want to use a space-filling design to ensure that the training points are well-distributed in the input space.
The most common choices are Latin hypercube sampling and Sobol sequences.

2. **Collect validation data**: Evaluate the expensive model $f$ at a set of validation points $\{x_i'\}_{i=1}^m$ to evaluate the accuracy of the surrogate model. The validation points should be different from the training points.

3. **Select a surrogate model**: Choose a surrogate model that can approximate the expensive model $f$. The choice of surrogate model depends on the characteristics of the expensive model and the training data. Common choices include polynomial regression, Gaussian process regression, and neural networks.

4. **Train the surrogate model**: Use the training data to train the surrogate model. This involves fitting the parameters of the surrogate model to the training data.

5. **Validate the surrogate model**: Use the validation data to evaluate the accuracy of the surrogate model. This involves comparing the predictions of the surrogate model to the true values of the expensive model at the validation points. Typically, we just use the mean squared error. If the accuracy is not satisfactory, collect more training data and go to step 4.
Otherwise go to step 6.

6. **Test the surrogate model**: After fixing the model choices, evaluate it once on a separate, previously unused test set. Look beyond the mean squared error at diagnostics relevant to the intended use. If these results lead us to revise the model, this set becomes validation data; reserve a fresh, untouched test set for the revised model's final assessment. If the diagnostics are satisfactory, go to step 7.

7. **Use the surrogate model**: Once the surrogate model has been validated, use it for prediction, optimization, sensitivity analysis, uncertainty propagation, and other tasks.

## Surrogate models

### Linear basis models
Linear basis models are of the form:

$$
    \hat{f}(x) = \sum_{j=1}^{p} w_j \phi_j(x),
$$

where $w_j$ are weights to be determined and $\phi_j(x)$ are basis functions.
If $x$ just includes a low number of random variables, we could use polynomial regression.
Another popular choice is radial basis functions.

Typically, we train the models by either minimizing the mean squared error, maximizing the likelihood of the data, or characterizing the posterior distribution of the weights by sampling or variational inference.
We use all three approaches in this book.

An example using polynomial chaos is {cite:t}`liu2020resampled`.

### Gaussian process regression
In Gaussian process regression, one starts with a prior

$$
    f\sim \operatorname{GP}(m, k),
$$

where $m$ is the mean function and $k$ is the covariance function.
The mean function is typically set to zero.
But it could also be a constant, a linear function, or a polynomial.
In the latter cases, it will attempt to capture the trend in the data.
The covariance function models our beliefs about the smoothness of the function.
The most common choice is the squared exponential kernel, which assumes that the function is infinitely differentiable.
Both the mean and covariance functions have hyperparameters that need to be optimized.
We typically train the model by maximizing the marginal likelihood of the data {cite:p}`rasmussen2006gaussian`.

Here are some examples of papers that use Gaussian process regression:

+ {cite:t}`sree2023autoinjectors`
+ {cite:t}`sahu2020magnetic`

Standard dense exact Gaussian process regression scales cubically with the number of training points, so it becomes impractical as the data set grows.
The practical limit depends on the hardware, implementation, and required turnaround time.
Several methods reduce this cost:

+ **Sparse GP regression**: This method approximates the GP using a small number of inducing variables {cite:p}`titsias2009variational,hensman2013gaussian`.

+ **Inputs on a grid**: If the inputs are on a regular grid and the kernel is separable, the Kronecker product can speed up computations {cite:p}`bilionis2013multioutput`. For matrices $A$ and $B$, the Kronecker product $A\otimes B$ is the block matrix obtained by replacing each entry $a_{ij}$ of $A$ with the block $a_{ij}B$.

+ **GPyTorch examples**: The GPyTorch repository collects [examples of scalable GP regression](https://github.com/cornellius-gp/gpytorch/tree/main/examples/02_Scalable_Exact_GPs).

### Neural networks
Neural networks are also commonly used as surrogate models.
The best neural network for the task depends on the characteristics of the data.

Here are some examples of papers that use neural networks:

+ {cite:t}`zhong2022autoinjectors`
+ {cite:t}`casey2020energetic`

## Surrogate diagnostics

Training error measures how closely a surrogate fits data already used to
construct it. It does not measure predictive accuracy. Validation begins with
an independent design that covers the region in which the surrogate will be
used. The design should reflect the intended input distribution or decision
domain; a uniformly space-filling validation set can hide poor accuracy in a
small region that carries most of the probability or decision value. Use the
validation results to revise the model or training design. After these choices
are fixed, assess the selected surrogate once on a separate test design.
{cite:t}`bastos2009diagnostics` develop validation diagnostics specifically for
Gaussian-process emulators.

For scalar outputs, let $(x_i,y_i)$, $i=1,\ldots,m$, be the independent
validation pairs and let $\widehat{f}(x_i)$ be the corresponding prediction.
The root mean squared error summarizes the typical absolute prediction error,

$$
\operatorname{RMSE}
=
\left[
\frac{1}{m}\sum_{i=1}^{m}
\left(y_i-\widehat{f}(x_i)\right)^2
\right]^{1/2}.
$$

A normalized RMSE makes comparisons across quantities or data sets easier, but
the normalization must be reported. Mean absolute error is less sensitive to a
small number of large residuals. Maximum absolute error and upper quantiles of
the absolute residuals are important when rare local failures matter. Relative
errors require care near zero outputs. For vector- or function-valued
predictions, the norm should match the scientific quantity being approximated,
and componentwise or spatial error plots should accompany a single aggregate
number.

Residual plots reveal failures that an average metric conceals. Plot residuals
against predicted value, each influential input, and location or time when the
output is structured. A systematic trend indicates bias. Changing residual
spread indicates nonuniform accuracy. Clusters of large residuals identify
regions in which the training design or model class is inadequate. Comparing
training and validation errors also helps distinguish underfitting from
overfitting.

A probabilistic surrogate requires additional checks. Predictive intervals
should attain their nominal coverage on independent data, and standardized
residuals should be compatible with the predictive distribution. Sharp
intervals are useful only when they are calibrated. Coverage should therefore
be examined across the input domain, not only after pooling all validation
points. Poor calibration can arise from an inappropriate covariance model,
inaccurate observation-noise assumptions, poorly determined plug-in
hyperparameters, ignored hyperparameter uncertainty, or extrapolation.

The final diagnostic must match the downstream task. A surrogate used for
uncertainty propagation should reproduce the output distribution and relevant
tail probabilities. A surrogate used for optimization must be accurate near
candidate optima and should not create false extrema. A surrogate used inside
an inverse problem must be accurate where the posterior places mass; small
global prediction error does not guarantee a small posterior error.

These diagnostics guide the model choices in the pages that follow. The next
examples build neural-network and Gaussian-process surrogates and then
develop a sparse Gaussian-process approximation for larger data sets. In each
case, the relevant question is whether the surrogate is accurate and, when
probabilistic, calibrated for the scientific task.
