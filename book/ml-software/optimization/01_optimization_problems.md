# Basics of Optimization Problems

In scientific machine learning, we typically have to find the minimum of a $n$-dimensional function. What we are optimizing over is usually the parameters of the model. The function we are optimizing could be:
+ the loss function of a neural network,
+ the negative log-likelihood of a probabilistic model,
+ the negative log-posterior of a Bayesian model,
+ the negative evidence lower bound of a variational formulation of a probabilistic model.

So, we need to learn how to solve such optimization problems.
There are several complications that are particular to optimization problems in scientific machine learning:

+ the number of parameters is very large (millions or billions in neural networks),
+ the function we are optimizing is very complex (non-convex, non-smooth, non-differentiable),
+ the function we are optimizing is very expensive to evaluate (it requires a lot of computation and memory).

We need taylor-made optimization algorithms that can deal with these complications.

## Reading

Instead of me reinventing the wheel, I will just point you to the best introduction to optimization in the context of machine learning that I can think of:

+ Chapter 8 of [Deep Learning](http://www.deeplearningbook.org/contents/optimization.html) by Goodfellow, Bengio, and Courville.

Please read it very carefully. It is part of the required reading.
Here are some of the key points.

### What is a stationary point?
A stationary point is a point where the gradient is zero. It could be a local minimum, a local maximum, or a saddle point.

### What is a local minimum?
It is a stationary point where the function has a lower value than at any other point in a small neighborhood around it. The Hessians of local minima are positive semi-definite. That is, all their eigenvalues are non-negative.

### What is a saddle point? 
A saddle point is a stationary point where the function has a lower value than at any other point in some directions and a higher value than at any other point in other directions. The Hessians of saddle points have both positive and negative eigenvalues.

### Good and bad local minima
There are good and bad local minima. Bad local minima are the ones that have a high value of the objective function (think loss function). Good local minima are the ones that have a low value of the objective function. Finding any good local minimum is usually good enough.

### Local minima are not a problem
Neural networks typically have a lot of local minima and saddle points. Local minima are abundant in neural networks. Take any neural network, relabel the neurons, and you will get a different neural network with the same loss function. This means that there are many local minima that are equivalent. Most machine learning practitioners believe that bad local minima are not a problem in practice, especially for deep neural networks.

### Saddle points are exponentially more common than local minima
Pick a random loss function. Think about the Hessian of a stationary point. Suppose the eigenvalues are picked randomly. To get a local minimum (or maximum) you have to pick all the eigenvalues to be positive (or negative). To get a saddle point, you have to pick some eigenvalues to be positive and some to be negative. The probability of getting a local minimum (or maximum) is exponentially smaller than the probability of getting a saddle point as the dimensionality of the problem increases.

### Stochastic gradient descent escapes saddle points quickly
Empirical evidence suggests stochastic gradient descent escapes saddle points quickly.

### Initialization matters

Initialization of neural network weights should break symmetry between neurons, avoid saturation of activation functions, and preserve variance of activations and gradients.

### Some remaining problems

#### Plateaus
Flat regions or plateaus are the big problem for optimization. We deal with them by using momentum and adaptive learning rates.

#### Cliffs
Cliff regions are also a problem for optimization. They are more common in recurrent neural networks. We deal with them with gradient clipping.

#### Vanishing or exploding gradients
Vanishing or exploding gradients are also a problem for optimization. We deal with them by using better activation functions and better initialization methods.