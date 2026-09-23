# Optimization for Scientific Machine Learning

Automatic differentiation supplies derivatives of a loss, but model training still requires an algorithm that uses those derivatives effectively. Scientific machine learning objectives are often high-dimensional, nonconvex, expensive, and evaluated from data in batches, which makes step selection and parameter initialization consequential.

The treatment assumes familiarity with JAX, pytrees, and automatic differentiation, together with basic multivariable calculus and probability. Optimization variables may therefore be vectors or structured model parameters.

We begin with objective geometry and gradient descent, then add momentum, stochastic gradients, and adaptive learning rates. Optax optimizers, second-order methods, neural-network initialization, and GPU training are summarized in print and developed in the companion notebooks. These components provide a practical basis for selecting, implementing, and diagnosing training algorithms used throughout scientific machine learning.
