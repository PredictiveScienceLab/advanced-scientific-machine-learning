# Limitations of Polynomial Chaos

Polynomial chaos is a powerful tool for uncertainty quantification, but it has some limitations. Here are some:

+ **Curse of dimensionality**: The number of terms in the polynomial chaos expansion grows exponentially with the number of random variables. This makes it computationally expensive to use polynomial chaos for high-dimensional problems.

+ **Convergence issues**: Polynomial chaos may not converge for some functions. This is especially true for functions with discontinuities or singularities. It also does not work well for chaotic systems or functions with high-frequency oscillations.

We will talk later about methods that address the curse of dimensionality.
We are not going to talk about discontinuities, but here are some references that discuss this issue:

+ [Bilionis et al. 2013](https://www.sciencedirect.com/science/article/abs/pii/S0021999112002513)

Polynomial chaos and chaotic systems just don't mix well.