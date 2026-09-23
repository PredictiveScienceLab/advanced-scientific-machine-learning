# Differentiable Programming

Training, sensitivity analysis, and inverse problems require derivatives of computations that may contain many intermediate operations. Hand derivations become fragile at that scale, while finite differences can be inaccurate and expensive.

The treatment assumes multivariable calculus and a functional view of JAX programs, including pure functions and structured inputs. The necessary computational-graph concepts are developed from simple scalar examples.

We compare numerical and symbolic differentiation before constructing automatic differentiation from the chain rule on computational graphs. Forward- and reverse-mode products then explain how derivative cost depends on input and output dimension, and JAX examples turn those constructions into executable gradient computations. Differentiable programming thereby supplies accurate derivatives of scientific machine learning programs without deriving each composite model by hand.
