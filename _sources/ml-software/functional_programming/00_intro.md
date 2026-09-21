# Functional Programming

JAX compiles, differentiates, and vectorizes Python functions by treating them as mathematical maps from inputs to outputs. Functions that mutate hidden state obscure that map, so scientific machine learning code benefits from pure functions and explicit data flow.

The treatment assumes familiarity with Python functions, arrays, loops, and conditionals. The functional-programming ideas needed for JAX are developed directly in Python.

We first distinguish pure functions from side effects and practice higher-order operations such as mapping, reduction, partial application, and composition. We then apply JAX just-in-time compilation and vectorization, and close by managing pseudo-random numbers through explicit keys. These patterns produce numerical code that remains readable while supporting reliable compilation and parallel execution.
