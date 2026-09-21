# Advanced Scientific Machine Learning

Scientific models encode what we know about a physical system, while data reveal where that knowledge is incomplete. Scientific machine learning combines these two sources of information. It uses differentiable scientific software, probabilistic modeling, and modern learning algorithms to propagate uncertainty, solve inverse problems, and learn maps between high-dimensional scientific objects.

The book is intended for graduate students and researchers in engineering and the physical sciences. Its emphasis is computational and mathematical: each method is introduced through the scientific problem that requires it, developed far enough to expose its assumptions, and then implemented in an executable example. The notebooks form the online computational companion to the printed text. They preserve the code, numerical experiments, diagnostics, and exercises that support the exposition.

## Prerequisites

The presentation assumes working knowledge of linear algebra, multivariable calculus, differential equations, probability, numerical methods, and standard machine learning. Familiarity with Python and basic scientific computing is also expected. A systematic introduction to this background is provided in *Introduction to Scientific Machine Learning for Engineering Students* {cite:p}`bilionis2026scientificml`. The following Purdue courses, or comparable preparation, provide the relevant background:

+ [ME 581 (Numerical Methods in Mechanical Engineering)](https://engineering.purdue.edu/online/courses/numerical-methods-mechanical-engineering), and
+ [ME 539 (Introduction to Scientific Machine Learning)](https://predictivesciencelab.github.io/data-analytics-se/index.html).

The book begins with the functional programming, differentiation, and optimization tools needed to express scientific learning algorithms. It then develops forward uncertainty propagation, high-dimensional inputs and operator learning, deterministic inverse problems, physics-informed neural networks, and inverse problems for stochastic dynamical systems. Across these topics, the same principle recurs: the structure of the scientific model should determine how data are represented, how uncertainty is quantified, and how learning is carried out.

```{tableofcontents}
```
