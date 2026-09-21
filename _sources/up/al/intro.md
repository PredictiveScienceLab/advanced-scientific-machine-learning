# Active Learning

When simulations or experiments are expensive, a limited evaluation budget must be allocated across candidate inputs. In a multi-fidelity problem, each candidate also specifies a fidelity level. Active learning makes data collection sequential: after each model update, an acquisition function scores candidate evaluations by their expected information or utility and selects the highest-scoring candidate for the next evaluation.

The treatment assumes Bayesian regression, posterior predictive distributions, Gaussian process surrogates, and numerical optimization. Coupled low- and high-fidelity models supply the predictive distributions needed when each candidate specifies both an input and a fidelity level.

We first formulate the sequential fit-score-acquire-update loop and develop acquisition functions based on expected information gain and value of information. Multi-fidelity selection and optimization with several competing objectives extend the same principle. One- and two-dimensional examples then illustrate uncertainty sampling, which selects the next input where the surrogate's posterior variance is largest. Active learning directs a limited evaluation budget toward the evaluations expected to be most informative or useful.
