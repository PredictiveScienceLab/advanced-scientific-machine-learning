# Limitations of Polynomial Chaos

Polynomial chaos is most effective when a model response depends smoothly on a
moderate number of uncertain variables. The method becomes less attractive
when any part of that regime fails.

The first limitation is dimension. A common isotropic total-degree truncation
of order $p$ treats all input directions equally. For $d$ uncertain variables,
it contains

$$
\binom{d+p}{p}
$$

basis functions {cite}`xiu2002wiener`. Increasing either the dimension or the
polynomial degree can therefore make coefficient estimation and model
evaluation expensive.

Several refinements reduce the candidate set. Least-angle regression can
construct a sparse expansion from a hyperbolically truncated candidate set
{cite}`blatman2011adaptive`. Anisotropic polynomial index sets use
direction-specific order parameters {cite}`hampton2018basepc`, while a greedy
admissible-neighbor algorithm grows a downward-closed index set adaptively
{cite}`loukrezis2020robust`. These methods can delay combinatorial growth, but
they still require enough data to identify the relevant directions and
interactions.

The second limitation is regularity. Spectral convergence relies on a smooth
dependence of the quantity of interest on the uncertain inputs. Discontinuities
in random space lead to slow convergence of a global polynomial expansion.
Local or multi-element expansions can recover accuracy when the nonsmooth
regions can be isolated by partitioning the random-input domain and fitting
local expansions {cite}`wan2005adaptive`. A single global polynomial basis is
then a poor representation.

Long-time dynamics create a related difficulty. Even a smooth dynamical system
can develop increasingly oscillatory dependence on uncertain inputs over time.
For periodic solutions with random frequencies, amplified phase differences
can cause a fixed global polynomial expansion to lose accuracy rapidly during
long-time integration {cite}`wan2005adaptive`. Statistical quantities or
shorter prediction horizons may remain accessible, but they require validation
tailored to those targets.

These limitations define the regime in which polynomial chaos can be expected
to work efficiently. When dimension, cost, or irregularity dominates, the
following sections replace a fixed global expansion with learned surrogates,
multiple information sources, adaptive data collection, and structural
constraints.
