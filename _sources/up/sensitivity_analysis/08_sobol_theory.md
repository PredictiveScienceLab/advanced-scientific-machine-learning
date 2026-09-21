# Variance-Based Global Sensitivity Analysis

Local derivatives describe the response near one nominal input. A global sensitivity analysis instead asks how the uncertainty assigned to each input contributes to output variance over the full input distribution. Sobol indices answer this question by decomposing the model response into mutually orthogonal main effects and interactions {cite:p}`sobol2001global`.

Let $d$ be a positive integer, let $X_1,\ldots,X_d$ be mutually independent
random inputs, and set

$$
Y=f(\mathbf{X}), \qquad \mathbf{X}=(X_1,\ldots,X_d),
$$

where the scalar output $Y$ satisfies $0<\operatorname{Var}(Y)<\infty$.
All expectations and variances below are taken with respect to this specified
input distribution. Independence is essential for the classical decomposition:
with dependent inputs, its components are generally not orthogonal, so the
classical indices no longer give the same unique, nonnegative allocation of
variance. Vector- or time-dependent outputs can be analyzed one scalar
component and one time point at a time.

## Conditional expectations as input effects

For $i\in\{1,\ldots,d\}$, the conditional expectation

$$
m_i(X_i)=\mathbb{E}[Y\mid X_i]
$$

is the best mean-square prediction of $Y$ among square-integrable functions of
$X_i$. If $m_i(X_i)$ varies substantially, then knowing $X_i$ explains a
substantial part of the output variation. The main-effect variance of $X_i$ is
therefore

$$
V_i=\operatorname{Var}\!\left(\mathbb{E}[Y\mid X_i]\right).
$$

For distinct indices $i$ and $j$, conditioning on both inputs includes their
joint effect. The part that cannot be attributed to either input separately is

$$
V_{ij}
=\operatorname{Var}\!\left(\mathbb{E}[Y\mid X_i,X_j]\right)-V_i-V_j.
$$

This subtraction is the first instance of the functional analysis-of-variance decomposition.

## Functional ANOVA decomposition

For a subset $u\subseteq\{1,\ldots,d\}$, let $\mathbf{X}_u$ collect the inputs indexed by $u$. Define the constant term by

$$
f_{\varnothing}=\mathbb{E}[Y],
$$

and define every nonconstant component recursively by

$$
f_u(\mathbf{X}_u)
=\mathbb{E}[Y\mid \mathbf{X}_u]
-\sum_{w\subsetneq u}f_w(\mathbf{X}_w).
$$

Under the assumptions above, this recursion gives a unique functional ANOVA
decomposition up to changes on events of probability zero
{cite:p}`sobol2001global`:

$$
f(\mathbf{X})
=f_{\varnothing}
+\sum_{i=1}^d f_i(X_i)
+\sum_{1\le i<j\le d}f_{ij}(X_i,X_j)
+\cdots
+f_{\{1,\ldots,d\}}(\mathbf{X}).
$$

Each nonconstant component has zero mean. Independence of the inputs also makes distinct components orthogonal:

$$
\mathbb{E}\!\left[f_u(\mathbf{X}_u)f_v(\mathbf{X}_v)\right]=0,
\qquad u\ne v.
$$

Consequently, if

$$
V=\operatorname{Var}(Y), \qquad
V_u=\operatorname{Var}\!\left(f_u(\mathbf{X}_u)\right),
$$

then the total variance separates into nonnegative contributions:

$$
V=\sum_{\varnothing\ne u\subseteq\{1,\ldots,d\}}V_u.
$$

Here $V_i=V_{\{i\}}$ is the main-effect contribution of $X_i$, and
$V_{ij}=V_{\{i,j\}}$ is the nonadditive variance assigned to their interaction
under the specified input distribution. Higher-order terms have the same
interpretation.

## Sobol indices

The Sobol index associated with a nonempty subset $u$ is the fraction of output variance assigned to that ANOVA component:

$$
S_u=\frac{V_u}{V}.
$$

Thus $0\le S_u\le 1$. All subset indices form a partition of the variance,

$$
\sum_{\varnothing\ne u\subseteq\{1,\ldots,d\}}S_u=1.
$$

The first-order index

$$
S_i=\frac{\operatorname{Var}(\mathbb{E}[Y\mid X_i])}{\operatorname{Var}(Y)}
$$

measures the contribution of $X_i$ by itself. The first-order indices satisfy

$$
\sum_{i=1}^d S_i\le 1.
$$

For the exact indices,

$$
1-\sum_{i=1}^d S_i
=\sum_{\substack{\varnothing\ne u\subseteq\{1,\ldots,d\}\\ |u|\ge 2}}S_u,
$$

where $|u|$ is the number of inputs indexed by $u$. Equality in the preceding
inequality holds if and only if every interaction variance is zero. Indices
estimated from a finite sample can violate the individual bounds or the sum
identities in either direction because of sampling error.

The total-effect index for $X_i$ collects every component that contains $i$
{cite:p}`homma1996importance`:

$$
S_{T_i}=\sum_{u:\,i\in u}S_u.
$$

Let $\mathbf{X}_{-i}$ contain all inputs except $X_i$. The total-effect index
has two equivalent conditional-variance forms:

$$
S_{T_i}
=1-\frac{\operatorname{Var}(\mathbb{E}[Y\mid\mathbf{X}_{-i}])}{V}
=\frac{\mathbb{E}[\operatorname{Var}(Y\mid\mathbf{X}_{-i})]}{V},
$$

and it satisfies

$$
0\le S_i\le S_{T_i}\le 1.
$$

The difference

$$
S_{T_i}-S_i
$$

is the fraction of variance assigned to interactions between $X_i$ and at least
one other input. Because an interaction indexed by $u$ appears in $S_{T_i}$ for
every $i\in u$,

$$
\sum_{i=1}^d S_{T_i}
=\sum_{\varnothing\ne u\subseteq\{1,\ldots,d\}}|u|S_u
\ge 1.
$$

Equality holds if and only if every interaction variance is zero. Thus,
total-effect indices generally do not sum to one; an interaction of order
$|u|$ is counted $|u|$ times.

## A two-input example

Let $X_1$ and $X_2$ be independent and uniformly distributed on $[-1,1]$, and
consider

$$
Y=X_1+X_2+X_1X_2.
$$

Because both inputs have zero mean, the three nonconstant ANOVA components are
$f_1(X_1)=X_1$, $f_2(X_2)=X_2$, and
$f_{12}(X_1,X_2)=X_1X_2$. Their variance contributions are

$$
V_1=V_2=\frac{1}{3}, \qquad V_{12}=\frac{1}{9}, \qquad
V=\frac{7}{9}.
$$

Therefore,

$$
S_1=S_2=\frac{3}{7}, \qquad S_{12}=\frac{1}{7}, \qquad
S_{T_1}=S_{T_2}=\frac{4}{7}.
$$

The first-order indices sum to $6/7$ because $1/7$ of the variance is an
interaction. The total-effect indices sum to $8/7$ because that same
interaction is counted once for each input.

## From definitions to estimates

After each coordinate of a unit-cube design is transformed to its intended
marginal distribution, the resulting design targets the product input
distribution assumed above. A standard pick--freeze construction begins with two
$N\times d$ base design matrices, $A$ and $B$, where $N$ is the number of base
points. For each input $i$, the hybrid matrix $A_B^{(i)}$ takes column $i$ from
$B$ and all other columns from $A$. Comparisons among the model evaluations at
$A$, $B$, and the hybrid designs yield estimates of $S_i$ and $S_{T_i}$
{cite:p}`saltelli2010variance`. The base designs may be pseudorandom or
low-discrepancy; this choice changes the numerical integration, not the
population definitions of the indices.

The next notebook applies this construction to scalar outputs of the Duffing
oscillator and interprets the estimated first-order and total-effect indices.
