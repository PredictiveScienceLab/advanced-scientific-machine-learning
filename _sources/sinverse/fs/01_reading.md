# Particle Filtering and Smoothing

Filtering and smoothing infer a time-dependent latent state in a partially
observed dynamical system. Filtering estimates the state $x_t$ from the data
available through time $t$. Smoothing revises an earlier state after later data
arrive. Prediction propagates the filtered state beyond the latest observation.
The three tasks use the same state-space model but condition on different
information.

This is one of the core inverse problems in stochastic scientific models. In deterministic inverse problems we often infer a fixed parameter from data. In filtering we infer a **time-dependent latent state** from sequential data. That difference changes both the statistical model and the computational strategy.

## Controlled Hidden Markov Models

Let $x_t$ be the latent state, $y_t$ the observation, and $u_t$ a known input
at time $t$. For $t=1,\ldots,T$, where $T$ is the final observation time, a
**state-space model**, also called a **hidden Markov model**, is specified by
the initial density {cite:p}`cappe2005hmm`

$$
x_0\sim p_0,
$$

and the transition and observation densities:

$$
x_t \mid x_{t-1},u_t \sim f_t(x_t \mid x_{t-1},u_t),
$$

$$
y_t \mid x_t,u_t \sim g_t(y_t \mid x_t,u_t).
$$

The known input $u_t$ acts during the transition from time $t-1$ to time $t$.
The transition density $f_t$ describes the stochastic dynamics of the hidden
state, and the observation density $g_t$ describes how the measurement is
generated. An uncontrolled model is the special case in which neither density
depends on $u_t$.

A known control is a conditioning variable, not a latent state. If the applied
input is uncertain, its uncertainty must instead be included in the state or
parameter model.

The densities $p_0$, $f_t$, and $g_t$ may depend on model parameters
$\theta$. Filtering and smoothing in this section treat $\theta$ as known, so
we suppress it in the notation. Parameter inference treats $\theta$ as unknown
and is developed separately.

We use the block notation $x_{a:b}=(x_a,\ldots,x_b)$ and similarly for
observations and controls.

The Markov assumption says that, after the previous state and current input are
known, earlier states and observations provide no additional information about
$x_t$:

$$
p(x_t\mid x_{0:t-1},y_{1:t-1},u_{1:t})
=f_t(x_t\mid x_{t-1},u_t).
$$

Likewise, the current observation depends on the history only through the
current state and input:

$$
p(y_t\mid x_{0:t},y_{1:t-1},u_{1:t})
=g_t(y_t\mid x_t,u_t).
$$

Together, these assumptions give the joint density

$$
p(x_{0:T},y_{1:T}\mid u_{1:T})
=
p_0(x_0)
\prod_{t=1}^T
f_t(x_t\mid x_{t-1},u_t)
g_t(y_t\mid x_t,u_t).
$$

This factorization makes recursive filtering and smoothing possible.

## Filtering Recursion

The filtering density at time $t-1$ is

$$
\pi_{t-1}(x_{t-1})
=p(x_{t-1} \mid y_{1:t-1},u_{1:t-1}).
$$

The recursion has a prediction step and an observation update. The prediction
density is

$$
\pi_{t\mid t-1}(x_t)
=
\int f_t(x_t \mid x_{t-1},u_t)
\pi_{t-1}(x_{t-1}) \, dx_{t-1}.
$$

This pushes the previous posterior through the stochastic dynamics.

The new observation contributes the one-step predictive likelihood

$$
L_t
=p(y_t\mid y_{1:t-1},u_{1:t})
=\int
g_t(y_t \mid x_t,u_t)
\pi_{t\mid t-1}(x_t)\,dx_t.
$$

Bayes' rule then gives the updated filtering density

$$
\pi_t(x_t)
=p(x_t \mid y_{1:t},u_{1:t})
=\frac{
g_t(y_t \mid x_t,u_t)\pi_{t\mid t-1}(x_t)
}{L_t}.
$$

This corrects the prediction using the new observation.

The product $\prod_{s=1}^t L_s$ is the marginal likelihood
$p(y_{1:t}\mid u_{1:t})$ used for parameter inference.

If the model is linear and Gaussian, the filtering recursion closes
analytically and gives the Kalman filter. For nonlinear or non-Gaussian models,
closed-form recursions are usually unavailable, and approximation is needed.

Scientific models are often nonlinear, and their process or observation noise
may be non-Gaussian. The filtering distribution can then be skewed, multimodal,
or heavy-tailed. Particle filters address this setting by applying importance
sampling recursively to the filtering distributions {cite:p}`doucet2001smc`.

## Empirical Measures

A particle method represents $\pi_t$ with $N$ particle locations
$x_t^{(1)},\ldots,x_t^{(N)}$ and nonnegative weights
$w_t^{(1)},\ldots,w_t^{(N)}$. Their weighted empirical measure is

$$
\pi_t^N(dx)
=
\sum_{i=1}^N w_t^{(i)}\,\delta_{x_t^{(i)}}(dx),
\qquad
\;w_t^{(i)}\geq 0,\qquad
\sum_{i=1}^N w_t^{(i)}=1.
$$

Here $\delta_{x_t^{(i)}}$ is a point mass at particle $x_t^{(i)}$. For a
quantity of interest $h$, integration against the empirical measure becomes a
weighted sum:

$$
\mathbb{E}_{\pi_t}[h(x_t)]
\approx
\sum_{i=1}^N w_t^{(i)}h(x_t^{(i)}).
$$

Particles therefore represent the possible states, while their weights
represent the relative posterior probability assigned to those states.

## Importance Sampling

Importance sampling supplies the basic update behind a particle filter. Let a
target density be known up to a normalizing constant as $\pi(x)\propto
\pi^*(x)$, and let $q(x)$ be a proposal density from which we can sample. The
proposal must cover the support of the target, meaning every region where the
target density is positive:

$$
\pi^*(x)>0 \quad\Longrightarrow\quad q(x)>0.
$$

For independent samples $x^{(i)}\sim q$, define the unnormalized importance
weights

$$
\widetilde w^{(i)}=\frac{\pi^*(x^{(i)})}{q(x^{(i)})}
$$

and the self-normalized weights

$$
w^{(i)}
=
\frac{\widetilde w^{(i)}}{\sum_{j=1}^N\widetilde w^{(j)}}.
$$

The self-normalized estimator

$$
\mathbb{E}_{\pi}[h(x)]
\approx
\sum_{i=1}^N w^{(i)}h(x^{(i)})
$$

does not require the unknown normalizing constant of $\pi^*$. Its accuracy
depends on the proposal placing particles where the target density is large.

## Sequential Importance Sampling

At the initial time, draw $x_0^{(i)}\sim q_0$ and set

$$
\widetilde w_0^{(i)}
=
\frac{p_0(x_0^{(i)})}{q_0(x_0^{(i)})}.
$$

The initial proposal must cover the support of $p_0$. Suppose the resulting
weighted particles
$\{x_{t-1}^{(i)},w_{t-1}^{(i)}\}_{i=1}^N$ approximate the filtering
distribution at time $t-1$. A common observation-informed Markov proposal uses
the previous state, the new observation, and the known input:

$$
x_t^{(i)}
\sim
q_t(x_t\mid x_{t-1}^{(i)},y_t,u_t).
$$

The corresponding unnormalized weight is

$$
\widetilde w_t^{(i)}
=
w_{t-1}^{(i)}
\frac{
g_t(y_t\mid x_t^{(i)},u_t)
f_t(x_t^{(i)}\mid x_{t-1}^{(i)},u_t)
}{
q_t(x_t^{(i)}\mid x_{t-1}^{(i)},y_t,u_t)
}.
$$

The proposal must be positive wherever the numerator is positive for any
particle with positive previous weight. Normalizing the new weights produces
the next empirical filtering distribution. If the particles have just been
resampled, their previous weights are all $1/N$ and this common factor cancels
during normalization.

## The Bootstrap Particle Filter

The **bootstrap filter** chooses the transition density itself as the proposal
{cite:p}`gordon1993bootstrap`:

$$
q_t(x_t\mid x_{t-1}^{(i)},y_t,u_t)
=
f_t(x_t\mid x_{t-1}^{(i)},u_t).
$$

Initialize the filter by drawing

$$
x_0^{(i)}\sim p_0,\qquad w_0^{(i)}=\frac{1}{N}.
$$

When resampling is performed before every transition, one step from
$t-1$ to $t$ is:

1. Draw ancestor indices according to the current weights and copy the selected
   states:

   $$
   A_{t-1}^{(i)}
   \sim
   \operatorname{Categorical}
   \left(w_{t-1}^{(1)},\ldots,w_{t-1}^{(N)}\right),
   \qquad
   \bar{x}_{t-1}^{(i)}=x_{t-1}^{(A_{t-1}^{(i)})}.
   $$

2. Propagate each selected ancestor through the transition:

   $$
   x_t^{(i)} \sim f_t(x_t \mid \bar{x}_{t-1}^{(i)},u_t).
   $$

3. Weight the propagated states using the observation likelihood:

   $$
   \widetilde{w}_t^{(i)}
   =
   g_t(y_t \mid x_t^{(i)},u_t).
   $$

4. Normalize the weights:

   $$
   w_t^{(i)}
   =
   \frac{\widetilde{w}_t^{(i)}}{
   \sum_{j=1}^N \widetilde{w}_t^{(j)}}.
   $$

The transition and proposal factors cancel, and the equal post-resampling
weights contribute only a common factor. With adaptive resampling, a step that
does not resample retains the previous weight:

$$
\widetilde{w}_t^{(i)}
=
w_{t-1}^{(i)}g_t(y_t\mid x_t^{(i)},u_t).
$$

Each particle represents a possible current state. Retaining its ancestor
indices also records a possible latent trajectory.

## Weight Degeneracy

The main failure mode of a particle filter is **weight degeneracy**. After several updates, one or two particles may carry almost all the probability mass while the rest have negligible weights.

Only a few particles then contribute meaningfully to weighted estimates. A
standard diagnostic is the estimated **effective sample size** (ESS)

$$
\widehat{N}_{\mathrm{eff}}
=
\frac{1}{\sum_{i=1}^N (w_t^{(i)})^2},
\qquad
1\leq \widehat{N}_{\mathrm{eff}}\leq N.
$$

Equal weights give $\widehat{N}_{\mathrm{eff}}=N$, while one nonzero weight
gives $\widehat{N}_{\mathrm{eff}}=1$. The quantity is a diagnostic rather than
an exact count of independent samples.

## Resampling

Resampling replaces the weighted particle cloud by an equally weighted sample
drawn according to the current weights. High-weight particles are copied,
low-weight particles are removed, and every copy receives weight $1/N$.

Resampling controls weight degeneracy but introduces **genealogical
degeneracy**: repeated copies cause many later particles to share the same
ancestors. This loss of ancestral diversity is especially important in
smoothing. Practical filters therefore often resample only when

$$
\widehat{N}_{\mathrm{eff}}<\tau N,
$$

where $0<\tau<1$ is a chosen threshold.

## Smoothing

Filtering uses observations only through the state time. Fixed-interval
smoothing uses every observation in a completed interval. For $0\leq t\leq T$,
the smoothing density is

$$
\pi_{t\mid T}(x_t)
=
p(x_t\mid y_{1:T},u_{1:T}).
$$

At the final time, smoothing and filtering agree:
$\pi_{T\mid T}=\pi_T$. Earlier smoothing densities can be computed backward.
For any $x'$ with $\pi_{t+1\mid t}(x')>0$, define the backward kernel

$$
B_t(x\mid x')
=
\frac{
\pi_t(x)f_{t+1}(x'\mid x,u_{t+1})
}{
\pi_{t+1\mid t}(x')
}.
$$

The kernel combines the plausibility of state $x$ after filtering at time $t$
with the probability that it transitions to $x'$. The fixed-interval smoothing
recursion is

$$
\pi_{t\mid T}(x)
=
\int B_t(x\mid x')\pi_{t+1\mid T}(x')\,dx',
\qquad
t=T-1,\ldots,0.
$$

Equivalently, the joint smoothing density factors backward as

$$
p(x_{0:T}\mid y_{1:T},u_{1:T})
=
\pi_T(x_T)
\prod_{t=0}^{T-1}B_t(x_t\mid x_{t+1}).
$$

This factorization leads directly to particle backward simulation
{cite:p}`doucet2001smc`. Store the filtering particles and weights at every
time, then sample the terminal index

$$
I_T
\sim
\operatorname{Categorical}
\left(w_T^{(1)},\ldots,w_T^{(N)}\right).
$$

If the selected index at time $t+1$ is $I_{t+1}=j$, assign each particle at
time $t$ the backward probability

$$
b_t^{(i\mid j)}
=
\frac{
w_t^{(i)}
f_{t+1}(x_{t+1}^{(j)}\mid x_t^{(i)},u_{t+1})
}{
\sum_{\ell=1}^N
w_t^{(\ell)}
f_{t+1}(x_{t+1}^{(j)}\mid x_t^{(\ell)},u_{t+1})
}.
$$

Draw $I_t$ from these probabilities and continue backward to time zero. The
states $x_0^{(I_0)},\ldots,x_T^{(I_T)}$ form one approximate draw from the
joint smoothing distribution. Repeating the backward pass produces a
trajectory ensemble. Unlike filtering alone, this algorithm must evaluate the
transition density between particle pairs.

## Prediction Beyond the Observations

Filtering conditions on data observed through the current time. Prediction
propagates that filtering distribution through future transitions without
using observations that have not yet arrived. The one-step predictive density
is

$$
p(x_{t+1}\mid y_{1:t},u_{1:t+1})
=
\int f_{t+1}(x_{t+1}\mid x_t,u_{t+1})
p(x_t\mid y_{1:t},u_{1:t})\,dx_t.
$$

To approximate it, sample ancestors according to the filtering weights and
propagate each selected particle through $f_{t+1}$. Repeating this operation
under known or planned future controls produces a multistep state forecast.
Future observations can also be simulated from $g_{t+1}$ after each state is
propagated. No likelihood weighting occurs until an actual new observation is
assimilated. Forecast uncertainty is determined jointly by the dynamics and
process noise; it may grow, contract, or approach a long-run level.

## Scientific Applications

In stochastic scientific models, the latent state may be the position and velocity of a mechanical system, the concentration field of a hidden contaminant, or the latent forcing driving a noisy dynamical system. The observations are usually sparse and indirect.

Particle filters are attractive because they can work with:

- nonlinear transition models,
- non-Gaussian process noise,
- nonlinear observation operators,
- multimodal filtering distributions.

The [particle-filtering example](02_filter.ipynb) applies these ideas to a
stochastic Duffing oscillator. The
[particle-smoothing example](03_smoother.ipynb) then uses the stored filtering
clouds and transition density to sample complete latent trajectories.

## Exercises

1. Verify that the filtering update integrates to one and that
   $\prod_{t=1}^T L_t=p(y_{1:T}\mid u_{1:T})$.
2. Derive the bootstrap-filter weight from the sequential importance-sampling
   weight. State separately what remains when resampling has and has not just
   occurred.
3. Explain why the backward probability $b_t^{(i\mid j)}$ depends on the
   transition density but not directly on the observation likelihood at time
   $t+1$.
