(sec-hup-op-01-reading)=
# Learning Operators

Operator learning seeks one surrogate for a solver's entire input-to-output map, whether that map sends a coefficient field to a PDE solution or a forcing history to a state trajectory.

## Solution operators

Let $D_{\mathrm{in}}\subset\mathbb{R}^{d_{\mathrm{in}}}$ and $D_{\mathrm{out}}\subset\mathbb{R}^{d_{\mathrm{out}}}$ be the input and output domains. Let $\mathcal{A}$ be an admissible subset of a normed space of input functions on $D_{\mathrm{in}}$, and let $\mathcal{U}$ be a normed space of output functions on $D_{\mathrm{out}}$. We reserve *functional* for a scalar-valued map $\mathcal{A}\to\mathbb{R}$ and use *operator* for a function-valued map

$$
\mathcal{G}^{\dagger}:\mathcal{A}\to\mathcal{U}.
$$

The superscript $\dagger$ labels the reference operator that generates the data; it does not denote an adjoint. The operator may be nonlinear.

The antiderivative gives a simple example. Take $\mathcal{A}=\mathcal{U}=C([0,1])$ with the supremum norm. Define

$$
\big[\mathcal{G}^{\dagger}(a)\big](y)
=
\int_0^y a(s)\,\mathrm{d}s,
\qquad 0\leq y\leq 1.
$$

This operator is linear, and it is continuous because

$$
\left\lVert\mathcal{G}^{\dagger}(a)\right\rVert_{\infty}
\leq
\lVert a\rVert_{\infty}.
$$

A coefficient-to-solution map provides a nonlinear example. Let $\Omega$ be a bounded Lipschitz domain: intuitively, a finite region whose boundary may have corners but has no cusps or arbitrarily rough features. Fix $f\in L^2(\Omega)$, impose homogeneous Dirichlet boundary conditions, and choose constants $0<a_{\min}\leq a_{\max}<\infty$. The admissible coefficient fields are

$$
\mathcal{A}
=
\left\{
a\in L^{\infty}(\Omega):
0<a_{\min}\leq a(x)\leq a_{\max}
\text{ almost everywhere}
\right\}.
$$

For $a\in\mathcal{A}$, consider the boundary-value problem

$$
-\nabla\!\cdot\!\big(a(x)\nabla u_a(x)\big)=f(x)
\quad\text{in }\Omega,
\qquad
u_a=0
\quad\text{on }\partial\Omega.
$$

A weak solution satisfies the PDE in an integrated sense, so it need not have the classical second derivatives that appear in the differential equation. The space $H_0^1(\Omega)$ consists of functions with square-integrable values and first derivatives and zero boundary trace. The Lax--Milgram theorem gives a unique weak solution $u_a\in H_0^1(\Omega)$ for every $a\in\mathcal{A}$. The solution operator is therefore

$$
\mathcal{G}^{\dagger}:\mathcal{A}\to H_0^1(\Omega),
\qquad
\mathcal{G}^{\dagger}(a)=u_a.
$$

The PDE is linear in $u_a$ for fixed $a$, but the map $a\mapsto u_a$ is generally nonlinear.

## From finite coordinates to an operator

The preceding singular-value, principal-component, and Karhunen--Loève constructions represent functions by finite coordinate vectors. Let

$$
E_r:\mathcal{A}\to\mathbb{R}^r
$$

be an input encoder, let $f_{\theta}:\mathbb{R}^r\to\mathbb{R}^s$ be a learned coordinate map, and let $D_s:\mathbb{R}^s\to\mathcal{U}$ reconstruct an output function. Their composition

$$
\widehat{\mathcal{G}}^{\mathrm{coord}}_{\theta}
=
D_s\circ f_{\theta}\circ E_r
$$

already approximates the continuum operator through finite coordinates. Every implementation uses finite representations. Operator-learning architectures organize those representations so that the learned model can be evaluated as a function-valued map and, in some architectures, can share parameters across compatible discretizations.

## Learning from finite observations

A computer receives only finite information about each function. For scalar-valued inputs and outputs, choose input sensors $X=(x_1,\ldots,x_m)$ and output locations $Y=(y_1,\ldots,y_n)$. The corresponding observation maps are

$$
P_Xa
=
\big(a(x_1),\ldots,a(x_m)\big),
\qquad
Q_Yu
=
\big(u(y_1),\ldots,u(y_n)\big).
$$

A value at an isolated sensor is not always defined by the function-space model: changing a function at that one point can leave all its integral properties unchanged. We can instead measure an average over a small region around the sensor.

Formally, $L^2$ and Sobolev spaces such as $H^1$ identify functions that agree almost everywhere; each such collection is called an *equivalence class*. Sobolev spaces also control derivatives through integral norms. Point evaluation is not continuous on $H^1(\Omega)$ in general when the spatial dimension is at least two. In these settings, we use continuous linear measurements, such as cell averages or basis coefficients, also called *bounded linear observation functionals*. Pointwise values from a numerical solver refer to its finite-dimensional reconstruction, which supplies specific values at the sensor locations.

Finite observations impose an information limit. If $P_Xa=P_X\widetilde a$, then every model that depends only on $P_Xa$ must make the same prediction for $a$ and $\widetilde a$. Such a model cannot reproduce both outputs when $\mathcal{G}^{\dagger}(a)\neq\mathcal{G}^{\dagger}(\widetilde a)$. Sensor placement and basis truncation are modeling choices that control the approximation.

Suppose $\mu$ is a probability measure supported on $\mathcal{A}$, the training functions $a_1,\ldots,a_N$ are sampled from $\mu$, and a reference solver supplies reconstructed functions $u_i\approx\mathcal{G}^{\dagger}(a_i)$ or finite observations of them. Assuming that the expectation below is finite, the ideal population error of a learned operator $\mathcal{G}_{\theta}$ is

$$
\mathcal{R}(\theta)
=
\mathbb{E}_{a\sim\mu}
\left[
\left\lVert
\mathcal{G}_{\theta}(a)-\mathcal{G}^{\dagger}(a)
\right\rVert_{\mathcal{U}}^2
\right].
$$

Training replaces this expectation and function-space norm by finite samples. For pointwise output data, let $Y_i=(y_{i1},\ldots,y_{in_i})$ be the query set for the $i$th function. A common weighted empirical loss is

$$
\widehat{\mathcal{R}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^N
\sum_{j=1}^{n_i}
w_{ij}
\left|
\mathcal{G}_{\theta}(a_i)(y_{ij})-u_i(y_{ij})
\right|^2,
$$

where the nonnegative weights satisfy $\sum_j w_{ij}=1$. If $\mathcal{U}=L^2(D_{\mathrm{out}},\rho)$ for a probability measure $\rho$ and the weights form a quadrature or sampling rule for $\rho$, this sum approximates the squared $L^2$ norm. It should not be interpreted as an approximation of a supremum or $H^1$ norm. A DeepONet uses observations of the form $(P_Xa_i,y_{ij},u_i(y_{ij}))$. An FNO usually receives and predicts whole arrays $(P_Xa_i,Q_Yu_i)$ on a grid. The train/test split must be made at the level of input functions; splitting individual query triples can place values from the same function in both sets.

## Generalization in operator learning

Held-out-input generalization measures prediction for a new function drawn from the target distribution $\mu$. Output-location generalization asks whether a point-query model can evaluate the output at locations absent from its training queries. Resolution generalization asks whether one set of parameters remains accurate after the input or output grid changes. If geometry or boundary data are components of $a$ and remain distributed according to $\mu$, a new realization is still a held-out input. Moving to a geometry, boundary-condition type or range, or input law not represented by $\mu$ is a further extrapolation problem. Each claim requires its own test.

## Two representation choices

### DeepONet

A DeepONet sends the sensor values $P_Xa$ through a branch network and the output coordinate $y$ through a trunk network. If both networks produce $p$ features, the prediction has the form

$$
\mathcal{G}_{\theta}(a)(y)
=
c_{\theta}
+
\sum_{k=1}^p
b_{\theta,k}(P_Xa)\,t_{\theta,k}(y).
$$

The branch features describe the observed input function, while the trunk features describe where the output is queried. Universal approximation results motivate this finite-sensor, finite-feature form in a setting such as a compact set $K\subset C(D_{\mathrm{in}})$ with the supremum norm and a continuous operator $\mathcal{G}^{\dagger}:K\to C(D_{\mathrm{out}})$ {cite:p}`chen1995universal,lu2021learning`. These assumptions do not automatically hold for the full $L^{\infty}$ coefficient set used in the PDE example. The results establish expressivity under their assumptions; successful optimization and generalization from finite data require separate evidence, and the standard branch network remains tied to its chosen input sensors.

### Fourier neural operators

A neural-operator layer can combine a pointwise linear map with a learned integral operator. Let $v_{\ell}:D\to\mathbb{R}^{c_{\ell}}$ be the feature field at layer $\ell$ on a common domain $D=D_{\mathrm{in}}=D_{\mathrm{out}}$, let $W_{\ell}:\mathbb{R}^{c_{\ell}}\to\mathbb{R}^{c_{\ell+1}}$ act on feature channels, and let $\kappa_{\ell,\theta}:D\times D\to\mathbb{R}^{c_{\ell+1}\times c_{\ell}}$ be a learned kernel. A representative layer is

$$
v_{\ell+1}(x)
=
\sigma\!\left(
W_{\ell}v_{\ell}(x)
+
\int_{D}
\kappa_{\ell,\theta}(x,z)v_{\ell}(z)\,\mathrm{d}z
\right),
$$

where $\sigma$ acts componentwise {cite:p}`kovachki2023neuraloperator`. On a periodic domain, an FNO specializes the kernel to the translation-invariant form ${\kappa_{\ell,\theta}(x,z)=\kappa_{\ell,\theta}(x-z)}$. The integral then becomes a convolution, represented by the Fourier multiplier

$$
\big(\mathcal{K}_{\ell,\theta}v_{\ell}\big)(x)
=
\mathcal{F}^{-1}
\!\left(
R_{\ell,\theta}(k)\,\mathcal{F}(v_{\ell})(k)
\right)(x),
$$

where the learned matrix $R_{\ell,\theta}(k)$ is retained for only finitely many Fourier modes and set to zero outside that set. The convolution can then be evaluated efficiently with fast Fourier transforms {cite:p}`li2021fourier`.

The same learned spectral parameters can be evaluated on compatible uniform grids when the retained frequencies are resolved. Parameter sharing is an architectural property; accurate zero-shot resolution transfer requires a separate test. Accuracy across grids can still be limited by spatial discretization, spectral truncation, and aliasing.

The DeepONet example that follows instantiates the point-query construction for the antiderivative operator. The FNO companion notebook then learns a grid-to-grid Darcy-flow solution operator.

## Exercises

1. Prove the linearity and supremum-norm bound of the antiderivative operator. For fixed sensors $x_1,\ldots,x_m$, construct two continuous functions that agree at every sensor but have different antiderivatives. Explain what this implies for any model that uses only $P_Xa$.

2. Consider

   $$
   -a u''(x)=1,
   \qquad
   u(0)=u(1)=0,
   $$

   where $a>0$ is constant. Verify that $u_a(x)=x(1-x)/(2a)$ and use this expression to show that the coefficient-to-solution map $a\mapsto u_a$ is nonlinear.

3. For an operator-learning experiment, distinguish evidence for held-out-input, output-location, resolution, and new-geometry generalization. Design a data split that tests held-out-input generalization without placing query values from one function in both training and test sets.
