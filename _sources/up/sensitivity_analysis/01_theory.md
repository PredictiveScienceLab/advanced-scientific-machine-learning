# Local Sensitivity Analysis for Ordinary Differential Equations

Local sensitivity analysis uses a first-order approximation to propagate small parameter uncertainty through an ordinary differential equation. Let $T>0$, and let the positive integers $n$ and $p$ be the numbers of state variables and parameters. Consider the initial value problem

$$
\begin{aligned}
\dot{\mathbf{x}}(t;\boldsymbol{\theta})
&=\mathbf{f}(\mathbf{x}(t;\boldsymbol{\theta}),t,\boldsymbol{\theta}),\\
\mathbf{x}(0;\boldsymbol{\theta})
&=\mathbf{x}_0(\boldsymbol{\theta}),
\end{aligned}
$$

where

$$
\mathbf{f}:\mathbb{R}^n\times[0,T]\times\mathbb{R}^p\to\mathbb{R}^n
\qquad\text{and}\qquad
\mathbf{x}_0:\mathbb{R}^p\to\mathbb{R}^n
$$

are differentiable maps. For the parameter values considered below, assume that the solution $\mathbf{x}(t;\boldsymbol{\theta})\in\mathbb{R}^n$ exists uniquely for $t\in[0,T]$ and depends differentiably on the parameters.

Represent the uncertain parameter by a random vector $\boldsymbol{\Theta}$ with mean $\boldsymbol{\mu}\in\mathbb{R}^p$ and covariance matrix $\Sigma\in\mathbb{R}^{p\times p}$. The mean $\boldsymbol{\mu}$ serves as the nominal parameter value. Define the sensitivity matrix along the nominal trajectory by

$$
S(t)
=\left.
\frac{\partial\mathbf{x}(t;\boldsymbol{\theta})}
{\partial\boldsymbol{\theta}}
\right|_{\boldsymbol{\theta}=\boldsymbol{\mu}}
\in\mathbb{R}^{n\times p}.
$$

This matrix is the Jacobian of the vector-valued solution with respect to the parameters. For a deterministic parameter value $\boldsymbol{\theta}$ near $\boldsymbol{\mu}$, the first-order Taylor approximation is

$$
\mathbf{x}(t;\boldsymbol{\theta})
\approx
\mathbf{x}(t;\boldsymbol{\mu})
+S(t)(\boldsymbol{\theta}-\boldsymbol{\mu}).
$$

The parameter uncertainty is small in the sense that most of its probability mass lies in a region where this linear approximation is accurate. The uncertain state is $\mathbf{X}(t)=\mathbf{x}(t;\boldsymbol{\Theta})$. As $t$ varies, $(\mathbf{X}(t))_{t\in[0,T]}$ is a **stochastic process**: a collection of random vectors indexed by time. Substituting $\boldsymbol{\Theta}$ for $\boldsymbol{\theta}$ gives its first-order approximation

$$
\widetilde{\mathbf{X}}(t)
=\mathbf{x}(t;\boldsymbol{\mu})
+S(t)(\boldsymbol{\Theta}-\boldsymbol{\mu}).
$$

Its mean function is

$$
\mathbf{m}(t)
=\mathbb{E}[\widetilde{\mathbf{X}}(t)]
=\mathbf{x}(t;\boldsymbol{\mu}),
$$

For $t,t'\in[0,T]$, its cross-time covariance function is

$$
\begin{aligned}
C(t,t')
&=\mathbb{E}\!\left[
(\widetilde{\mathbf{X}}(t)-\mathbf{m}(t))
(\widetilde{\mathbf{X}}(t')-\mathbf{m}(t'))^{\mathsf T}
\right]\\
&=S(t)\Sigma S(t')^{\mathsf T}
\in\mathbb{R}^{n\times n}.
\end{aligned}
$$

Here, the superscript ${\mathsf T}$ denotes transpose. Thus, $C(t,t)$ is the approximate covariance matrix of the state at time $t$. For $i\in\{1,\ldots,n\}$, its diagonal entry $C_{ii}(t,t)$ is the approximate variance of the $i$th state component. If $\boldsymbol{\Theta}$ is Gaussian, then every finite collection of components of $\widetilde{\mathbf{X}}$ at selected times is jointly Gaussian, possibly with a singular covariance matrix. In this sense, $\widetilde{\mathbf{X}}$ is a vector-valued Gaussian process with mean function $\mathbf{m}$ and matrix-valued covariance function $C$. For a non-Gaussian parameter distribution with finite second moments, the same first-order mean and covariance formulas apply, but the process need not be Gaussian.

When nearby trajectories separate rapidly, as they often do in a chaotic system, the sensitivities can grow rapidly as well. A local approximation that is accurate over a short interval can then become unreliable at later times. Its useful time horizon depends on the uncertainty scale and the dynamics.

These formulas require the sensitivity matrix $S(t)$. The next section develops practical ways to compute it.
