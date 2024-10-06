# Required Functional Analysis

It is beyond the scope of this book to cover the basics of functional analysis.
I will only cover the minimum required to understand polynomial chaos.
And I will do that without proofs.

If you are interested in learning more about functional analysis, I recommend you take a course on the subject.
There are also excellent books out there.
The material you would need to understand everything can be found in [Introductory Functional Analysis with Applications](https://www.amazon.com/Introductory-Functional-Analysis-Applications-Kreyszig/dp/0471504599) by Erwin Kreyszig.
Specifically, all the following topics:

+ 1.1. Metric Space
+ 1.2. Further Examples of Metric Spaces
+ 1.3. Open Set, Closed Set, Neighborhood
+ 1.4. Convergence, Cauchy Sequence, Completeness
+ 2.1. Vector Space
+ 2.2. Normed Space. Banach Space
+ 3.1. Inner Product Space. Hiblert Space
+ 3.2. Further Properties of Inner Product Spaces
+ 3.3. Orthogonal Complements and Direct Sums
+ 3.4. Orthonormal Sets and Sequences
+ 3.5. Series Related to Orthonormal Sequences
+ 3.6. Total Orthonormal SEts and Sequences
+ 3.7. Legendre, Hermite and Laguerre Polynomials

## Motivation

Let $\Xi$ be a random vector taking values in a space $\Omega$, typically a subset of $\mathbb{R}^d$.
We will think of a scientific model as a function:

$$
    f: \Omega \to \mathbb{R}
$$

that maps some random input $\Xi$ in $\Omega$ to a real number $Y=f(\Xi)$.
Because the input is random, the output is also random and we want to characterize it fast.

What we want to do, is expand the scientific model in some sort of basis:

$$
    f(\xi) = \sum_{n=0}^{\infty} c_n \phi_n(\xi),
$$

where the $\phi_n$'s will be in some sort of *orthonormal basis*.
Using this expansion, we will be able to propagate uncertainty through the scientific model fast.

To introduce the details of these concepts, we are going to talk about the space in which the scientific model lives.
We will have to show that it is a vector space, introduce an innner product, show that the space is complete and separable.
As I said earlier, we will not explain all these concepts in detail, but we will at least define them and help you develop some intution about them.

## Vector Spaces

Recall from linear algebra that a vector space is a set $V$ with two operations, addition and scalar multiplication, that satisfy the following properties:

1. $u + v = v + u$ for all $u, v \in V$ (commutativity of addition)
2. $u + (v + w) = (u + v) + w$ for all $u, v, w \in V$ (associativity of addition)
3. There exists an element $0 \in V$ such that $u + 0 = u$ for all $u \in V$ (additive identity)
4. For each $u \in V$ there exists an element $-u \in V$ such that $u + (-u) = 0$ (additive inverse)
5. $a(u + v) = au + av$ for all $a \in \mathbb{R}$ and $u, v \in V$ (distributivity of scalar multiplication with respect to vector addition)
6. $(a + b)u = au + bu$ for all $a, b \in \mathbb{R}$ and $u \in V$ (distributivity of scalar multiplication with respect to scalar addition)
7. $a(bu) = (ab)u$ for all $a, b \in \mathbb{R}$ and $u \in V$ (compatibility of scalar multiplication with scalar multiplication)

## The Vector Space $\mathcal{L}^2(\Xi)$

Consider the set of functions:

$$
    \mathcal{L}^2(\Xi) = \left\{ g: \Omega \to \mathbb{R} \mid \mathbb{E}[g^2(\Xi)] < \infty \right\}
$$

We will assume that the scientific model $f$ belongs to this space.
This is also called the space of square integrable functions with respect to the probability measure induced by $\Xi$.

This space is a vector space.
The addition of two functions is defined as:

$$
    (f + g)(\xi) = f(\xi) + g(\xi)
$$

and the scalar multiplication is defined as:

$$
    (af)(\xi) = a f(\xi)
$$

for all $f, g \in \mathcal{L}^2(\Xi)$, $a \in \mathbb{R}$ and $\xi \in \Omega$.
You can check that all the properties of a vector space are satisfied.

## Inner Product

Let $V$ be a vector space.
An inner product on $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ that satisfies the following properties:

1. $\langle u, v \rangle = \langle v, u \rangle$ for all $u, v \in V$ (symmetry)
2. $\langle u + v, w \rangle = \langle u, w \rangle + \langle v, w \rangle$ for all $u, v, w \in V$ (linearity in the first argument)
3. $\langle au, v \rangle = a \langle u, v \rangle$ for all $a \in \mathbb{R}$ and $u, v \in V$ (linearity in the second argument)
4. $\langle u, u \rangle \geq 0$ for all $u \in V$ (positive definiteness)
5. $\langle u, u \rangle = 0$ if and only if $u = 0$ (positive definiteness)

We say that $V$ is an *inner product space* if it has an inner product.

The inner product induces a norm on the vector space:

$$
    \| u \| = \sqrt{\langle u, u \rangle}
$$

The norm tells us how long a vector is.

Once you have a norm, you can define a distance between two vectors:

$$
    d(u, v) = \| u - v \|
$$

As an example, think of $\mathbb{R}^n$ with the usual inner product:

$$
    \langle u, v \rangle = \sum_{i=1}^n u_i v_i
$$

The norm is:

$$
    \| u \| = \sqrt{\sum_{i=1}^n u_i^2}
$$

and the distance is:

$$
    d(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}.
$$

## The Inner Product Space $\mathcal{L}^2(\Xi)$

The inner product of two functions $f, g \in \mathcal{L}^2(\Xi)$ is defined as:

$$
    \langle f, g \rangle = \mathbb{E}[f(\Xi) g(\Xi)]
$$

for all $f, g \in \mathcal{L}^2(\Xi)$.
It is easy to check the first three properties of an inner product as they stem from the linearity of the expectation operator
The fourth and fifth properties are a bit more involved and require some measure theory.

## Convergence, Cauchy Sequence, Completeness
And once you have a distance, you have a metric space and you can talk about convergence and completeness.

A sequence $v_n$ in a vector space $V$ converges to a limit $v$ if:

$$
    \lim_{n \to \infty} \| v_n - v \| = 0
$$

A sequence $v_n$ is a Cauchy sequence if its terms get arbitrarily close to each other, i.e., for all $\epsilon > 0$ there exists an $N$ such that for all $n, m > N$:

$$
    \| v_n - v_m \| < \epsilon.
$$

A Cauchy sequence may or may not converge.
A vector space is *complete* if every Cauchy sequence converges to a limit in the space.

## Hilbert Space

A *Hilbert space* is a complete inner product space.

The Euclidean space $\mathbb{R}^n$ with the usual inner product is a Hilbert space.

The space $\mathcal{L}^2(\Xi)$ is a Hilbert space.

## Separable Space

A vector space is *separable* if it has a countable dense subset.

Countable means that the set can be put in one-to-one correspondence with the natural numbers.
Dense means that every point in the space is a limit of a sequence of points in the subset.

An example of a separable space is $\mathbb{R}^n$.
The dense subset is the set of points with rational coordinates.

The space $\mathcal{L}^2(\Xi)$ is separable.

## Orthonormal Basis

A set of vectors $\{ \phi_n \}_{n=0}^{\infty}$ in a Hilbert space $V$ is an *orthonormal basis* if:

1. $\langle \phi_n, \phi_m \rangle = 0$ for all $n \neq m$ (orthogonality)
2. $\langle \phi_n, \phi_n \rangle = 1$ for all $n$ (normalization)
3. The set is complete, i.e., every vector in $V$ can be written as a linear combination of the basis vectors.

### Theorem 1
Every separable Hilbert space has an orthonormal basis.

### Theorem 2
Under certain technical assumptions about the random variable $\Xi$,$\mathcal{L}^2(\Xi)$ is a separable Hilbert space and thus has an orthonormal basis.

Note that these technical assumptions are highly non-trivial.
But it sufficies to say that they hold if $\Xi$ is one of the standard random variables like the normal, uniform, etc.

## Some Important Properties to Remember

Let $V$ be a Hilbert space with an orthonormal basis $\{ \phi_n \}_{n=0}^{\infty}$.
Let $v$ be a vector in $V$.
Then, the coefficients of the expansion of $v$ in the basis are given by:

$$
    c_n = \langle v, \phi_n \rangle
$$

and the expansion is:

$$
    v = \sum_{n=0}^{\infty} c_n \phi_n
$$

The Parseval's identity states that:

$$
    \| v \|^2 = \sum_{n=0}^{\infty} |c_n|^2
$$

These are both proved by using the properties of the inner product and the orthonormality of the basis.
We are going to need them.

In terms of $\mathcal{L}^2(\Xi)$, the scientific model $f$ can be expanded in an orthonormal basis $\{ \phi_n \}_{n=0}^{\infty}$ as:

$$
    f(\xi) = \sum_{n=0}^{\infty} c_n \phi_n(\xi)
$$

where the coefficients are:

$$
    c_n = \langle f, \phi_n \rangle = \mathbb{E}[f(\Xi) \phi_n(\Xi)]
$$
