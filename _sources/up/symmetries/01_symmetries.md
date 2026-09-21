# Enforcing symmetries in neural networks

A physical model often should not depend on an arbitrary choice of origin or orientation. Symmetry makes this requirement precise and lets a neural network enforce the corresponding transformation laws by construction.

## Symmetry

A mathematical object has a symmetry when a transformation either leaves it unchanged or changes it according to a specified rule. We are particularly interested in physical equations written in coordinates. Their form and predictions should transform consistently when the coordinate origin or orientation changes.

## Groups

Group theory organizes transformations and their compositions. A **group** consists of a set $G$ and a binary operation that maps each ordered pair $(g_1,g_2)\in G\times G$ to an element denoted by $g_1g_2$:

$$
\begin{aligned}
\mathbin{\cdot}:G\times G&\longrightarrow G,\\
(g_1,g_2)&\longmapsto g_1\cdot g_2.
\end{aligned}
$$

We usually omit the symbol $\cdot$ and write $g_1g_2$. The operation must satisfy four properties:

1. **Closure**: For all $g_1, g_2 \in G$, $g_1 g_2 \in G$.

2. **Associativity**: For all $g_1, g_2, g_3 \in G$, $(g_1 g_2) g_3 = g_1 (g_2 g_3)$.

3. **Identity**: There exists an element $e \in G$ such that for all $g \in G$, $e g = g e = g$.

4. **Inverses**: For all $g \in G$, there exists an element $g^{-1} \in G$ such that $g g^{-1} = g^{-1} g = e$.


### Examples of groups

Many familiar number systems and transformations form groups. Here are several examples.

#### The set of integers with addition

The set of integers $\mathbb{Z}$ forms a group under addition $+$.
We denote this group by $(\mathbb{Z}, +)$.
The set of integers is closed under addition, the addition is associative, the identity element is $0$, and the inverse of an integer $n$ is $-n$.

#### The set of non-zero rational numbers with multiplication

The set of non-zero rational numbers, $\mathbb{Q}\setminus \{0\}$, forms a group under multiplication $\times$.
We denote this group by $(\mathbb{Q}\setminus\{0\}, \times)$.
The nonzero rational numbers are closed under multiplication, multiplication is associative, the identity element is $1$, and the inverse of $q\ne 0$ is $1/q$.

#### The set of real numbers with addition

The set of real numbers $\mathbb{R}$ forms a group under addition $+$, denoted by $(\mathbb{R},+)$. The identity element is $0$, and the inverse of a real number $x$ is $-x$.

#### The set of non-zero real numbers with multiplication

The set of non-zero real numbers, $\mathbb{R}\setminus \{0\}$, forms a group under multiplication. Its identity is $1$, and the inverse of $x\ne 0$ is $1/x$.


#### The translation group

Consider the Euclidean space $\mathbb{R}^3$ with elementwise addition:

$$
(x_1,x_2,x_3) + (y_1,y_2,y_3) = (x_1+y_1, x_2+y_2, x_3+y_3).
$$

For each translation vector $\mathbf{b}\in\mathbb{R}^3$, define the map $t_{\mathbf{b}}:\mathbb{R}^3\to\mathbb{R}^3$ by

$$
t_{\mathbf{b}}(\mathbf{x})=\mathbf{x}+\mathbf{b}.
$$

The translation group is

$$
T(3)=\{t_{\mathbf{b}}\mid \mathbf{b}\in\mathbb{R}^3\},
$$

with function composition as its operation. Since $t_{\mathbf{b}_1}\circ t_{\mathbf{b}_2}=t_{\mathbf{b}_1+\mathbf{b}_2}$, the map $\mathbf{b}\mapsto t_{\mathbf{b}}$ is an isomorphism from $(\mathbb{R}^3,+)$ to $T(3)$. The same construction defines $T(n)$ in $n$ dimensions.

#### The general linear group

The general linear group of dimension $n$, denoted by $GL(n,\mathbb{R})$, is the set of all invertible $n \times n$ matrices.
Mathematically,

$$
GL(n,\mathbb{R}) = \{ A \in \mathbb{R}^{n \times n} \mid \det(A) \neq 0 \}.
$$

The group operation is matrix multiplication. The identity element is the identity matrix $I$, and the inverse of $A$ is its matrix inverse $A^{-1}$.

#### The special orthogonal group

The special orthogonal group of dimension $n$, denoted by $SO(n)$, is the set of all $n \times n$ orthogonal matrices with determinant one.
Mathematically,

$$
SO(n) = \{ A \in \mathbb{R}^{n \times n} \mid A A^T = I, \det(A) = 1 \}.
$$

The group operation is matrix multiplication. The matrices in $SO(n)$ represent orientation-preserving rotations in $n$ dimensions. If $A,B\in SO(n)$, then $AB$ is orthogonal and $\det(AB)=1$, so $SO(n)$ is closed under multiplication.

$SO(n)$ is a subgroup of $GL(n,\mathbb{R})$. We write:

$$
SO(n) \le GL(n,\mathbb{R}).
$$

This means that $SO(n)$ is a group in its own right, and it is a subset of $GL(n,\mathbb{R})$.

Consider two examples in three-dimensional space. A rotation through angle $\theta$ about the $z$-axis is represented by the matrix $R_1\in SO(3)$:

$$
R_1 = \begin{bmatrix}
\cos(\theta) & -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Similarly, a rotation through angle $\phi$ about the $y$-axis is represented by $R_2\in SO(3)$:

$$
R_2 = \begin{bmatrix}
\cos(\phi) & 0 & \sin(\phi) \\
0 & 1 & 0 \\
-\sin(\phi) & 0 & \cos(\phi)
\end{bmatrix}.
$$

Their product is

$$
R_3 = R_1 R_2 = \begin{bmatrix}
\cos(\theta)\cos(\phi) & -\sin(\theta) & \cos(\theta)\sin(\phi) \\
\sin(\theta)\cos(\phi) & \cos(\theta) & \sin(\theta)\sin(\phi) \\
-\sin(\phi) & 0 & \cos(\phi)
\end{bmatrix}.
$$

The product satisfies $\det(R_3)=1$, as required by closure. In general, $R_2R_1\ne R_1R_2$, so rotations in three dimensions do not commute.

#### The orthogonal group

The orthogonal group of dimension $n$, denoted by $O(n)$, is the set of all $n \times n$ orthogonal matrices.
Mathematically,

$$
O(n) = \{ A \in \mathbb{R}^{n \times n} \mid A A^T = I \}.
$$

The group operation is matrix multiplication. The matrices in $O(n)$ are precisely the linear transformations that preserve Euclidean lengths and angles. They include rotations and reflections. The product of two orthogonal matrices is orthogonal, so $O(n)$ is closed under multiplication.

$O(n)$ is a subgroup of $GL(n,\mathbb{R})$ and it contains $SO(n)$ as a subgroup.
We write:

$$
SO(n) \le O(n) \le GL(n,\mathbb{R}).
$$

An element of the orthogonal group that is not in the special orthogonal group is the reflection

$$
R_4 = \begin{bmatrix}
-1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

It satisfies $R_4R_4^T=I$ and $\det(R_4)=-1$.

## Group homorphisms

A homomorphism is a map from one group to another that preserves multiplication.
It may identify several elements of the source group, so it does not generally amount to relabeling the group elements.

Mathematically, let $G$ and $H$ be two groups.
If we can find a function $\phi: G \to H$ such that:

$$
\phi(g_1 g_2) = \phi(g_1) \phi(g_2),
$$

for all $g_1, g_2 \in G$, then $\phi$ is a homomorphism from $G$ to $H$.

If the homomorphism is bijective, i.e., one-to-one and onto, then it is called an isomorphism.
When $G$ and $H$ are isomorphic, we write $G \cong H$.

### Example: A homomorphism from $\mathbb{Z}$ to $\mathbb{Z}_3$

For a homomorphism that is not an isomorphism, consider the group of integers modulo $3$, denoted by $\mathbb{Z}_3$. Write $[x]_3$ for the congruence class containing all integers with the same remainder as $x$ after division by $3$. The three classes are

$$
\mathbb{Z}_3=\{[0]_3,[1]_3,[2]_3\}.
$$

The group operation is addition modulo $3$; for example, $[1]_3+[2]_3=[0]_3$.

Now, consider the function

$$
f: \mathbb{Z} \to \mathbb{Z}_3,
$$

that sends an integer to its congruence class:

$$
f(x)=[x]_3.
$$

This is a homomorphism because

$$
f(x+y)=[x+y]_3=[x]_3+[y]_3=f(x)+f(y).
$$

But it is not an isomorphism because it is not one-to-one.

### Example: The group of real numbers with addition is isomorphic to the group of positive real numbers with multiplication

Let $\mathbb{R}^+$ be the set of positive real numbers. It is a group under multiplication, with identity $1$ and inverse $1/x$ for each $x>0$.

Consider the function:

$$
f: \mathbb{R} \to \mathbb{R}^+,
$$

defined by:

$$
f(x) = e^x.
$$

The function is one-to-one and onto, and

$$
f(x+y)=e^{x+y}=e^xe^y=f(x)f(y).
$$

It is therefore an isomorphism from $(\mathbb{R},+)$ to $(\mathbb{R}^+,\times)$.

## Group of transformations

Let $V$ be a set. The set $\operatorname{Bij}(V)$ of all bijections from $V$ to itself is a group under function composition. A **group of transformations** of $V$ is a subgroup

$$
G\leq \operatorname{Bij}(V).
$$

Equivalently, $G$ is a collection of bijections that contains the identity map and is closed under composition and inverses.
So, an element $g$ of $G$ is a function:

$$
g : V \to V,
$$

that is one-to-one and onto (bijective).
Here the group operation is the composition of functions.
So, if $g_1$ and $g_2$ are in $G$, then the composition:

$$
g = g_1 \circ g_2,
$$

defined by:

$$
x\mapsto (g_1 \circ g_2)(x) = g_1(g_2(x)),
$$

is also in $G$.

Function composition is associative:

$$
(g_1 \circ g_2) \circ g_3 = g_1 \circ (g_2 \circ g_3).
$$

The identity element is the identity function

$$
e(x) = x,
$$

and the inverse of each $g\in G$ is its inverse bijection $g^{-1}\in G$. Merely requiring a collection of bijections to be closed under composition would not be enough; the identity and inverse conditions are part of the subgroup definition.

## Group actions

A group can transform another set without itself being presented as a collection of transformations. An **action** of a group $G$ on a set $X$ assigns a map $D_X(g):X\to X$ to every $g\in G$ such that

$$
D_X(e)=\operatorname{id}_X,
\qquad
D_X(g_1g_2)=D_X(g_1)\circ D_X(g_2).
$$

Unlike a group of transformations, a general action may assign the same transformation to more than one group element. The action also does not require $X$ to be a vector space or the maps $D_X(g)$ to be linear.

### Example: Permutations

The permutation group $S_N$ consists of all bijections $\sigma:\{1,\ldots,N\}\to\{1,\ldots,N\}$ under composition. It acts on an ordered collection $x=(x_1,\ldots,x_N)$ by relabeling its entries:

$$
\bigl(D_X(\sigma)x\bigr)_i=x_{\sigma^{-1}(i)},
\qquad i=1,\ldots,N.
$$

The inverse in this definition ensures that $D_X(\sigma_1\sigma_2)=D_X(\sigma_1)\circ D_X(\sigma_2)$. For a collection of identical atoms, a predicted total energy should be invariant to this relabeling, while atom-indexed outputs should be permuted in the same way {cite:p}`batzner2022e3equivariant`.

## Group representations

Let $W$ be a finite-dimensional real vector space, and let $GL(W)$ denote the group of invertible linear maps from $W$ to itself under composition. A **representation** of a group $G$ on $W$ is a homomorphism

$$
D:G\to GL(W).
$$

Thus $D(e)=I$ and

$$
D(g_1g_2)=D(g_1)D(g_2).
$$

After choosing a basis of $W$, each linear map $D(g)$ is a matrix.
A representation need not be injective; an injective representation is called **faithful**.

### Example: The group of invertible linear transformations is isomorphic to the general linear group

Let $V$ be a *real* vector space of dimension $n$ and let $GL(V)$ be the set of all invertible linear transformations of $V$, i.e.,

$$
GL(V) = \{ f: V \to V \mid f \text{ is linear and invertible} \}.
$$

$GL(V)$ is a group under function composition.

Choosing a basis identifies this group with the matrix group $GL(n,\mathbb{R})$.
Let $B = \{ \mathbf{e}_1, \ldots, \mathbf{e}_n \}$ be a basis of $V$.
Then, any linear transformation $f \in GL(V)$ can be represented by a matrix $A$ such that:

$$
f(\mathbf{e}_i) = \sum_{j=1}^n A_{ji} \mathbf{e}_j.
$$

The matrix $A$ is invertible because $f$ is invertible.

The map

$$
\phi : GL(V) \to GL(n,\mathbb{R}),
$$

that sends a linear transformation to its matrix representation

$$
f \mapsto \phi(f) = A,
$$

is an isomorphism between $GL(V)$ and $GL(n,\mathbb{R})$:

$$
GL(V) \cong GL(n,\mathbb{R}).
$$

## The Euclidean group

The **Euclidean group** $E(n)$ collects the rigid transformations of $\mathbb{R}^n$. Its elements are the isometries, meaning the maps $f:\mathbb{R}^n\to\mathbb{R}^n$ that preserve Euclidean distance:

$$
\| f(\mathbf{x}) - f(\mathbf{y}) \| = \| \mathbf{x} - \mathbf{y} \|,
$$

for all $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$. The group operation is function composition.

Every Euclidean isometry has a unique affine form

$$
f(\mathbf{x})=A\mathbf{x}+\mathbf{b},
$$

where $A\in O(n)$ and $\mathbf{b}\in\mathbb{R}^n$. To see why, set $\mathbf{b}=f(\mathbf{0})$ and define $q(\mathbf{x})=f(\mathbf{x})-\mathbf{b}$. Distance preservation gives

$$
\langle q(\mathbf{x}),q(\mathbf{y})\rangle
=\frac{1}{2}\left(
\|q(\mathbf{x})\|^2+\|q(\mathbf{y})\|^2
-\|q(\mathbf{x})-q(\mathbf{y})\|^2
\right)
=\langle\mathbf{x},\mathbf{y}\rangle.
$$

An inner-product-preserving map that fixes the origin is linear, so $q(\mathbf{x})=A\mathbf{x}$ with $A^{\mathsf T}A=I$.

We identify the isometry with the pair $(\mathbf{b},A)$. Its action on a point $\mathbf{x}\in\mathbb{R}^n$ is

$$
(\mathbf{b},A)\cdot\mathbf{x}=A\mathbf{x}+\mathbf{b}.
$$

Composing two such actions gives the pair multiplication law

$$
(\mathbf{b}_1,A_1)(\mathbf{b}_2,A_2)
=\left(\mathbf{b}_1+A_1\mathbf{b}_2,A_1A_2\right).
$$

The term $A_1\mathbf{b}_2$ shows that the orthogonal component acts on translation vectors. This interaction is the defining feature of a semidirect product.

## Semidirect products

Let $G$ be a group with subgroups $H$ and $K$. Assume that $H$ is normal in $G$, that the two subgroups intersect only at the identity, and that every element of $G$ is a product of an element of $H$ and an element of $K$:

$$
H\mathrel{\trianglelefteq}G,
\qquad
K\leq G,
\qquad
H\cap K=\{e\},
\qquad
G=HK.
$$

Normality means that

$$
ghg^{-1}\in H
$$

for every $g\in G$ and $h\in H$. The factorization $g=hk$ is unique. Indeed, if $hk=h'k'$ with $h,h'\in H$ and $k,k'\in K$, then

$$
(h')^{-1}h=k'k^{-1}\in H\cap K=\{e\},
$$

so $h=h'$ and $k=k'$. Consequently, the map

$$
\phi:H\times K\longrightarrow G,
\qquad
\phi(h,k)=hk,
$$

is a bijection.

Normality also ensures that $khk^{-1}\in H$ for each $k\in K$ and $h\in H$. Multiplication in $G$ therefore gives

$$
(h_1k_1)(h_2k_2)
=h_1\left(k_1h_2k_1^{-1}\right)(k_1k_2).
$$

This identity determines the multiplication on $H\times K$:

$$
(h_1,k_1)(h_2,k_2)
=\left(h_1\left(k_1h_2k_1^{-1}\right),k_1k_2\right).
$$

With this operation, $H\times K$ is the **semidirect product** $H\rtimes K$. The bijection $\phi$ transports the group structure of $G$ to $H\rtimes K$, so it preserves multiplication and gives the isomorphism

$$
G\cong H\rtimes K.
$$

### Connection to the Euclidean group

Under the pair representation of $E(n)$, the translation $t_{\mathbf{b}}$ is $(\mathbf{b},I)$ and the orthogonal map $A$ is $(\mathbf{0},A)$. Every pair has the unique factorization

$$
(\mathbf{b},A)=(\mathbf{b},I)(\mathbf{0},A),
$$

and $T(n)\cap O(n)$ contains only the identity. Translations also form a normal subgroup: if $g=(\mathbf{c},A)\in E(n)$, then

$$
g\,t_{\mathbf{b}}\,g^{-1}=t_{A\mathbf{b}}.
$$

The orthogonal part therefore acts on translation vectors by $\mathbf{b}\mapsto A\mathbf{b}$. The semidirect-product multiplication is exactly the pair law derived above, so

$$
E(n)\cong T(n)\rtimes O(n).
$$

## Transformation laws for physical quantities

We use the active convention: a Euclidean transformation $g=(\mathbf{b},A)$ moves a position according to

$$
\mathbf{r}\mapsto A\mathbf{r}+\mathbf{b}.
$$

An equivalent passive change of coordinates is represented by the inverse group element. Other physical quantities transform according to their type. Under $A\in O(n)$, a true scalar $s$, a pseudoscalar $p$, and a polar vector $\mathbf{v}$ transform as

$$
s\mapsto s,
\qquad
p\mapsto \det(A)p,
\qquad
\mathbf{v}\mapsto A\mathbf{v}.
$$

In three dimensions, an axial vector $\mathbf{w}$ transforms as $\mathbf{w}\mapsto\det(A)A\mathbf{w}$. Thus polar and axial vectors transform identically under rotations and acquire opposite parity under reflections. Velocity and force are polar vectors, whereas angular momentum is an axial vector.

A rank-two polar Cartesian tensor $\mathbf{T}$ transforms as $\mathbf{T}\mapsto A\mathbf{T}A^{\mathsf T}$. More generally, the components of a rank-$k$ polar Cartesian tensor transform according to

$$
T'_{i_1\cdots i_k}
=\sum_{j_1=1}^n\cdots\sum_{j_k=1}^n
A_{i_1j_1}\cdots A_{i_kj_k}T_{j_1\cdots j_k}.
$$

A pseudotensor acquires an additional factor $\det(A)$. Translations affect positions, while free vectors and tensors depend only on the orthogonal part $A$. These rules define different linear representations of the same group, whereas the position rule is affine. A model whose input contains several types must apply the appropriate transformation to each component.

## Invariance

Let $D_X(g)$ denote the action of $G$ on an input space $X$. A scalar-valued function $f:X\to\mathbb{R}$ is invariant if

$$
f(D_X(g)x)=f(x)
$$

for every $x\in X$ and $g\in G$. For an isolated molecule in the absence of external fields, the potential energy should remain unchanged when all atomic positions undergo the same rigid transformation {cite:p}`batzner2022e3equivariant`.

For a finite group with $|G|$ elements, let $h_\theta:X\to\mathbb{R}$ be any scalar function with parameters $\theta$. Averaging over the group produces the invariant function

$$
f_\theta(x)=\frac{1}{|G|}\sum_{g\in G}h_\theta(D_X(g)x).
$$

Reindexing the sum by right multiplication shows that $f_\theta$ is invariant. For a compact continuous group, the corresponding average is

$$
f_\theta(x)=\int_G h_\theta(D_X(g)x)\,\mathrm{d}\mu(g),
$$

where $\mu$ is normalized Haar measure, the probability measure on $G$ that is unchanged by left or right multiplication by a fixed group element.

## Equivariance

Let $D_X$ and $D_Y$ be actions of $G$ on the input space $X$ and output space $Y$. A map $f:X\to Y$ is equivariant if

$$
f(D_X(g)x)=D_Y(g)f(x)
$$

for every $x\in X$ and $g\in G$. The two actions generally differ.

Invariance is the special case of equivariance in which the output action is trivial: $D_Y(g)y=y$ for every $g\in G$ and $y\in Y$.

For a molecule with the position tuple

$$
\mathbf{x}=(\mathbf{r}_1,\ldots,\mathbf{r}_N),
$$

the Euclidean group acts on the input by

$$
D_X(\mathbf{b},A)\mathbf{x}
=\left(A\mathbf{r}_1+\mathbf{b},\ldots,A\mathbf{r}_N+\mathbf{b}\right).
$$

If $f(\mathbf{x})=(\mathbf{F}_1,\ldots,\mathbf{F}_N)$ returns the atomic forces, the output action is

$$
D_Y(\mathbf{b},A)f(\mathbf{x})
=\left(A\mathbf{F}_1,\ldots,A\mathbf{F}_N\right).
$$

An equivariant force model therefore rotates or reflects its predicted forces with the molecule and leaves them unchanged under a common translation {cite:p}`batzner2022e3equivariant`.

## Constructing equivariant neural networks

Group-equivariant convolutional networks extend convolutional weight sharing from translations to larger symmetry groups {cite:p}`cohen2016group`.
Equivariance can be preserved layer by layer. The following closure rules provide the basic construction.

First, equivariant maps with the same output action are closed under addition. Let $Y$ be a vector space on which $G$ acts through the linear representation $D_Y$. If $f_1,f_2:X\to Y$ are equivariant, then

$$
\begin{aligned}
(f_1+f_2)(D_X(g)x)
&=D_Y(g)f_1(x)+D_Y(g)f_2(x)\\
&=D_Y(g)(f_1+f_2)(x).
\end{aligned}
$$

Second, equivariant maps are closed under compatible composition. If $f:X\to Y$ is equivariant for the actions $D_X,D_Y$ and $h:Y\to Z$ is equivariant for $D_Y,D_Z$, then

$$
(h\circ f)(D_X(g)x)=D_Z(g)(h\circ f)(x).
$$

Third, tensor products combine feature types. Let $V$ and $W$ carry the linear representations $D_V$ and $D_W$. If $a:X\to V$ and $b:X\to W$ are equivariant, define their pointwise tensor product by $(a\otimes b)(x)=a(x)\otimes b(x)$. It satisfies

$$
(a\otimes b)(D_X(g)x)
=\left(D_V(g)\otimes D_W(g)\right)(a(x)\otimes b(x)).
$$

For polar vectors under $O(n)$, the dot product is an invariant scalar and the outer product is a rank-two polar tensor. In three dimensions, the cross product is an axial vector: for $A\in O(3)$,

$$
(A\mathbf{u})\times(A\mathbf{v})
=\det(A)A(\mathbf{u}\times\mathbf{v}).
$$

The determinant factor is required when reflections are included.

Linear layers between feature types must also respect the group action. A linear map $L:V\to W$ is equivariant precisely when it is an intertwiner:

$$
L D_V(g)=D_W(g)L
$$

for every $g\in G$. An invariant scalar can multiply, or **gate**, an equivariant tensor without changing its type. Arbitrary componentwise nonlinearities do not generally preserve vector or tensor equivariance, so nonlinear layers must be assembled from type-preserving operations such as invariant gates and tensor products {cite:p}`geiger2022e3nn`.

An equivariant network therefore tags every channel by its representation, uses intertwiners for linear mixing, and combines channels only through operations with known output types. An **irreducible representation** is a feature type with no nonzero proper subspace preserved by every group element. Euclidean neural networks organize their channels into these types and use spherical harmonics, angular basis functions with known rotation laws, to construct equivariant features {cite:p}`geiger2022e3nn`.

## Exercises

1. Verify directly that the permutation rule

   $$
   \bigl(D_X(\sigma)x\bigr)_i=x_{\sigma^{-1}(i)}
   $$

   satisfies the identity and composition requirements of a group action. Explain why a total energy should be invariant to this action while atom-indexed forces should be equivariant.

2. Derive the inverse of $(\mathbf{b},A)\in E(n)$ and show that

   $$
   (\mathbf{b},A)^{-1}
   =\left(-A^{\mathsf T}\mathbf{b},A^{\mathsf T}\right).
   $$

   Use the result to verify $g\,t_{\mathbf{c}}\,g^{-1}=t_{A\mathbf{c}}$.

3. Prove the finite-group averaging formula is invariant. Then use the transformation laws to classify the dot product, outer product, and three-dimensional cross product of two polar vectors.

## From transformation laws to an equivariant model

The next notebook implements these constructions with e3nn. It represents features by irreducible $O(3)$ types, forms angular features from spherical harmonics of relative positions, uses tensor products to build equivariant layers, and checks the resulting $E(3)$ transformation laws numerically.
