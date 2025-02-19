# Enforcing Symmetries in Neural Networks

We are going to explain what we mean by symmetries, why they are important, and how we can enforce them in neural networks.

## What are symmetries?

Generally speaking, we say that some mathematical object possesses a certain symmetry if it either remains unchanged, or changes in a predictable way, under some transformation.
We are particularly interested in symmetries of mathematical equations that describe a physical system.
Typically these equations are written in terms of variables expressed in a coordinate system.
But the origin and orientation of that coordinate system are arbitrary, and the form of the equations should not depend on them.

## How do we describe symmetries?
Symmetries are described using the language of group theory.
A group $G$ is a set of elements typically denoted by $g_1, g_2, \ldots$, along with a binary operation:

$$
\cdot:G\times G\to G,
$$ 

that combines two elements to produce a third element:

$$
(g_1,g_2) \mapsto \cdot (g_1,g_2) \equiv g_1 \cdot g_2 = g_3.
$$

Typically, we do not need to write the $\cdot$ explicitly, and we can simply write $g_1 g_2 = g_3$.
The group must satisfy the following properties:

1. **Closure**: For all $g_1, g_2 \in G$, $g_1 g_2 \in G$.

2. **Associativity**: For all $g_1, g_2, g_3 \in G$, $(g_1 g_2) g_3 = g_1 (g_2 g_3)$.

3. **Identity**: There exists an element $e \in G$ such that for all $g \in G$, $e g = g e = g$.

4. **Inverses**: For all $g \in G$, there exists an element $g^{-1} \in G$ such that $g g^{-1} = g^{-1} g = e$.


### Examples of groups

You already know many examples of groups. Here is some of them.

#### The set of integers with addition

The set of integers $\mathbb{Z}$ forms a group under addition $+$.
We write the group by $(\mathbb{Z}, +)$.
All the properties of a group are satisfied.
The set of integers is closed under addition, the addition is associative, the identity element is $0$, and the inverse of an integer $n$ is $-n$.

#### The set of non-zero rational numbers with multiplication

The set of non-zero rational numbers, $\mathbb{Q}\setminus \{0\}$, forms a group under multiplication $\times$.
Again, we write the group by $(\mathbb{Q}\setminus\{0\}, \times)$.
All the properties of a group are satisfied.
The set of rational numbers is closed under multiplication, the multiplication is associative, the identity element is $1$, and the inverse of a rational number $q$ is $1/q$.

#### The set of real numbers with addition

The set of real numbers $\mathbb{R}$ forms a group under addition $+$.
We write the group by $(\mathbb{R}, +)$.
All the properties of a group are satisfied in an obvious way. The identity element is $0$ and the inverse of a real number $x$ is $-x$.

#### The set of non-zero real numbers with multiplication

The set of non-zero real numbers, $\mathbb{R}\setminus \{0\}$, forms a group under multiplication $\times$.
Check the group properties for yourself.


#### The translation group

Consider the Euclidean space $\mathbb{R}^3$ with elementwise addition:

$$
(x_1,x_2,x_3) + (y_1,y_2,y_3) = (x_1+y_1, x_2+y_2, x_3+y_3).
$$

$(\mathbb{R}^3, +)$ forms a group. The identity element is $(0,0,0)$ and the inverse of $(x_1,x_2,x_3)$ is $(-x_1,-x_2,-x_3)$.
This is called the translation group denoted by $T(3)$.
It can be generalized to $n$ dimensions, $T(n)$.

#### The general linear group

The general linear group of dimension $n$, denoted by $GL(n,\mathbb{R})$, is the set of all invertible $n \times n$ matrices.
Mathematically,

$$
GL(n,\mathbb{R}) = \{ A \in \mathbb{R}^{n \times n} \mid \det(A) \neq 0 \}.
$$

The group operation is matrix multiplication. Check the group properties for yourself.
What is the identity element of $GL(n,\mathbb{R})$?

#### The special orthogonal group

The special orthogonal group of dimension $n$, denoted by $SO(n)$, is the set of all $n \times n$ orthogonal matrices with determinant one.
Mathematically,

$$
SO(n) = \{ A \in \mathbb{R}^{n \times n} \mid A A^T = I, \det(A) = 1 \}.
$$

The group operation is matrix multiplication.
You can think of $SO(n)$ as the group of all rotations in $n$ dimensions.
Check that the $SO(n)$ is indeed closed under matrix multiplication.

$SO(n)$ is a subgroup of $GL(n)$. We write:

$$
SO(n) \le GL(n).
$$

This means that $SO(n)$ is a group in its own right, and it is a subset of $GL(n)$.

Let's look at some examples of these groups in 3D space.
Rotation of degree $\theta$ about the $z$-axis is a member of $SO(3)$.
It corresponds to the matrix:

$$
R_1 = \begin{bmatrix}
\cos(\theta) & -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Check that $\det(R_1) = 1$.

Similarly, rotation of degree $\phi$ about the $y$-axis is:

$$
R_2 = \begin{bmatrix}
\cos(\phi) & 0 & \sin(\phi) \\
0 & 1 & 0 \\
-\sin(\phi) & 0 & \cos(\phi)
\end{bmatrix}.
$$

Check again that $\det(R_2) = 1$.

If you multiply the two, you get:

$$
R_3 = R_1 R_2 = \begin{bmatrix}
\cos(\theta)\cos(\phi) & -\sin(\theta) & \cos(\theta)\sin(\phi) \\
\sin(\theta)\cos(\phi) & \cos(\theta) & \sin(\theta)\sin(\phi) \\
-\sin(\phi) & 0 & \cos(\phi)
\end{bmatrix}.
$$

Check that $\det(R_3) = 1$.
Another thing to observe is that $R_2 R_1 \neq R_1 R_2$.
We say that matrix multiplication is *not commutative*.

#### The orthogonal group

The orthogonal group of dimension $n$, denoted by $O(n)$, is the set of all $n \times n$ orthogonal matrices.
Mathematically,

$$
O(n) = \{ A \in \mathbb{R}^{n \times n} \mid A A^T = I \}.
$$

Again, the group operation is matrix multiplicatin. 
You can think of $O(n)$ as the group of all transformations that preserve the length of vectors.
This includes rotations and reflections.
Check that $O(n)$ is closed under matrix multiplication.

$O(n)$ is a subgroup of $GL(n)$ and it contains $SO(n)$ as a subgroup.
We write:

$$
SO(n) \le O(n) \le GL(n).
$$

An element of the orthogonal group not in the special orthogonal group is a reflection, for example:

$$
R_4 = \begin{bmatrix}
-1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Check that $R_4R_4^T = I$.

## Group homorphisms

Two groups are homomorphic if there is a function between them that preserves the group structure.
This just means that the group elements of one group are relabeled versions of the group elements of the other group.
The function that does this is called a homomorphism.

Mathematically, let $G$ and $H$ be two groups.
If we can find a function $\phi: G \to H$ such that:

$$
\phi(g_1 g_2) = \phi(g_1) \phi(g_2),
$$

for all $g_1, g_2 \in G$, then $f$ is a homomorphism from $G$ to $H$.

If the homomorphism is bijective, i.e., one-to-one and onto, then it is called an isomorphism.
When $G$ and $H$ are isomorphic, we write $G \cong H$.

### Example: $\mathbb{Z}$ is homomorphic to $\mathbb{Z}_3$

Let's look at an example of an homorphism.
Consider the group of integers modulo $3$, denoted by $\mathbb{Z}_3$.
The group operation is addition modulo $3$.
For example, we have

$$
0 + 1 (\text{mod } 3) = 1, \quad 1 + 2 (\text{mod } 3) = 0.
$$

Note that $x (\text{mod } 3)$ is the remainder when $x$ is divided by $3$.
So, $\mathbb{Z}_3 = \{0,1,2\}$.

Now, consider the function

$$
f: \mathbb{Z} \to \mathbb{Z}_3,
$$

that gives the remainder of an integer is divided by $3$:

$$
f(x) = x (\text{mod } 3).
$$

This is an homomorphism. Check that:

$$
f(x + y) = (x + y) (\text{mod } 3) = x (\text{mod } 3) + y (\text{mod } 3)= f(x) + f(y).
$$

But it is not an isomorphism because it is not one-to-one.

### Example: The group of real numbers with addition is isomorphic to the group of positive real numbers with multiplication

Let $\mathbb{R}^+$ be the set of positive real numbers.
It is a group under multiplication. Why?

Consider the function:

$$
f: \mathbb{R} \to \mathbb{R}^+,
$$

defined by:

$$
f(x) = e^x.
$$

The function is one-to-one and onto.
Show that it is a homomorphism.

## Group of transformations

Let $V$ be a set.
A group of transformations of $V$ is a group, say $G$, of *bijections* from $V$ to $V$.
So, an element $g$ of $G$ is a function:

$$
g : V \to V,
$$

that is one-to-one and onto (bijective).
Here the group operation is the composition of functions and we are assuming that $G$ is closed under composition.
So, if $g_1$ and $g_2$ are in $G$, then the composition:

$$
g = g_1 \circ g_2,
$$

defined by:

$$
x\mapsto (g_1 \circ g_2)(x) = g_1(g_2(x)),
$$

is also in $G$.

Why is $G$ a group?
Well, first function composition is associative:

$$
(g_1 \circ g_2) \circ g_3 = g_1 \circ (g_2 \circ g_3).
$$

Second, the identity function $e$ is in $G$ (it is one-to-one and onto):

$$
e(x) = x,
$$

and it is the identity element of $G$.
Finally, the inverse of a function $g$ is also in $G$.

## Group representations

A group representation is a way to represent the elements of a group as matrices.
Using group representations you can study the group using linear algebra.

### Example: The group of invertible linear transformations is isomorphic to the general linear group

Let $V$ be a *real* vector space of dimension $n$ and let $GL(V)$ be the set of all invertible linear transformations of $V$, i.e.,

$$
GL(V) = \{ f: V \to V \mid f \text{ is invertible} \}.
$$

$GL(V)$ is a group under function composition.

We will show that it is isomorphic to the general linear group $GL(n)$.
Let $B = \{ \mathbf{e}_1, \ldots, \mathbf{e}_n \}$ be a basis of $V$.
Then, any linear transformation $f \in GL(V)$ can be represented by a matrix $A$ such that:

$$
f(\mathbf{e}_i) = \sum_{j=1}^n A_{ij} \mathbf{e}_j.
$$

The matrix $A$ is invertible because $f$ is invertible.

The map

$$
\phi : GL(V) \to GL(n),
$$

that sends a linear transformation to its matrix representation

$$
f \mapsto \phi(f) = A,
$$

is an isomorphism between $GL(V)$ and $GL(n)$. Why?
We can write:

$$
GL(V) \cong GL(n).
$$

## The Eucledian group

Consider the Euclidean space $\mathbb{R}^n$.
This is the space on which we write physical equations.
We will introduce the Euclidean group $E(n)$ which is a group of transformation of the Euclidean space that corresponds to changes of coordinates that, in many physical examples, do not change the form of the equations.
It is not the only such group, but it is an easy example to start with.

The Euclidean group $E(n)$ is the group of all isometries of $\mathbb{R}^n$.
To explain this, we need to introduce the concept of *affine* transformations.
An affine transformation is a linear transformation followed by a translation.
So, $f: \mathbb{R}^n \to \mathbb{R}^n$ is an affine transformation if:

$$
f(\mathbf{x}) = g(\mathbf{x}) + \mathbf{b},
$$

where $g: \mathbb{R}^n \to \mathbb{R}^n$ is a linear transformation and $\mathbf{b} \in \mathbb{R}^n$ is a translation vector.
An isometry is an affine transformation that preserves distances.
This means that:

$$
\| f(\mathbf{x}) - f(\mathbf{y}) \| = \| \mathbf{x} - \mathbf{y} \|,
$$

for all $\mathbf{x}, \mathbf{y}$ in $\mathbb{R}^n$.

Now, we can define the Euclidean group $E(n)$ as:

$$
E(n) = \{ f: \mathbb{R}^n \to \mathbb{R}^n \mid f \text{ is an isometry} \}.
$$

Again, the group operation is function composition.
We can show that $E(n)$ can be written as some sort of Cartesian product of the translation group $T(n)$ and the orthogonal group $O(n)$:

$$
E(n) \cong T(n) \rtimes O(n).
$$

Here the symbol $\rtimes$ stands for a *semidirect product*.
We say that $E(n)$ is a semidirect product of product of $O(n)$ extended by $T(n)$.
The semidirect product gives us a way to reduce the group structure of $E(n)$ to the group structures of $T(n)$ and $O(n)$ and to do everything in terms of linear algebra.

Intuitively, the semidirect product means that the group $E(n)$ is a combination of the translation group and the orthogonal group.
So, every element of $E(n)$ can be written as a translation followed by a rotation/reflection, i.e., as $(\mathbf{b}, A)$, where $\mathbf{b}$ is a translation vector and $A$ is an orthogonal matrix.
You can apply that transformation to any vector $\mathbf{x}$ of $\mathbb{R}^n$ by:

$$
(\mathbf{b}, A) \mathbf{x} = A (\mathbf{x} + \mathbf{b}).
$$

The semidirect product specifies how the translation and the rotation/reflection interact.
Take two elements $(\mathbf{b}_1, A_1)$ and $(\mathbf{b}_2, A_2)$ of $E(n)$.
Their composition is:

$$
(\mathbf{b}_1, A_1)\circ (\mathbf{b}_2, A_2) = (\mathbf{b}_1 + A_1 \mathbf{b}_2, A_1 A_2).
$$

It is the semidirect product that specifies how the two group operations interact.

Understanding the semidirect product is not trivial.
We do it in the next section, but feel free to skip it if you are not interested.

## Semidirect products

*You can skip this section if you are not interested in the details of the semidirect product.*

Let $G$ be a group and $H$ and $K$ be two subgroups of $G$, i.e.,

$$
H \le G, \quad K \le G.
$$

Furthermore, we assume that $H$ and $K$ only share the identity element, i.e.,

$$
H \cap K = \{ e \},
$$

and that elements of $G$ can be *uniquely* written as a product of elements of $H$ and $K$, i.e.,

$$
G = HK = \{ hk \mid h \in H, k \in K \quad\text{ in a unique way} \}.
$$

We are going to assume that $H$ is a *normal subgroup* of $G$, i.e.,
for all $h \in H$ and $g \in G$, we have:

$$
g h g^{-1} \in H.
$$

This is a wierd assumption, but it is necessary for the semidirect product to work.
We will show that under these assumptions the group $G$ can be written as:

$$
G \cong H \rtimes K.
$$

The meaning of everything will become apparent as we go.

First, $G\cong H \rtimes K$ means that $G$ means that there is a bijection from $H\times K$ to $G$ that preserves the group structure.
Let's make this bijection explicit.
We define:

$$
\phi : H \times K \to G,
$$

such that:

$$
(h, k) \mapsto \phi(h,k) = hk.
$$

This map is indeed onto because every element of $G$ can be written as a product of an element of $H$ and an element of $K$.
It is also one-to-one.
Suppose that:

$$
hk = h'k',
$$

for some $h,h' \in H$ and $k,k' \in K$.
Then, multiplying by $k^{-1}$ on the right, we get:

$$
h = h'k'k^{-1}.
$$

Multiplying with $(h')^{-1}$ on the left, we get:

$$
h(h')^{-1} = k'k^{-1}.
$$

Now, the left hand side is an element of $H$ and the right hand side is an element of $K$.
But $H$ and $K$ only share the identity element.
So, $h = h'$ and $k = k'$.
This shows that the map is one-to-one.

To show that the map preserves the group structure, we need to explicitly define the group operation on $H\times K$.
This is where the $\rtimes$ comes in.
It fixes the way the group operations of $H$ and $K$ interact.
We will construct the group operation so that the map $\phi$ preserves the group structure.
This is what we want to achieve:

$$
\phi((h_1,k_1)(h_2,k_2)) = \phi(h_1,k_1)\phi(h_2,k_2).
$$

On the right hand side, we just use the definition of the map:

$$
\phi((h_1,k_1)(h_2,k_2)) = \phi(h_1,k_1)\phi(h_2,k_2) = h_1k_1h_2k_2.
$$

Now, we try to turn the result into something that looks like a product of an $H$ with a $K$.
Let's just introduce a $k_1^{-1}k_1 = e$ in the middle:

$$
\phi((h_1,k_1)(h_2,k_2)) =  h_1k_1h_2k_2 = h_1k_1h_2{\textcolor{red}e} k_2 = h_1k_1h_2\textcolor{red}{k_1^{-1}}k_1k_2 = (h_1\textcolor{blue}{k_1h_2k_1^{-1}})(k_1k_2).
$$

Clearly, the term in the secon parenthesis is in $K$.
What about the term in the first parenthesis?
It is the product of an element of $H$ and the wierd blue term $\textcolor{blue}{k_1h_2k_1^{-1}}$.
This is where the normality of $H$ comes in.
$k_1$ is in $K$ which is a subgroup of $G$, so it is also in $G$.
$h_2$ is in $H$ which is a normal subgroup of $G$.
So, $k_1h_2k_1^{-1}$ is in $H$.
This means that the product of an element of $H$ and the blue term is in $H$.
From this, we see that we are forced to choose the following definition for the group operation on $H\times K$:

$$
(h_1,k_1)(h_2,k_2) = (h_1k_1h_2k_1^{-1}, k_1k_2).
$$

### Connection to the Euclidean group

This is all too theoretical.
How does this connect to the Euclidean group?
Take:

$$
G = E(n), \quad H = T(n), \quad K = O(n).
$$

First, we do have that $T(n)$ and $O(n)$ are subgroups of $E(n)$.
This is obvious.
Let's prove all the other things we need.

We start by proving that $E(n) = T(n)O(n)$.
Let $f$ be an element of $E(n)$.
Consider the vector to which $f$ maps the origin:

$$
\mathbf{b} = f(\mathbf{0}).
$$

Now, define the translation $t_{\mathbf{b}}$ by:

$$
t_{\mathbf{b}}(\mathbf{x}) = \mathbf{x} + \mathbf{b}.
$$

This is an element of $T(n)$.
Finally, define the function $g$ by:

$$
g = t_{\mathbf{b}}^{-1} f.
$$

Or in terms of its action on a vector $\mathbf{x}$:

$$
g(\mathbf{x}) = t_{\mathbf{b}}^{-1} f(\mathbf{x}) = \mathbf{x} = f(x) - \mathbf{b}.
$$

We will show that $g$ is an element of $O(n)$.
$g$ is obviously linear and an isometry.
All, we need to show is that it keeps the origin fixed.
Indeed, by construction:

$$
g(\mathbf{0}) = f(\mathbf{0}) - \mathbf{b} = \mathbf{b} - \mathbf{b} = \mathbf{0}.
$$

Try to show that the decomposition is unique.

Now, we need to show that $T(n)$ is a normal subgroup of $E(n)$.
Let $f$ be an element of $E(n)$ and $t_{\mathbf{b}}$ be an element of $T(n)$.
The latter is such that:

$$
t_{\mathbf{b}}(\mathbf{x}) = \mathbf{x} + \mathbf{b}.
$$

Now, consider the composition:

$$
h = f t_{\mathbf{b}} f^{-1}.
$$

We need to show that this is an translation, i.e., it is in $T(n)$.
Where does an arbitrary vector $\mathbf{x}$ go under this map?
We have:

$$
h(\mathbf{x}) = f t_{\mathbf{b}} f^{-1}(\mathbf{x}) = f(f^{-1}(\mathbf{x}) + \mathbf{b}) = \mathbf{x} + f(\mathbf{b}) = t_{f(\mathbf{b})}(\mathbf{x}).
$$

This shows that $h$ is a translation:

$$
h = f t_{\mathbf{b}} f^{-1} = t_{f(\mathbf{b})},
$$

which proves the desired result.

Finally, we need to show that $T(n) \cap O(n) = \{ e \}$, where $e$ is the identity element of $E(n)$, i.e.,

$$
e(\mathbf{x}) = \mathbf{x}.
$$

This is obvious. Why?

Having proved all these things, we can use the result above to write:

$$
E(n) \cong T(n) \rtimes O(n).
$$

Let's write down the group operation explicitly:

$$
(t_{\mathbf{b}_1}, A_1)(t_{\mathbf{b}_2}, A_2) = (t_{\mathbf{b}_1}A_1t_{\mathbf{b}_2}A_1^{-1}, A_1A_2).
$$

We can simplify the first term of the right hand side:

$$
t_{\mathbf{b}_1}A_1t_{\mathbf{b}_2}A_1^{-1}(\mathbf{x}) = t_{\mathbf{b}_1}A_1(A_1^{-1}\mathbf{x} + \mathbf{b}_2) = A_1A_1^{-1}\mathbf{x} + A_1\mathbf{b}_2 + \mathbf{b}_1 = \mathbf{x} + A_1\mathbf{b}_2 + \mathbf{b}_1.
$$

If we identify $\mathbf{b}_1$ with the translation vector $\mathbf{b}_1$ and $A_1$ with the rotation/reflection matrix $A_1$, we get exactly what we wrote in the previous section.
By the way, this *identification* is another isomorphism.

## Invariance

Now that we know what symmetries are, we can talk about invariance.
Consider a function $f$ from a vector space $V$ to the real numbers.
We say that $f$ is invariant under a group of transformations $G$ if:

$$
f(g(\mathbf{x})) = f(\mathbf{x}),
$$

for all $\mathbf{x}$ in $V$ and all $g$ in $G$.

If you have a physical problem with a known symmetry like that, you better construct a model that respects that symmetry.

If the symmetry group is finite, then there is an easy way to construct an invariant function from an arbitrary function.
Say $h_\theta$ is an arbitrary real function of $V$ parameterized by $\theta$.
Define the function:

$$
f_\theta(\mathbf{x}) = \sum_{g \in G} h_\theta(g(\mathbf{x})).
$$

Show that $f_\theta$ is invariant under $G$.

When $G$ is a continuous group, the sum above becomes an integral:

$$
f_\theta(\mathbf{x}) = \int_G h_\theta(g(\mathbf{x})) dg.
$$

But integrating over a continuous group is not trivial. You get into the theory of Lie groups. It is also very likely that you will not be able to find an explicit expression for $f_\theta$.

## Equivariance

Let $f$ be a function from a vector space $V$ to another vector space $W$.
Let $G$ be a group of transformations that can act both on $V$ and $W$.
Suppose that you know that $f$ changes in a predictable way under the action of $G$.
We can write this as:

$$
f(g(\mathbf{x})) = g(f(\mathbf{x})).
$$

When this happens, we say that $f$ is equivariant under $G$.

In the equation above, on the left hand-side $g$ acts on $V$ and on the right hand-side $g$ acts on $W$.
These actions can be very different.
Let me give you a specific example.
Suppose that $V$ is the set of 3D coordinates of a molecule with $n$ atoms.
We can write:

$$
V = \mathbb{R}^{3n},
$$

or for an $\mathbf{x}$ in $V$:

$$
x = (\mathbf{r}_1,\dots,\mathbf{r}_n),
$$

where $\mathbf{r}_i$ is the position of the $i$-th atom.

Now, suppose that $f$ gives us the force acting on each atom of the molecule.
Again, we have:

$$
W = \mathbb{R}^{3n},
$$

and for an $\mathbf{f}$ in $W$:

$$
f = (\mathbf{f}_1,\dots,\mathbf{f}_n),
$$

where $\mathbf{f}_i$ is the force acting on the $i$-th atom.

Now, take $G$ to be the Euclidean group $E(3)$.
An element $g = (\mathbf{b}, A)$ of $E(3)$ acts on $V$ by rotating and translating each atom of the molecule, i.e., by:

$$
g\mathbf{x} = (A\mathbf{r}_1 + \mathbf{b}, \dots, A\mathbf{r}_n + \mathbf{b}).
$$

How do the forces change under the action of $G$?
The translation part of $g$ does not change the forces.
But the rotation part does.
The force acting on the $i$-th atom changes by:

$$
g\mathbf{f}_i = A\mathbf{f}_i.
$$

## Eucledian neural networks

Eucledian neural networks, see [(Geiger et al. 2022)](https://arxiv.org/abs/2207.09453), are neural networks that respect the symmetries of the Euclidean group.
They rely on spherical harmonics to construct invariant and covariant functions.
More details in their paper.
We are going to demonstrate what they are capable of using numerical examples.