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

$SO(n)$ is a subgroup of $GL(n)$.
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
If we can find a function $f: G \to H$ such that:

$$
f(g_1 g_2) = f(g_1) f(g_2),
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
T : GL(V) \to GL(n),
$$

that sends a linear transformation to its matrix representation

$$
f \mapsto T(f) = A,
$$

is an isomorphism between $GL(V)$ and $GL(n)$. Why?
We can write:

$$
GL(V) \cong GL(n).
$$

### Example: The group of rotations and reflections

Suppose that $V$ is $\mathbb{R}^3$.
The group of rotations and reflections of the vectors in $V$ is a group of transformations of $V$.
It is isomorphic to the orthogonal group $O(3)$.

### Example: The group of translations

Again, suppose that $V$ is $\mathbb{R}^3$.
The group of translations of the vectors in $V$ is a group of transformations of $V$.
It is isomorphic to the translation group $T(3)$.

*This section is under construction.*