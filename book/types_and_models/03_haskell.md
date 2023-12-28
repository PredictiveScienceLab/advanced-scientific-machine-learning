# Haskell Type System

[Haskell](https://www.haskell.org/) is a pure functional programming language with a strong static type system. This means that all functions are pure and all types are known at compile time. This is in contrast to Python, which is an impure functional programming language with a weak dynamic type system. This means that functions can be impure and types are not known until runtime.

Haskell started out as a research project in the 1980s.
It has becoming increasingly popular, but not because it is a practical programming language.
It is popular because it is a great language for learning about functional programming and type systems.
It has a very simple syntax and a beautiful type system.
This is why many of `Jax` authors use Haskell types in their documentation.

## Primitive Types

There are some basic Haskell types that you need to know about.
The most basic type is the type `Int`, which is the type of integers.
The type `Int` is a primitive type, which means that it is built into the language.
The type `Int` is a concrete type, which means that it is not parameterized with any other type.
Another type is the type `Bool`, which is the type of booleans. It is also primitive and concrete.
Other primitive and concrete types are `Char`, `Double`, and `Float`.

## Composite Types

Lists are a *generic* or *parametric* type, which means that they are parameterized with another type.
Their type is written as `[T]`, where `T` is a type variable that can be replaced with any type.
For example, the type `[Int]` is the type of lists of integers.
`[Float]` is the type of lists of floats.
`[Bool]` is the type of lists of booleans.
`[Char]` is the type of lists of characters.
And so on.

Lists of lists are written as `[[T]]`, where `T` is a type variable that can be replaced with any type.
And lists of lists of lists are written as `[[[T]]]`, and so on.

## Function Types

The type of functions is written as `T -> U`, where `T` and `U` are type variables that can be replaced with any type.
For example a function that takes an integer and returns an integer has the type signature:

```haskell
f:: Int -> Int
```

A type that takes a list of floats and returns a float has the type signature:

```haskell
g: [Float] -> Float
```

You can also have a function that works with a list of any type and return a result of the same type:

```haskell
h:: [T] -> T
```

This is called a *polymorphic* function, which means that it can work with multiple types.

## Curried Functions

You can also have a function that takes two arguments of the same type and returns a result of the same type.
Think of it as `f(x, y)` in Python. It has the type signature:

```haskell
f:: T -> T -> T
```

Why this notation? You can think of the function `f` as a function that takes one argument of type `T` and returns a function that takes one argument of type `T` and returns a result of type `T`.
Like this:

+ Take and `x` of type `T`.
+ Plug `x` into `f(x, .)` and get a function `g:: T -> T` defined by `g(y) = f(x, y)`.

This is called a *curried* function, which means that it takes multiple arguments one at a time.
The name comes from [Haskell Curry](https://en.wikipedia.org/wiki/Haskell_Curry), who was a logician and mathematician.

Another similar example is a function that takes two arguments of different types and returns a result of the same type as the second argument.

```haskell
f:: T -> U -> U
```

Same deal. Think of the function `f` as a function that takes one argument of type `T` and returns a function that takes one argument of type `U` and returns a result of type `U`.
So `f(x, .)` is a function that takes one argument of type `U` and returns a result of type `U`.

Now take a function with three arguments of different types, returning a result that is the same type as the third argument.

```haskell
f:: T -> U -> V -> V
```

Do you get it? The function `f` takes one argument of type `T` and returns a function that takes one argument of type `U` and returns a function that takes one argument of type `V` and returns a result of type `V`.

## Types of higher-order functions

### Type signature of `map`

For example, the function {ref}`map` which takes a function of one argument, and a list, and returns a list containing the result of applying the function to each element of the list.
Here is its type signature:

```haskell
map:: (T -> U) -> [T] -> [U]
```

Let's break it down:

+ It takes one argument of type `T -> U`. This is a function that takes an argument of type `T` and returns a result of type `U`.
+ It takes another argument of type `[T]`. This is a list of values of type `T`.
+ It returns a result of type `[U]`. This is a list of values of type `U`. It has to be a `U` because the function we provide as an argument returns a `U`.

### Type signature of `filter`

Another example is the {ref}`filter` function, which takes a function of one argument, and a list, and returns a list containing the elements of the list for which the function returns `True`.
Here is its type signature:

```haskell
filter:: (T -> Bool) -> [T] -> [T]
```

### Type of function composition

Another example is the {ref}`function-composition` operator, which takes two functions and returns a function that is the composition of the two functions.
Its type signature is:

```haskell
compose:: (U -> V) -> (T -> U) -> (T -> V)
```

Meditate on this for a while:

+ It takes one argument of type `U -> V`.
+ It takes another argument of type `T -> U`.
+ It returns a result of type `T -> V`.


## Types of `Jax` functions

In `Jax` we work with arrays instead of lists.
The type of the elements of an array is called the *dtype* of the array.
It can be a primitive type, such as `int` or `float`, or a composite type, such as `float32` or `float64`.
In the `Jax` documentation, they write `a`, `b`, etc., for the primitive types, and `[a]`, `[b]`, etc., for the array types.

### Type signature of `vmap`

Recall that the {vmap}`vmap` function takes a function of one argument, and returns a function that can be applied to an array.
In its simplest form, the type signature of `vmap` is:

```haskell
vmap:: (a -> b) -> ([a] -> [b])
```

Let's break it down:

+ It takes one argument of type `a -> b`. This is a function that takes an argument of type `a` and returns a result of type `b`.
+ It takes another argument of type `[a]`. This is a list of values of type `a`.

### Signature of `fori_loop`

Many of the functions in `jax.lax` are documented with Haskell type signatures.
The {fori_loop}`fori_loop`, called as `fori_loop(lower, upper, body_fun, init_val)`, is essentially doing something equivalent this:

```python
def fori_loop(lower, upper, body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val
```

The signature of `fori_loop` is:

```haskell
fori_loop:: Int -> Int -> ((Int, a) -> a) -> a -> a
```

So:

+ It takes one argument of type `Int`. This is the lower bound of the loop.
+ It takes another argument of type `Int`. This is the upper bound of the loop.
+ It takes another argument of type `(Int, a) -> a`. This is a function that takes a tuple of type `(Int, a)` and returns a result of type `a`. The integer is the loop index and the `a` is the accumulator.
+ It takes another argument of type `a`. This is the initial value of the accumulator.
+ It returns a result of type `a`. This is the final value of the accumulator.

### Signature of `grad`

The {grad}`grad` function takes a function of one argument, and returns a function that computes the gradient of the function.
In its simplest form, the type signature of `grad` is:

```haskell
grad:: ([a] -> a) -> ([a] -> [a])
```

Let's break it down:

+ It takes one argument of type `[a] -> a`. This is the function we want to differentiate. It is a scalar function.
+ It returns a result of type `[a] -> [a]`. This is the function that computes the gradient of the function. It is a vector function. This is the gradient with respect to the input argument.

### Signature of `jacobian`

The {jacobian}`jacobian` function takes a function of one argument, and returns a function that computes the Jacobian of the function.
In its simplest form, the type signature of `jacobian` is:

```haskell
jacobian:: ([a] -> [a]) -> ([a] -> [[a]])
```

So:
+ It takes one argument of type `[a] -> [a]`. This is the function we want to differentiate. It is a vector function.
+ It returns a result of type `[a] -> [[a]]`. This is the function that computes the Jacobian of the function. It is a matrix function. This is the Jacobian with respect to the input argument.


### Signature of `hessian`

The {hessian}`hessian` function takes a function of one argument, and returns a function that computes the Hessian of the function.

In its simplest form, the type signature of `hessian` is:

```haskell
hessian:: ([a] -> a) -> ([a] -> [[a]])
```

So:

+ It takes one argument of type `[a] -> a`. This is the function we want to differentiate. It is a scalar function.
+ It returns a result of type `[a] -> [[a]]`. This is the function that computes the Hessian of the function. It is a matrix function. This is the Hessian with respect to the input argument.


### Signature of `jvp`

The {jvp}`jvp` function takes a function of one argument, and returns a function that computes the Jacobian-vector product of the function.
In its simplest form, the type signature of `jvp` is:

```haskell
jvp:: ([a] -> a) -> [a] -> [a] -> [a]
```