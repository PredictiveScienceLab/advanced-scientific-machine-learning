# Types and JAX

Python has dynamic typing, which means that the type of a variable is determined by the value it holds. This is in contrast to static typing, where the type of a variable is determined by its declaration. In a statically typed language, a variable can only be assigned a value of a type compatible with its declaration. In a dynamically typed language, a variable can be assigned a value of any type.

Dynamic typing is a double-edged sword. On the one hand, it makes programming easier and more flexible. On the other hand, it makes programming more error-prone. For example, if you have a variable that is supposed to hold a number, but you accidentally assign it a string, you will not get an error until you try to use the variable as a number. This can be a problem if the variable is used in many places, because you will have to check all of them to find the source of the error.

Python also supports type annotations. The interpreter does not enforce them, but external static-type checkers can use them to find inconsistencies before the code runs. Many scientific Python packages use annotations such as this one:
    
```python
def f(x: float) -> float:
    return x + 1.0
```

JAX documentation also uses compact Haskell-like signatures to describe functions. They look like this:

```haskell
f :: a -> b -> c
```

Here `a`, `b`, and `c` denote types. The arrows associate to the right: `f` takes an input of type `a` and returns a function from `b` to `c`. We can supply the two inputs one at a time.

JAX calls a nested structure of containers and array-valued leaves a *pytree*. A model's parameters and state are often stored in this form.

JAX traces a function for particular input shapes and data types, lowers the computation to a statically typed representation, and then compiles it. Type signatures and pytree structure therefore help us read what a transformed JAX function accepts and returns.
