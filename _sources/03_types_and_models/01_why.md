# Typing Systems and why We Care

Python has dynamic typing, which means that the type of a variable is determined by the value it holds. This is in contrast to static typing, where the type of a variable is determined by its declaration. In a statically typed language, a variable can only be assigned a value of a type compatible with its declaration. In a dynamically typed language, a variable can be assigned a value of any type.

Dynamic typing is a double-edged sword. On the one hand, it makes programming easier and more flexible. On the other hand, it makes programming more error-prone. For example, if you have a variable that is supposed to hold a number, but you accidentally assign it a string, you will not get an error until you try to use the variable as a number. This can be a problem if the variable is used in many places, because you will have to check all of them to find the source of the error.

Despite the fact that Python is a dynamically typed language, it is possible to add type annotations to Python code. These annotations are not enforced by the Python interpreter, but they can be used by external tools to check the type consistency of your code. This is called static type checking, and it can help you find errors in your code before you run it.
You will see these annotations in the code of many Python packages. You will also see them in the code of many scientific machine learning packages. These annotations look like this:
    
```python
def f(x: float) -> float:
    return x + 1.0
```

So, we will explain Python type annotations in this chapter.

As you go through modern scientific machine learning packages, like `Jax`, you will also note that they explain some of their functions using Haskell-like type signatures! We will explain these as well because they are so ubiquitous.
These look like this:

```haskell
f :: a -> b -> c
```

Finally, you will also see the term `Pytree` used in `Jax` documentation. We will explain what this is and why it is important.

Finally, you will also see the term `Pytree` used in `Jax` documentation. We will explain what this is and why it is important.

You can write Python code without ever thinking about types. However, if you want to use `Jax`, you will have to learn about types. This is because `Jax` just-in-time only works if the types of all the variables in your code are known and remain constant. `Jax` takes your code, turns into a low level representation that is statically typed, and then compiles it. This is why you need to know about types.
