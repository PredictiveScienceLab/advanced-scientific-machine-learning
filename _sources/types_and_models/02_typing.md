# Python Type Annotations

The basic syntax for annotating a variable is to add a colon after the variable name, followed by the type of the variable. For example, the following code declares a variable `x` of type `int`:

```python
x: int
```

From that point on it gets more complicated. For example, the following code declares a variable `y` of type `List[int]`:

```python
y: List[int]
```

The type `List[int]` is a type annotation for a list of integers. The type `List` is a generic type, which means that it can be parameterized with a type. In this case, the type `List` is parameterized with the type `int`. The type `List[int]` is a shorthand for `List[T]`, where `T` is a type variable that can be replaced with any type. In this case, `T` is replaced with `int`.

To make a list of lists of integers, you can use the type `List[List[int]]`. To make a list of lists of lists of integers, you can use the type `List[List[List[int]]]`. And so on.

The type `List` is defined in the `typing` module, which is part of the Python standard library. The `typing` module also defines other generic types, such as `Dict`, `Set`, `Tuple`, `Optional`, `Union`, `Callable`, `Iterable`, `Iterator`, `Sequence`, `Mapping`, `Any`, `TypeVar`, and `Generic`. We will explain some of these types later in this chapter.

Now, let's consider a function that takes two integers and returns their sum:

```python
def add(x: int, y: int) -> int:
    return x + y
```

The type annotation `-> int` indicates that the function returns an integer. The type annotation `int` indicates that the function takes two integers as arguments. The type annotation `x: int` indicates that the variable `x` is of type `int`. The type annotation `y: int` indicates that the variable `y` is of type `int`.

Now consider a higher-order function that takes a function as an argument:

```python
def twice(f: Callable[[int], int], x: int) -> int:
    return f(f(x))
```

The type annotation `f: Callable[[int], int]` indicates that the variable `f` is of type `Callable[[int], int]`. The type `Callable[[int], int]` is a generic type that is parameterized with two types: the type of the function's arguments and the type of the function's return value. In this case, the type `Callable[[int], int]` is parameterized with the type `int` for both the arguments and the return value. This means that the function `f` takes an integer as an argument and returns an integer.