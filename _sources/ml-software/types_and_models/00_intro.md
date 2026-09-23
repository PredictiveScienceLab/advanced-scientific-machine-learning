# Type Systems, Pytrees, and Models

Functional transformations operate reliably only when the inputs and outputs of a computation have a well-defined structure. Scientific machine learning models add nested parameter collections, so shape, type, and tree structure become part of the model interface.

The treatment assumes familiarity with pure functions, JAX transformations, higher-order functions, and ordinary Python containers. The required type notation is developed locally.

We begin with the role of types in compiled JAX programs and introduce Python type annotations. Compact Haskell-style signatures help us read what transformations such as `vmap` and `grad` accept and return; pytrees organize nested model parameters. The companion develops these ideas further, including models built with the Equinox library. These tools make model interfaces explicit and allow JAX transformations to act on complex parameter structures as coherent mathematical objects.
