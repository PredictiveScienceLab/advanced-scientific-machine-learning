# Embedding Symmetries in Surrogate Models

Known physical symmetries restrict a model to input-output relationships that transform consistently when the physical system is translated, rotated, or reflected. Encoding these symmetries reduces what the model must learn from data.

The treatment assumes linear algebra, function composition, and basic neural networks. It develops the required group-theoretic language from elementary examples before applying it to rigid transformations of Euclidean space.

We begin with groups, homomorphisms, group actions, and representations, then assemble translations and orthogonal transformations into the Euclidean group. The resulting actions define transformation laws, invariance, and equivariance for scalar, vector, and tensor quantities. A Euclidean neural-network example implements and tests these constraints on a three-dimensional point-cloud classification problem. By enforcing known transformation laws by construction, symmetry-aware surrogates focus learning on the behavior that those laws do not already determine.
