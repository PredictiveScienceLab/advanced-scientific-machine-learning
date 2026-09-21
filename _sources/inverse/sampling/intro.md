# Sampling from Posteriors

Once an inverse problem has been posed in Bayesian form, the central computational task is to characterize the posterior distribution. In realistic models that posterior is not available analytically, so we approximate it with samples.

This section studies Markov chain Monte Carlo as a tool for exploring posterior geometry beyond one optimum. The resulting samples approximate the joint posterior distribution of the unknown quantities, quantify uncertainty, and reveal parameter correlations, multimodality, and other pathologies that deterministic optimization can miss.
