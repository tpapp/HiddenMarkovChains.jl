# HiddenMarkovChains — a Julia library

This is a preliminary collection of functions I use for calculations that involve Hidden Markov Chains (HMC).

Recommended usage:
```julia
import HiddenMarkovChains            # import instead of using
const HMC = HiddenMarkovChains       # use a shorter alias

# given transition matrix P and observation probabilities Q ...
p = HMC.steadystate(P)
path_probs = HMC.path_probabilities(P, Q, p, 4)
```
where the last line will return the probability of each *observed* path of length 4, as a `Dict`.

## Bibliography

*Golub, Gene H., and Carl D. Meyer, Jr.* "Using the QR factorization and group inversion to compute, differentiate, and estimate the sensitivity of stationary probabilities for Markov chains." SIAM Journal on Algebraic Discrete Methods 7.2 (1986): 273-281.
