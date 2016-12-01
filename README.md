# HiddenMarkovChains — a Julia library

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)

[![Build Status](https://travis-ci.org/tpapp/HiddenMarkovChains.jl.svg?branch=master)](https://travis-ci.org/tpapp/HiddenMarkovChains.jl)

[![Coverage Status](https://coveralls.io/repos/tpapp/HiddenMarkovChains.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tpapp/HiddenMarkovChains.jl?branch=master)

[![codecov.io](http://codecov.io/github/tpapp/HiddenMarkovChains.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/HiddenMarkovChains.jl?branch=master)

This is a preliminary collection of functions I use for calculations that involve Hidden Markov Chains (HMC). It is mainly used for [indirect inference](http://www.econ.yale.edu/smith/palgrave7.pdf) for macroeconomic models, in which the state space is discretized. Exact calculation of path probabilities allows smooth functions of model parameters, obviating the need for [explicit smoothing](http://arxiv.org/abs/1507.06115) (of course discretization brings in another set of problems).


## Bibliography

*Golub, Gene H., and Carl D. Meyer, Jr.* "Using the QR factorization and group inversion to compute, differentiate, and estimate the sensitivity of stationary probabilities for Markov chains." SIAM Journal on Algebraic Discrete Methods 7.2 (1986): 273-281.

*Rabiner, Lawrence R.* "A tutorial on hidden Markov models and selected applications in speech recognition." Proceedings of the IEEE 77.2 (1989): 257-286.
