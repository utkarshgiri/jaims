# jaims

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A jax based affine-invariant MCMC sampler that can leverage GPUs to speed up sampling for computationally intensive likelihoods. It implements the [Goodman-Weare](https://msp.org/camcos/2010/5-1/p04.xhtml) algorithm as described in [dfm++](https://arxiv.org/abs/1202.3665) and is inspired by the popular [`emcee`](https://github.com/dfm/emcee) library. The `just-in-time` compilation together with `vectorized` likelihood evaluation for the walkers gives significant speed-up even on CPUs when compared to emcee

## Installation

To install `jaims`, please clone this repository and then run `python setup.py install` inside it \
You can also install this via `pip` using
```
pip install jaims
```
To run it on a GPU, you must have an installation of `jaxlib` compatible with your CUDA version. For more information, please refer to the official [guidelines](https://github.com/google/jax#installation) 

The API for `jaims` is slightly different from `emcee`. This might change in the future.
