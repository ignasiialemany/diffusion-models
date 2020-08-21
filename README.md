# Diffusion Models

Analytical and random walk models of diffusion in permeable layered media.

## Installation

It is recommended to use a virtual environment for Python, such as Anaconda.
First, install the external packages via `pip install -r requirements.txt`.
Then, install the code in this repository using `pip install -e .`.
The `-e` flag only adds the package path to Python and leaves the source files editable.

The chebpy dependency is currently set to a [fork](https://github.com/janniklasrose/chebpy).
This will be changed back to the original repository once the relevant merge requests have been accepted.

## Analytical solution

A semi-analytical solution was first proposed by [Moutal & Grebenkov](https://doi.org/10.1007/s10915-019-01055-5).
We present an improved algorithm to find the eigenvalues (roots of the transcendental equation), which is fast and robust.
The original algorithm is dependent on several parameters and often leads to missed roots.
This new approach recursively fits Chebyshev polynomials, which are represented by [chebpy](https://github.com/chebpy/chebpy) (similar to [Chebfun](http://chebfun.org)).

## Monte Carlo random walk

For higher-dimensional (2D and 3D) cases, there exists no analytical solution to the diffusion problem.
Monte Carlo random walk simulations are commonly used in these situations as they are fast and versatile.
However, care must be taken to handle discontinuities correctly.
We provide a random walk code to investigate different membrane permeability models by comparing the results to the analytical solution.

## Problem scaling

Working in SI units is usually encouraged for consistency.
However, given the physical scales of the modelled geometries and processes, it is wise to find a scaling that ensures parameter values are near unity.
This also helps to reduce numerical error.
The following table illustrates a consistent choice for units.

|   Parameter  |  Symbol |        Unit       |      Scale      |
| ------------ |:-------:| ----------------- |:---------------:|
| Length       | `L`/`x` | µm                | 10<sup>-6</sup> |
| Time         |   `t`   | ms                | 10<sup>-3</sup> |
| Diffusivity  |   `D`   | µm<sup>2</sup>/ms | 10<sup>-9</sup> |
| Permeability |   `K`   | µm/ms             | 10<sup>-3</sup> |
