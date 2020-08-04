# Diffusion Models

Analytical and random walk models of diffusion in permeable layered media.

## Installation

It is recommended to use a virtual environment for Python, such as Anaconda. First, install the external packages via `pip install -r requirements.txt`. Then, install the code in this repository using `pip install -e .`. The `-e` flag only adds the package path to Python and leaves the source files editable.

## Problem scaling

Working in SI units is usually encouraged for consistency. However, given the physical scales of the modelled geometries and processes, it is wise to find a scaling that ensures parameter values are near unity. This also helps to reduce numerical error. The following table illustrates a consistent choice for units.

|   Parameter  |  Symbol |        Unit       |      Scale      |
| ------------ |:-------:| ----------------- |:---------------:|
| Length       | `L`/`x` | µm                | 10<sup>-6</sup> |
| Time         |   `t`   | ms                | 10<sup>-3</sup> |
| Diffusivity  |   `D`   | µm<sup>2</sup>/ms | 10<sup>-9</sup> |
| Permeability |   `K`   | µm/ms             | 10<sup>-3</sup> |
