[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/wetai/)
[![tests](https://github.com/Ciela-Institute/caustic/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Ciela-Institute/caustic/actions)
[![Docs](https://github.com/Ciela-Institute/caustic/actions/workflows/documentation.yaml/badge.svg)](https://github.com/Ciela-Institute/caustic/actions/workflows/documentation.yaml)
[![PyPI version](https://badge.fury.io/py/caustic.svg)](https://pypi.org/project/caustic/)
[![coverage](https://img.shields.io/codecov/c/github/Ciela-Institute/caustic)](https://app.codecov.io/gh/Ciela-Institute/caustic)

# caustics

## Getting Started

Welcome to the lensing pipeline of the future: GPU-accelerated, automatically-differentiable,
highly modular. Currently under heavy development: expect interface changes and
some imprecise/untested calculations.

## Installation 

Simply install caustics from PyPI:
```bash
pip install caustics
```

## Documentation

Please see our [documentation page](Ciela-Institute.github.io/caustic/) for more detailed information.

## Contributing

Please reach out to one of us if you're interested in contributing!

To start, follow the installation instructions, replacing the last line with
```bash
pip install -e ".[dev]"
```
This creates an editable install and installs the dev dependencies.

Please use `isort` and `black` to format your code. Open up issues for bugs/missing
features. Use pull requests for additions to the code. Write tests that can be run
by [`pytest`](https://docs.pytest.org/).
