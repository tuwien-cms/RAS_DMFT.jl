# RAS_DMFT

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tuwien-cms.github.io/RAS_DMFT.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tuwien-cms.github.io/RAS_DMFT.jl/dev)
[![Build Status](https://github.com/tuwien-cms/RAS_DMFT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tuwien-cms/RAS_DMFT.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/tuwien-cms/RAS_DMFT.jl/graph/badge.svg?token=5ACAMMA64E)](https://codecov.io/gh/tuwien-cms/RAS_DMFT.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

Restricted active space DMFT solver on the real frequency axis.

## Installation

This package depends on the private `Fermions.jl`,
which has to be installed first.

As the package is not inside the
[General registry](https://github.com/JuliaRegistries/General),
it needs to be added
[manually](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages).
Once `Fermions.jl` is installed, run

```sh
julia --project=path/to/project --eval 'using Pkg; Pkg.add(url="https://github.com/tuwien-cms/RAS_DMFT.jl")'
```

## Usage

The [tutorial](https://tuwien-cms.github.io/RAS_DMFT.jl/stable/generated/tutorial/)
solves a half-filled single-band Hubbard model on the Bethe lattice.
See also the [API reference](https://tuwien-cms.github.io/RAS_DMFT.jl/stable/api/).
