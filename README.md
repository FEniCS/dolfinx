# DOLFINx

[![DOLFINx CI](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml)
[![CircleCI](https://circleci.com/gh/FEniCS/dolfinx.svg?style=shield)](https://circleci.com/gh/FEniCS/dolfinx)
[![Actions Docker images](https://github.com/FEniCS/dolfinx/actions/workflows/docker.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/docker.yml)
[![Actions Spack build](https://github.com/FEniCS/dolfinx/actions/workflows/spack.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/spack.yml)
[![Actions Conda install](https://github.com/FEniCS/dolfinx/actions/workflows/conda.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/conda.yml)
[![Actions macOS/Homebrew install](https://github.com/FEniCS/dolfinx/actions/workflows/macos.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/macos.yml)

DOLFINx is the computational environment of
[FEniCSx](https://fenicsproject.org) and implements the FEniCS Problem
Solving Environment in C++ and Python.

DOLFINx is a new version of DOLFIN and is being actively developed.

## Documentation

Documentation can be viewed at:

- <https://docs.fenicsproject.org/dolfinx/main/cpp/>
- <https://docs.fenicsproject.org/dolfinx/main/python/>

## Installation

### From source

#### C++ core

To build and install the C++ core, in the ``cpp/`` directory, run::

```shell
mkdir build
cd build
cmake ..
make install
```

#### Python interface

To install the Python interface, first install the C++ core, and then
in the ``python/`` directory run::

```shell
pip install .
```

(you may need to use ``pip3``, depending on your system).

For detailed instructions, see
<https://docs.fenicsproject.org/dolfinx/main/python/installation>.

### Binary

#### Operating System Recommendations

- macOS: use [conda](#conda).
- Linux: use [apt](#ubuntu-packages) ([Ubuntu](#ubuntu-packages)/[Debian](#debian-packages)), [docker](#docker-images) or [conda](#conda). See also [Spack](#spack).
- Windows: use [docker](#docker-images), or install Microsoft's [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) and use [Ubuntu](#ubuntu-packages).

#### conda

Conda is the recommended install method for Mac OS users. Linux users may also use it.

To install the latest stable release of the Python interface, with pyvista support for
visualisation, using [conda](https://conda.io):

```shell
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

conda is distributed with [Anaconda](https://www.anaconda.com/) and
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). The conda
recipe is hosted on
[conda-forge](https://github.com/conda-forge/fenics-dolfinx-feedstock).

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-fenics--dolfinx-green.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/fenics-dolfinx.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/fenics-dolfinx.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/fenics-dolfinx.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) |

> **Note**
> Windows packages are not available. This is due to some DOLFINx
> dependencies not supporting Windows.

#### Spack

Spack is recommended for building DOLFINx on HPC systems.

To build the most recent release using
[Spack](https://spack.readthedocs.io/) (assuming a bash-compatible
shell):

```shell
git clone https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack add py-fenics-dolfinx cflags="-O3" fflags="-O3"
spack install
```

See the Spack [documentation](https://spack.readthedocs.io/) for
comprehensive instructions.

#### Ubuntu packages

The [Ubuntu
PPA](https://launchpad.net/~fenics-packages/+archive/ubuntu/fenics) contains
binary packages of the FEniCSx components for Ubuntu.

To install:

```shell
add-apt-repository ppa:fenics-packages/fenics
apt update
apt install fenicsx
```

When a version of DOLFINx is released we aim to provide a build for the
latest stable version of Ubuntu. All other versions are provided on a
best-effort basis.

#### Debian packages

[DOLFINx](https://tracker.debian.org/pkg/fenics-dolfinx) has been included with [various
versions](https://packages.debian.org/search?keywords=python3-dolfinx&searchon=names&exact=1&suite=all&section=all)
of Debian. Install with `apt-get install fenicsx`.

#### Docker images

A Docker image with the latest stable release of DOLFINx:

```shell
docker run -ti dolfinx/dolfinx:stable
```

To switch between real and complex builds of DOLFINx/PETSc.

```shell
source /usr/local/bin/dolfinx-complex-mode
source /usr/local/bin/dolfinx-real-mode
```

A Jupyter Lab environment with the latest stable release of DOLFINx:

```shell
docker run --init -ti -p 8888:8888 dolfinx/lab:stable  # Access at http://localhost:8888
```

A Docker image with DOLFINx built nightly:

```shell
docker run -ti dolfinx/dolfinx:nightly
```

A development image with all of the dependencies required
to build the latest stable release of the FEniCSx components:

```shell
docker run -ti dolfinx/dev-env:stable
```

A development image with all of the dependencies required
to build the `main` branch of the FEniCSx components:

```shell
docker run -ti dolfinx/dev-env:nightly
```

All Docker images support arm64 and amd64 architectures.

For a full list of tags, including versioned images, see
<https://hub.docker.com/u/dolfinx>

## Contributing

Information about how to contribute to DOLFINx can be found
[here](CONTRIBUTING.md).

## License

DOLFINx is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DOLFINx is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with DOLFINx. If not, see
<http://www.gnu.org/licenses/>.

## Contact

For questions about using DOLFINx, visit the FEniCS Discourse page:

<https://fenicsproject.discourse.group/>

or use the FEniCS Slack channel:

<https://fenicsproject.slack.com/>

(use <https://fenicsproject-slack-invite.herokuapp.com/> to sign up)

For bug reports visit:

<https://github.com/FEniCS/dolfinx>
