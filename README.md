# DOLFINx

[![DOLFINx CI](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml)
[![CircleCI](https://circleci.com/gh/FEniCS/dolfinx.svg?style=shield)](https://circleci.com/gh/FEniCS/dolfinx)
[![Actions Docker environment images](https://github.com/FEniCS/dolfinx/workflows/Docker%20environment%20images/badge.svg)](https://github.com/FEniCS/dolfinx/actions?query=workflow%3A%22Docker+environment+images%22)
[![Actions Docker image builds](https://github.com/FEniCS/dolfinx/workflows/Docker%20end-user%20images/badge.svg)](https://github.com/FEniCS/dolfinx/actions?query=workflow%3A%22Docker+end-user+images%22)
[![Actions Spack build](https://github.com/FEniCS/dolfinx/workflows/Spack%20build/badge.svg)](https://github.com/FEniCS/dolfinx/actions?query=workflow%3A%22Spack+build%22)

DOLFINx is a new version of DOLFIN. It is being actively developed and
features may come and go as development proceeds.

DOLFINx is the computational environment of
[FEniCS](https://fenicsproject.org) and implements the FEniCS Problem
Solving Environment in Python and C++.

## Documentation

Documentation can be viewed at:

- https://docs.fenicsproject.org/dolfinx/main/cpp/
- https://docs.fenicsproject.org/dolfinx/main/python/

## Installation

### From source

#### C++ core

To build and install the C++ core, in the ``cpp/`` directory, run::
```
mkdir build
cd build
cmake ..
make install
```

#### Python interface

To install the Python interface, first install the C++ core, and then
in the ``python/`` directory run::
```
pip install .
```
(you may need to use ``pip3``, depending on your system).

For detailed instructions, see the file INSTALL.

### Spack

To build from source using [Spack](https://spack.readthedocs.io/) (assuming a bash shell):
```
git clone https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
echo "  concretization: together" >> ./spack/var/spack/environments/fenicsx-env/spack.yaml
spack add py-fenics-dolfinx@main ^petsc+mumps+hypre cflags="-O3" fflags="-O3"
spack install
```
See the Spack [documentation](https://spack.readthedocs.io/) for
comprehensive instructions.


## Docker images


A Docker image with DOLFINx built nightly:
```
docker run -ti dolfinx/dolfinx:latest
```

To switch between real and complex builds of DOLFINx.
```
source /usr/local/bin/dolfinx-complex-mode
source /usr/local/bin/dolfinx-real-mode
```

A Jupyter Lab environment with DOLFINx built nightly:
```
docker run --init -ti -p 8888:8888 dolfinx/lab:latest # Access at http://localhost:8888
```

A development image with all of the dependencies required
to build DOLFINx:
```
docker run -ti dolfinx/dev-env:latest
```

All Docker images support arm64 and amd64 architectures.

For more information, see https://hub.docker.com/u/dolfinx

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

https://fenicsproject.discourse.group/

For bug reports, visit the DOLFINx GitHub page:

https://github.com/FEniCS/dolfinx

For comments and requests, send an email to the FEniCS mailing list:

fenics-dev@googlegroups.com

For questions related to obtaining, building or installing DOLFINx,
send an email to the FEniCS support mailing list:

fenics-support@googlegroups.com
