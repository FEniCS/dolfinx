# DOLFINx

[![DOLFINx CI](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml)
[![Actions Conda install](https://github.com/FEniCS/dolfinx/actions/workflows/conda.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/conda.yml)
[![Actions macOS/Homebrew install](https://github.com/FEniCS/dolfinx/actions/workflows/macos.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/macos.yml)
[![Actions Docker images](https://github.com/FEniCS/dolfinx/actions/workflows/docker-end-user.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/docker-end-user.yml)
[![Actions Windows/vcpkg install](https://github.com/FEniCS/dolfinx/actions/workflows/windows.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/windows.yml)

DOLFINx is the computational environment of
[FEniCSx](https://fenicsproject.org) and implements the FEniCS Problem
Solving Environment in C++ and Python. DOLFINx is a new version of
DOLFIN and is actively developed.

For questions about using DOLFINx, visit the [FEniCS
Discourse](https://fenicsproject.discourse.group/) page or use the
[FEniCS Slack channel](https://fenicsproject.slack.com/) (use
[this](https://join.slack.com/t/fenicsproject/shared_invite/zt-2wa76d0xw-ecfxcPj25K8pc7q7~1AK8g)
link to sign up to the Slack channel).

## Documentation

Documentation can be viewed [here](https://docs.fenicsproject.org).

## Installation

### Binary packages

**Recommendations**

- Most users: [pip (preview)](#pip-preview) or [conda](#conda), on Linux,
  macOS and Windows. pip does not include PETSc yet, and conda does not
  include PETSc on Windows.
- With PETSc: [conda](#conda) (Linux and macOS),
  [Ubuntu](#ubuntu-packages)/[Debian](#debian-packages) packages or
  [Docker](#docker-images). On Windows, install
  [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and use
  [Ubuntu](#ubuntu-packages).
- High performance computers: [Spack](#spack) or
  [from source](#from-source), both using system-provided MPI.
  [pip (preview)](#pip-preview) works with an MPICH-compatible MPI (MPICH,
  Cray MPICH, Intel MPI, MVAPICH) loaded from the module system, in place
  of the `mpich` package.

#### conda

To install the latest release of the Python interface, including pyvista for
visualisation, using [conda](https://conda.io):

```shell
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista # Linux and macOS
conda install -c conda-forge fenics-dolfinx pyvista pyamg # Windows
```

*Windows only*: PETSc and petsc4py are not available on Windows. Because
FEniCS uses just-in-time compilation it is necessary to install
[Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/).

conda is distributed with [Anaconda](https://www.anaconda.com/) and
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). The recipe
is hosted on
[conda-forge](https://github.com/conda-forge/fenics-dolfinx-feedstock).

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-fenics--dolfinx-green.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) | [![Conda Downloads](https://anaconda.org/conda-forge/fenics-dolfinx/badges/downloads.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) | [![Conda Version](https://anaconda.org/conda-forge/fenics-dolfinx/badges/version.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) | [![Conda Platforms](https://anaconda.org/conda-forge/fenics-dolfinx/badges/platforms.svg)](https://anaconda.org/conda-forge/fenics-dolfinx) |

#### Ubuntu packages

The [Ubuntu
PPA](https://launchpad.net/~fenics-packages/+archive/ubuntu/fenics)
provides FEniCSx packages. To install:

```shell
add-apt-repository ppa:fenics-packages/fenics
apt update
apt install fenicsx
```

When a version of DOLFINx is released we aim to provide a package for
the most recent LTS version of Ubuntu. All other versions are provided
on a best-effort basis.

#### Debian packages

[DOLFINx](https://tracker.debian.org/pkg/fenics-dolfinx) is included
with [various
versions](https://packages.debian.org/search?keywords=python3-dolfinx&searchon=names&exact=1&suite=all&section=all)
of Debian. Install with `apt-get install fenicsx`.

#### Docker images

To run a Docker image with the latest release of DOLFINx:

```shell
docker run -ti dolfinx/dolfinx:stable
```

To switch between real and complex builds of DOLFINx/PETSc.

```shell
source /usr/local/bin/dolfinx-complex-mode
source /usr/local/bin/dolfinx-real-mode
```

A Jupyter Lab environment with the latest release of DOLFINx:

```shell
docker run --init -ti -p 8888:8888 dolfinx/lab:stable  # Access at http://localhost:8888
```

A Docker image with DOLFINx built nightly:

```shell
docker run -ti dolfinx/dolfinx:nightly
```

A development image with all of the dependencies required to build the
latest release of the FEniCSx components:

```shell
docker run -ti dolfinx/dev-env:stable
```

A development image with all of the dependencies required
to build the `main` branch of the FEniCSx components:

```shell
docker run -ti dolfinx/dev-env:current
```

The Docker images support arm64 and amd64 architectures. For a full list
of tags, including versioned images, see
<https://hub.docker.com/u/dolfinx>

#### pip (preview)

Development builds of `main` are available from
[Test PyPI](https://test.pypi.org/) for Python 3.12+. They are built without
PETSc and with SuperLU_DIST for sparse direct solves.

Linux (x86_64 and aarch64):

```shell
python3 -m venv fenicsx-env
source fenicsx-env/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "fenics-dolfinx>=0.12.0.dev0" mpich
```

The Linux wheels need glibc 2.34 or later, e.g. RHEL 9 or Ubuntu 22.04. Check
with `ldd --version`. On older systems pip reports `from versions: none`.

macOS (Apple silicon and Intel), with the Xcode Command Line Tools:

```shell
python3 -m venv fenicsx-env
source fenicsx-env/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "fenics-dolfinx>=0.12.0.dev0" openmpi
```

Windows (x86_64), in PowerShell:

```powershell
# Or install Python 3.12 or later and the Visual Studio C++ build tools manually
winget install -e --id Python.Python.3.13
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Then in a new PowerShell:

```powershell
py -3 -m venv fenicsx-env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
fenicsx-env\Scripts\Activate.ps1
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "fenics-dolfinx>=0.12.0.dev0" impi-rt
# Put mpiexec from impi-rt on the PATH, in each new session
$env:Path = "$env:VIRTUAL_ENV\Library\bin;$env:Path"
```

Check that `mpiexec` comes from `fenicsx-env`, since another MPI on the `PATH`,
e.g. from Homebrew, can shadow it. `which -a mpiexec` on Linux and macOS, or
`Get-Command -All mpiexec` on Windows, should list it first. A new login shell
can put the system MPI back in front, so reactivate the environment there.

With `uv pip install`, also pass `--index-strategy unsafe-best-match`.

### From source

For detailed instructions and a list of dependencies, see
[here](https://docs.fenicsproject.org/dolfinx/main/python/installation).

#### C++ core

To build and install the C++ core, in the `cpp/` directory, run:

```shell
mkdir build
cd build
cmake ..
make install
```

#### Python interface

To install the Python interface, first install the C++ core, and then in
the `python/` directory run:

```shell
pip install --group pyproject.toml:build
pip install --check-build-dependencies --no-build-isolation .
```

### Spack

To build the most recent release using
[Spack](https://spack.readthedocs.io/) (assuming a bash-compatible
shell):

```shell
git clone https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack install --add py-fenics-dolfinx+petsc4py+slepc4py
```

Spack is the recommended approach for HPC systems.

Although Spack already contains FEniCS Project packages, we also maintain a
separate [overlay package repository](https://github.com/FEniCS/spack-fenics)
which may contain bug fixes and package specs for newer releases of the FEniCS
Project components, particularly after a recent release.

```shell
spack repo add https://github.com/fenics/spack-fenics
spack repo ls # shows fenics above builtin
```

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
<https://www.gnu.org/licenses/>.
