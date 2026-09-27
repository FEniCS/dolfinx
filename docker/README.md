# DOLFINx Docker containers

This document summarises all of the `Dockerfile`s in this directory, the images
they are built into, and how they are built.

## `Dockerfile.test-env`

This Dockerfile describes a complete development and testing environment for
DOLFINx based on Ubuntu. It does not contain the FEniCS components.

The following image is produced and pushed to both `docker.io/fenicsproject`
and `ghcr.io/fenics` and is used in our testing infrastructure:

* `test-env:current` - Debugging on. Used on GitHub Actions.

The following image is pushed to both `docker.io/dolfinx` and
`ghcr.io/fenics/dolfinx` and can be used by end-users to build FEniCS from
source:

* `dev-env:current` - OpenMPI, debugging off.
* Versioned images e.g. `dev-env:v0.11.0` suitable for the matching DOLFINx
  version.
* A special tag `:stable` points at the latest versioned image.

Both images use Ubuntu's default system MPI (`mpi-default-dev`, which
resolves to OpenMPI) and system parallel HDF5.

A build must be triggered manually via GitHub Actions to update the `:current-*`
tags. All images are multi-architecture (x86-64 and ARM64).

These images are not built automatically on a fixed schedule, so they can move
out-of-sync with what is in `Dockerfile.test-env`.

## `Dockerfile.end-user`

This Dockerfile describes complete DOLFINx environments based on Ubuntu. By
default, it uses the `dev-env:current` image as a base image. The images are
intended for end-users.

The following images are pushed to both `docker.io/dolfinx` and
`ghcr.io/fenics/dolfinx`:

* `dolfinx:nightly` - Terminal environment.
* `lab:nightly` - JupyterLab environment.
* `dolfinx-onbuild:nightly` - Onbuild environment to automatically build FEniCS
  from source.
* Versioned images e.g. `dolfinx:v0.11.0`, `dev-env:v0.11.0`.
* A special tag `:stable` points at the latest versioned images, e.g. `lab:stable`.

A build of this Dockerfile is triggered automatically every night to produce
the `:nightly` tags. All images are multi-architecture (x86-64 and ARM64).

In addition, a build of this Dockerfile can be triggered manually via GitHub
Actions with a specific set of FEniCSx git tags to produce versioned images
e.g. `:v0.6.0-r1`.
