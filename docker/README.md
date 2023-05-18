# DOLFINx Docker containers

This document summarises all of the `Dockerfile`s in this directory, the images
they are built into, and how they are built.

## `Dockerfile.test-env`

This Dockerfile describes a complete development and testing environment for
DOLFINx based on Ubuntu. It does not contain the FEniCS components.

The following images are produced and pushed to both `docker.io/fenicsproject`
and `ghcr.io/fenics` and are used in our testing infrastructure:

* `test-env:current-mpich` - MPICH, debugging on. Used on CircleCI.
* `test-env:current-openmpi` - OpenMPI, debugging on. Used on GitHub Actions.

The following images are pushed to both `docker.io/dolfinx` and
`ghcr.io/fenics/dolfinx` and can be used by end-users to build FEniCS from
source:

* `dev-env:current-mpich` -  MPICH, debugging off.
* `dev-env:current-openmpi` - OpenMPI, debugging off.
* `dev-env:current` - Points at `dev-env:current-mpich`.

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

A build of this Dockerfile is triggered automatically every night to produce
the `:nightly` tags. All images are multi-architecture (x86-64 and ARM64).

In addition, a build of this Dockerfile can be triggered manually via GitHub
Actions with a specific set of FEniCSx git tags to produce versioned images
e.g. `:v0.6.0-r1`.

A special tag `:stable` points at the latest versioned image.

## `Dockerfile.oneapi`

This Dockerfile describes a complete development and testing environment for
DOLFINx based on Ubuntu with the Intel OneAPI development tools installed. It
does not contain the FEniCS components.

A build of this `Dockerfile` produces an image at
`docker.io/fenicsproject/test-env:current-oneapi` which is used on GitHub
Actions for testing. This image is x86-64 only.

A build must be triggered manually via GitHub Actions to update the image.

## `Dockerfile.redhat`

This Dockerfile describes a complete development and testing environment for
DOLFINx based on a RedHat-compatible distribution. It does not contain the
FEniCS components.

A build of this `Dockerfile` produces an image at
`docker.io/fenicsproject/test-env:current-redhat` which is used on GitHub
Actions for testing. This image is x86-64 only at this time. 

A build must be triggered manually via GitHub Actions to update the image.
