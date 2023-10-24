DOLFINx Python interface
========================

This document explains how to install the DOLFINx Python interface. Note that
this interface must be built without PEP517 build isolation by passing
`--no-build-isolation` to `pip`. This is because the Python interface
explicitly depends on the system built petsc4py and mpi4py.

1. Build and install the DOLFINx C++ library in the usual way.

2. Ensure the build time requirements are installed::

     pip -v -r build-requirements.txt

3. Build DOLFINx Python interface::

     pip -v install --no-build-isolation .

   To build in debug mode for development::

     pip -v install --config-settings=build-dir="build" --config-settings=cmake.build-type="Debug" --no-build-isolation -e .
