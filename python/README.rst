DOLFINx Python interface
========================

This document explains how to install the DOLFINx Python interface. Note that
this interface must be built without PEP517 build isolation by passing
`--no-build-isolation` flag to `pip install`. This is because the Python
interface explicitly depends on the system built petsc4py and mpi4py.

1. Build and install the DOLFINx C++ library in the usual way.

2. Ensure the build time requirements are installed::

     pip install -r build-requirements.txt

3. Build DOLFINx Python interface::

     pip install --check-build-dependencies --no-build-isolation .

   To build in debug and editable mode for development::

     pip -v install --check-build-dependencies --config-settings=build-dir="build" --config-settings=cmake.build-type="Debug" --no-build-isolation -e .
