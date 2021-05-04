DOLFINx Python interface
========================

Building the DOLFINx Python interface uses pybind11.

1. Install pybdind11 version 2.2.1 or later. Use CMake to install
   pybind11, e.g.::

     wget -nc --quiet https://github.com/pybind/pybind11/archive/v2.2.1.tar.gz
     tar -xf v2.2.1.tar.gz
     cd pybind11-2.2.1
     mkdir build-dir
     cd build-dir
     cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=/path/to/pybind11/install ..
     make install

   The FEniCS Docker images are configured with pybind11.

2. Build and install the DOLFINx C++ library in the usual way.

3. Build DOLFINx Python interface::

     export PYBIND11_DIR=/path/to/pybind11/install
     pip3 -v install . --user

To install in a local build directory::

  python3 setup.py build

and set the ``PYTHONPATH``. To build in debug mode::

  python3 setup.py build --debug
