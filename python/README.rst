Python wrapping with pybind11
=============================

This work is experimental and under active development. It supports
Python 3 only.

1. Install the development version of pybdind11. Use CMake to install
   pybind11, e.g.::

     git clone https://github.com/pybind/pybind11.git
     cd pybind11
     mkdir build-dir
     cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=/path/to/pybind11/install ../
     make install

2. Build and install DOLFIN in the usual way (with or without SWIG
   bindings).

3. Build DOLFIN Python interface::

     export PYBIND11_DIR=/path/to/pybind11/install
     pip3 -v install --user

   To install locally:

     python3 setup.py build

   and set the ``PYTHONPATH``.
