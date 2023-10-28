DOLFINx Python interface
========================

Building the DOLFINx Python interface uses nanobind.

1. Install nanobind from source, e.g.::

     python3 -m pip install git+https://github.com/wjakob/nanobind.git

   The FEniCS Docker images are configured with nanobind.

2. Build and install the DOLFINx C++ library in the usual way.

3. Build DOLFINx Python interface::

     pip3 -v install . --user

To install in a local build directory::

  python3 setup.py build

and set the ``PYTHONPATH``. To build in debug mode::

  python3 setup.py build --debug
