========
DOLFIN-X
========

.. image:: https://circleci.com/gh/FEniCS/dolfinx.svg?style=shield
    :target: https://circleci.com/gh/FEniCS/dolfinx

DOLFIN-X is an experimental version of DOLFIN. It is being actively
developed, but is **not ready for production use**. New experimental
features may come and go as development proceeds.

DOLFIN is the computational backend of FEniCS and implements the FEniCS
Problem Solving Environment in Python and C++.


Documentation
=============

Documentation can be viewed at:

- https://fenicsproject.org/docs/dolfinx/dev/cpp/
- https://fenicsproject.org/docs/dolfinx/dev/python/


Installation
============

C++ core
--------

To build and install the C++ core, in the ``cpp/`` directory, run::

  mkdir build
  cd build
  cmake ..
  make install

Python interface
----------------

To install the Python interface, first install the C++ core, and then
in the ``python/`` directory run::

  pip install .

(you may need to use ``pip3``, depending on your system).

For detailed instructions, see the file INSTALL.


License
=======

DOLFIN-X is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DOLFIN-X is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with DOLFIN-X. If not, see
<http://www.gnu.org/licenses/>.


Contact
=======

For comments and requests, send an email to the FEniCS mailing list:

fenics-dev@googlegroups.com

For questions related to obtaining, building or installing DOLFIN-X,
send an email to the FEniCS support mailing list:

fenics-support@googlegroups.com

For questions about using DOLFIN-X, visit the FEniCS Discourse page:

https://fenicsproject.discourse.group/

For bug reports, visit the DOLFIN-X GitHub page:

https://github.com/FEniCS/dolfinx
