========
DOLFIN-X
========

DOLFIN-X is an experimental version of DOLFIN. It is being actively
developed, but is **not ready for production use**. Many new
experimental features may come and go as development proceeds.

DOLFIN is the computational backend of FEniCS and implements the
FEniCS Problem Solving Environment in Python and C++.


Installation
============

C++ core
--------

To build and install the C++ core, in the `cpp/` directory, run::

  mkdir build
  cd build
  cmake ..
  make install

Python
------

To install the Python interface, first install the C++ core, and then
in the `python/` directory run::

  pip install .

(you may need to use `pip3`, depending on your system).

For detailed instructions, see the file INSTALL.


Documentation
=============

Documentation can be viewed at http://fenics-dolfin.readthedocs.org/.

.. image:: https://readthedocs.org/projects/fenics-dolfin/badge/?version=latest
   :target: http://fenics.readthedocs.io/projects/dolfin/en/latest/?badge=latest
   :alt: Documentation Status


Automated Testing
=================

We use CircleCI to perform automated testing.

.. image:: https://bitbucket-badges.useast.atlassian.io/badge/fenics-project/dolfin.svg
   :target: https://bitbucket.org/fenics-project/dolfin/addon/pipelines/home
   :alt: Pipelines Build Status


Code Coverage
=============

Code coverage reports can be viewed at
https://coveralls.io/bitbucket/fenics-project/dolfin.

.. image:: https://coveralls.io/repos/bitbucket/fenics-project/dolfin/badge.svg?branch=master
   :target: https://coveralls.io/bitbucket/fenics-project/dolfin?branch=master
   :alt: Coverage Status


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

For questions about using DOLFIN-X, visit the FEniCS Q&A page:

  https://www.allanswered.com/community/s/fenics-project/

For bug reports, visit the DOLFIN-X Bitbucket page:

  http://bitbucket.org/fenics-project/dolfinx


About
=====

DOLFIN is developed by a group of mathematicians, computational
scientists and engineers distributed around the world. A list of
authors can be found in the file AUTHORS. For more information about
DOLFIN, visit

  http://fenicsproject.org
