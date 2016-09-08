.. DOLFIN API documentation

Demo documentation
==================

*Under development*

Using the Python interface
--------------------------

Introductory DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^^^^^

These demos illustrate core DOLFIN/FEniCS usage and are a good way to
begin learning FEniCS. We recommend that you go through these examples
in the given order.

1. Getting started: :ref:`Solving the Poisson equation
   <demo_poisson_equation>`.

2. Solving nonlinear PDEs: :ref:`Solving a nonlinear Poisson equation
   <demo_nonlinear_poisson>`

3. Using mixed elements: :ref:`Solving the Stokes equations
   <demo_stokes_taylor_hood>`

4. Using iterative linear solvers: :ref:`Solving the Stokes equations
   more efficiently <demo_stokes_iterative>`

More advanced DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^^^^^^

These examples typically demonstrate how to solve a certain PDE using
more advanced techniques. We recommend that you take a look at these
demos for tips and tricks on how to use more advanced or lower-level
functionality and optimizations.

* Implementing a nonlinear :ref:`hyperelasticity equation
  <demo_hyperelasticity>`

* Implementing a splitting method for solving the :ref:`incompressible
  Navier-Stokes equations <demo_navier_stokes>`

* Using a mixed formulation to solve the time-dependent, nonlinear
  :ref:`Cahn-Hilliard equation <demo_cahn_hilliard>`

* Computing eigenvalues of the :ref:`Maxwell eigenvalue problem
  <demos/demo_maxwell-eigenvalues.py>`

Demos illustrating specific features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How to

* work with :ref:`built-in meshes <demo_built_in_meshes>`

* define and store :ref:`subdomains <demo_subdomains>`

* integrate over :ref:`subdomains <demo_subdomains_poisson>`

* set :ref:`boundary conditions on non-trivial geometries <demo_bcs>`

* solve :ref:`a basic eigenvalue problem <demo_eigenvalue>`

* set :ref:`periodic boundary conditions <demo_periodic>`

* :ref:`de-singularize a pure Neumann problem <demo_singular_poisson>`
  by specifying the nullspace

* :ref:`de-singularize a pure Neumann problem <demo_neumann_poisson>`
  by adding a constraint

* use :ref:`automated goal-oriented error control
  <demo_auto_adaptive_poisson>`

* specify a :ref:`Discontinuous Galerkin formulation <demo_biharmonic>`

* work with :ref:`c++ expressions in Python programs
  <demo_tensor_weighted_poisson>`

* specify various finite element spaces

  * :ref:`Brezzi-Douglas-Marini elements for mixed Poisson <demo_mixed_poisson>`
  * :ref:`discontinuous Raviart-Thomas spaces for dual mixed Poisson
    <demo_mixed_poisson_dual>`
  * :ref:`the Mini element for Stokes equations <demo_stokes_mini>`


Working list of Python demos
----------------------------

* :doc:`demos/demo_poisson.py`
* :doc:`demos/demo_eigenvalue.py`
* :doc:`demos/demo_built-in-meshes.py`
* :doc:`demos/demo_biharmonic.py`
* :doc:`demos/demo_auto-adaptive-poisson.py`
* :doc:`demos/demo_cahn-hilliard.py`


Using the C++ interface
-----------------------

* :doc:`demos/poisson/main.cpp`
* :doc:`demos/eigenvalue/main.cpp`
* :doc:`demos/built-in-meshes/main.cpp`
* :doc:`demos/biharmonic/main.cpp`
* :doc:`demos/auto-adaptive-poisson/main.cpp`

.. todo:: Fix the toctree

.. toctree::
   :maxdepth: 1

   demo_poisson_equation
   demo_nonlinear_poisson
   demo_stokes_taylor_hood
   demo_stokes_iterative
   demo_hyperelasticity
   demo_navier_stokes
   demo_cahn_hilliard
   demo_built_in_meshes
   demo_subdomains
   demo_subdomains_poisson
   demo_bcs
   demo_eigenvalue
   demo_periodic
   demo_singular_poisson
   demo_neumann_poisson
   demo_auto_adaptive_poisson
   demo_biharmonic
   demo_tensor_weighted_poisson
   demo_mixed_poisson
   demo_mixed_poisson_dual
   demo_stokes_mini
