.. DOLFIN API documentation

Demo documentation
==================

Using the Python interface
--------------------------

Introductory DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^^^^^

These demos illustrate core DOLFIN/FEniCS usage and are a good way to
begin learning FEniCS. We recommend that you go through these examples
in the given order.

1. Getting started: :doc:`Solving the Poisson equation
   <demos/demo_poisson.py>`.

2. Solving nonlinear PDEs: :doc:`Solving a nonlinear Poisson equation
   <demos/demo_nonlinear-poisson.py>`

3. Using mixed elements: :doc:`Solving the Stokes equations
   <demos/demo_stokes_taylor_hood.py>`

4. Using iterative linear solvers: :doc:`Solving the Stokes equations
   more efficiently <demos/demo_stokes_iterative.py>`

More advanced DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^^^^^^

These examples typically demonstrate how to solve a certain PDE using
more advanced techniques. We recommend that you take a look at these
demos for tips and tricks on how to use more advanced or lower-level
functionality and optimizations.

* Implementing a nonlinear :doc:`hyperelasticity equation
  <demos/demo_hyperelasticity.py>`

* Implementing a splitting method for solving the :doc:`incompressible
  Navier-Stokes equations <demos/demo_navier_stokes.py>`

* Using a mixed formulation to solve the time-dependent, nonlinear
  :doc:`Cahn-Hilliard equation <demos/demo_cahn-hilliard.py>`

* Computing eigenvalues of the :doc:`Maxwell eigenvalue problem
  <demos/demo_maxwell-eigenvalues.py>`

Demos illustrating specific features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How to

* work with :doc:`built-in meshes <demos/demo_built-in-meshes.py>`

* define and store :doc:`subdomains <demo_subdomains>`

* integrate over :doc:`subdomains <demo_subdomains_poisson>`

* set :doc:`boundary conditions on non-trivial geometries <demo_bcs>`

* solve :doc:`a basic eigenvalue problem <demos/demo_eigenvalue.py>`

* set :doc:`periodic boundary conditions <demo_periodic>`

* :doc:`de-singularize a pure Neumann problem <demos/demo_singular-poisson-rst.py>`
  by specifying the nullspace

* :doc:`de-singularize a pure Neumann problem <demo_neumann_poisson>`
  by adding a constraint

* use :doc:`automated goal-oriented error control
  <demos/demo_auto-adaptive-poisson.py>`

* specify a :doc:`Discontinuous Galerkin formulation <demos/demo_biharmonic.py>`

* work with :doc:`c++ expressions in Python programs
  <demo_tensor_weighted_poisson>`

* specify various finite element spaces

  * :doc:`Brezzi-Douglas-Marini elements for mixed Poisson <demos/demo_mixed-poisson.py>`
  * :doc:`the Mini element for Stokes equations <demo_stokes_mini>`


All documented Python demos
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   demos/demo_poisson.py.rst
   demos/demo_eigenvalue.py.rst
   demos/demo_built-in-meshes.py.rst
   demos/demo_mixed-poisson.py.rst
   demos/demo_biharmonic.py.rst
   demos/demo_auto-adaptive-poisson.py.rst
   demos/demo_cahn-hilliard.py.rst
   demos/demo_maxwell-eigenvalues.py.rst
   demos/demo_hyperelasticity.py.rst
   demos/demo_nonlinear-poisson.py.rst
   demos/demo_singular-poisson-rst.py.rst
   demos/demo_nonmatching-interpolation.py.rst


Using the C++ interface
-----------------------


All documented C++ demos
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   demos/poisson/main.cpp.rst
   demos/eigenvalue/main.cpp.rst
   demos/built-in-meshes/main.cpp.rst
   demos/mixed-poisson/main.cpp.rst
   demos/biharmonic/main.cpp.rst
   demos/auto-adaptive-poisson/main.cpp.rst
   demos/nonmatching-interpolation/main.cpp.rst


.. These were previously listed here, but many of them do not exist

   demos/demo_poisson_equation.py.rst
   demos/demo_nonlinear_poisson.py.rst
   demos/demos/demo_stokes_taylor_hood.py
   demos/demo_stokes_iterative.py.rst
   demos/demo_hyperelasticity.py.rst
   demos/demo_navier_stokes.py.rst
   demos/demo_cahn_hilliard.py.rst
   demos/demo_built_in_meshes.py.rst
   demos/demo_subdomains.py.rst
   demos/demo_subdomains_poisson.py.rst
   demos/demo_bcs.py.rst
   demos/demo_eigenvalue.py.rst
   demos/demo_periodic.py.rst
   demos/demo_singular_poisson.py.rst
   demos/demo_neumann_poisson.py.rst
   demos/demo_auto_adaptive_poisson.py.rst
   demos/demo_biharmonic.py.rst
   demos/demo_tensor_weighted_poisson.py.rst
   demos/demo_mixed_poisson.py.rst
   demos/demo_stokes_mini.py.rst

