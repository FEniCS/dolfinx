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
   <demos/poisson/python/demo_poisson.py>`.

2. Solving nonlinear PDEs: :doc:`Solving a nonlinear Poisson equation
   <demos/nonlinear-poisson/python/demo_nonlinear-poisson.py>`

3. Using mixed elements: :doc:`Solving the Stokes equations
   <demos/stokes_taylor_hood/python/demo_stokes_taylor_hood.py>`

4. Using iterative linear solvers: :doc:`Solving the Stokes equations
   more efficiently <demos/stokes_iterative/python/demo_stokes_iterative.py>`

More advanced DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^^^^^^

These examples typically demonstrate how to solve a certain PDE using
more advanced techniques. We recommend that you take a look at these
demos for tips and tricks on how to use more advanced or lower-level
functionality and optimizations.

* Implementing a nonlinear :doc:`hyperelasticity equation
  <demos/hyperelasticity/python/demo_hyperelasticity.py>`

* Using a mixed formulation to solve the time-dependent, nonlinear
  :doc:`Cahn-Hilliard equation <demos/cahn-hilliard/python/demo_cahn-hilliard.py>`

* Computing eigenvalues of the :doc:`Maxwell eigenvalue problem
  <demos/maxwell-eigenvalues/python/demo_maxwell-eigenvalues.py>`

All documented Python demos
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   demos/poisson/python/demo_poisson.py.rst
   demos/eigenvalue/python/demo_eigenvalue.py.rst
   demos/built-in-meshes/python/demo_built-in-meshes.py.rst
   demos/mixed-poisson/python/demo_mixed-poisson.py.rst
   demos/biharmonic/python/demo_biharmonic.py.rst
   demos/auto-adaptive-poisson/python/demo_auto-adaptive-poisson.py.rst
   demos/cahn-hilliard/python/demo_cahn-hilliard.py.rst
   demos/maxwell-eigenvalues/python/demo_maxwell-eigenvalues.py.rst
   demos/hyperelasticity/python/demo_hyperelasticity.py.rst
   demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.rst
   demos/singular-poisson/python/demo_singular-poisson-rst.py.rst
   demos/neumann-poisson/python/demo_neumann-poisson.py.rst
   demos/nonmatching-interpolation/python/demo_nonmatching-interpolation.py.rst
   demos/stokes-iterative/python/demo_stokes-iterative.py.rst

Using the C++ interface
-----------------------

All documented C++ demos
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   demos/poisson/cpp/main.cpp.rst
   demos/eigenvalue/cpp/main.cpp.rst
   demos/built-in-meshes/cpp/main.cpp.rst
   demos/mixed-poisson/cpp/main.cpp.rst
   demos/biharmonic/cpp/main.cpp.rst
   demos/auto-adaptive-poisson/cpp/main.cpp.rst
   demos/nonmatching-interpolation/cpp/main.cpp.rst
   demos/hyperelasticity/cpp/main.cpp.rst


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

