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

* Implementing a nonlinear :doc:`hyperelasticity equation
  <demos/hyperelasticity/python/demo_hyperelasticity.py>`

* Using a mixed formulation to solve the time-dependent, nonlinear
  :doc:`Cahn-Hilliard equation <demos/cahn-hilliard/python/demo_cahn-hilliard.py>`

* Computing eigenvalues of the :doc:`Maxwell eigenvalue problem
  <demos/maxwell-eigenvalues/python/demo_maxwell-eigenvalues.py>`


Working list of Python demos
----------------------------

* :doc:`demos/poisson/python/demo_poisson.py`
* :doc:`demos/eigenvalue/python/demo_eigenvalue.py`
* :doc:`demos/built-in-meshes/python/demo_built-in-meshes.py`
* :doc:`demos/mixed-poisson/python/demo_mixed-poisson.py`
* :doc:`demos/biharmonic/python/demo_biharmonic.py`
* :doc:`demos/auto-adaptive-poisson/python/demo_auto-adaptive-poisson.py`
* :doc:`demos/cahn-hilliard/python/demo_cahn-hilliard.py`
* :doc:`demos/maxwell-eigenvalues/python/demo_maxwell-eigenvalues.py`
* :doc:`demos/built-in-meshes/python/demo_built-in-meshes.py`
* :doc:`demos/hyperelasticity/python/demo_hyperelasticity.py`
* :doc:`demos/nonlinear-poisson/python/demo_nonlinear-poisson.py`
* :doc:`demos/nonmatching-interpolation/python/demo_nonmatching-interpolation.py`


Using the C++ interface
-----------------------

* :doc:`demos/poisson/cpp/main.cpp`
* :doc:`demos/eigenvalue/cpp/main.cpp`
* :doc:`demos/built-in-meshes/cpp/main.cpp`
* :doc:`demos/mixed-poisson/cpp/main.cpp`
* :doc:`demos/biharmonic/cpp/main.cpp`
* :doc:`demos/auto-adaptive-poisson/cpp/main.cpp`
* :doc:`demos/nonmatching-interpolation/cpp/main.cpp`
* :doc:`demos/hyperelasticity/cpp/main.cpp`

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
   demo_stokes_mini
