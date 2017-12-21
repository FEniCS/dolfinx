.. DOLFIN demos

Demos
=====


Introductory demos
------------------

These demos illustrate core DOLFIN/FEniCS usage and are a good way to
begin learning FEniCS. We recommend that you go through these examples
in the given order.

1. Getting started: :doc:`Solving the Poisson equation
   <demos/poisson/demo_poisson.py>`.

2. Solving nonlinear PDEs: :doc:`Solving a nonlinear Poisson equation
   <demos/nonlinear-poisson/demo_nonlinear-poisson.py>`

3. Using mixed elements: :doc:`Solving the Stokes equations
   <demos/stokes_taylor_hood/demo_stokes_taylor_hood.py>`

4. Using iterative linear solvers: :doc:`Solving the Stokes equations
   more efficiently <demos/stokes_iterative/demo_stokes_iterative.py>`


More advanced demos
-------------------

These examples typically demonstrate how to solve a certain PDE using
more advanced techniques. We recommend that you take a look at these
demos for tips and tricks on how to use more advanced or lower-level
functionality and optimizations.

* Implementing a nonlinear :doc:`hyperelasticity equation
  <demos/hyperelasticity/demo_hyperelasticity.py>`

* Using a mixed formulation to solve the time-dependent, nonlinear
  :doc:`Cahn-Hilliard equation <demos/cahn-hilliard/demo_cahn-hilliard.py>`

* Computing eigenvalues of the :doc:`Maxwell eigenvalue problem
  <demos/maxwell-eigenvalues/demo_maxwell-eigenvalues.py>`


Documented demos
----------------

.. toctree::
   :maxdepth: 1

   demos/poisson/demo_poisson.py.rst
   demos/eigenvalue/demo_eigenvalue.py.rst
   demos/built-in-meshes/demo_built-in-meshes.py.rst
   demos/mixed-poisson/demo_mixed-poisson.py.rst
   demos/biharmonic/demo_biharmonic.py.rst
   demos/auto-adaptive-poisson/demo_auto-adaptive-poisson.py.rst
   demos/cahn-hilliard/demo_cahn-hilliard.py.rst
   demos/maxwell-eigenvalues/demo_maxwell-eigenvalues.py.rst
   demos/hyperelasticity/demo_hyperelasticity.py.rst
   demos/nonlinear-poisson/demo_nonlinear-poisson.py.rst
   demos/singular-poisson/demo_singular-poisson.py.rst
   demos/neumann-poisson/demo_neumann-poisson.py.rst
   demos/nonmatching-interpolation/demo_nonmatching-interpolation.py.rst
   demos/stokes-iterative/demo_stokes-iterative.py.rst
