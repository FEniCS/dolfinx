.. DOLFIN API documentation


Demo documentation
==================

*Under development*

.. todo:: Give some structure to the demos, e.g. beginner, advanced,
          linear nonlinear, etc.

Using the Python interface
--------------------------

Introductory DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^^^^^

These demos illustrate core DOLFIN/FEniCS usage and are a good way to
begin learning FEniCS. We recommend that you go through these examples
in the given order.

1. Getting started: :ref:`Solving the Poisson equation
   <demo_poisson_equation>`.

2. Solving nonlinear PDEs: :ref:`Solving a hyperelasticity equation
   <demo_hyperelasticity>`

Advanced DOLFIN demos
^^^^^^^^^^^^^^^^^^^^^

These examples typically demonstrate how to solve a certain PDE using
more advanced techniques. We recommend that you take a look at these
demos for tips and tricks on how to use more advanced or lower-level
functionality and optimizations.

* Using a mixed formulation to solving the time-dependent, nonlinear
  :ref:`Cahn-Hilliard equation <demo_cahn_hilliard>`

Demos illustrating specific features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How to

* work with :ref:`built-in meshes <demo_built_in_meshes>`

* solve :ref:`variational eigenvalue problems <demo_eigenvalue>`

* set :ref:`boundary conditions on non-trivial geometries <demo_bcs>`

* use :ref:`automated goal-oriented error control
  <demo_auto_adaptive_poisson>`

* specify a :ref:`Discontinuous Galerkin formulation <demo_biharmonic>`



Using the C++ interface
-----------------------


..
   Advanced
   --------

   .. toctree::
      :caption: Python
      :maxdepth: 1
      :includehidden:

      Poisson equation <demos/demo_poisson-pylit.py>
      Poisson equation (singular)<demos/demo_singular-poisson-rst.py>
      Maxwell eigenvalue problem<demos/demo_maxwell-eigenvalues.py>

   .. toctree::
      :caption: C++
      :maxdepth: 1

      Poisson equation<demos/poisson-pylit/main.cpp>
      Poisson equation (test)<demos/poisson-pylit-test/main.cpp>

.. toctree::
   :hidden:
