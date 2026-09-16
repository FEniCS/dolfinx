.. DOLFINx demos

Demos
=====

These demos illustrate the use of DOLFINx. Each is available as a Python
script and as a Jupyter notebook (see the "Download sources" box at the
top of each demo page). If you are new to DOLFINx, start with
:doc:`demos/demo_poisson` and then work through *Getting started*.

The remaining sections are organised by topic rather than by difficulty,
and a demo that illustrates more than one topic is listed in more than
one section. Some demos have additional requirements, noted below: the
electromagnetics demos require DOLFINx to be built with complex PETSc
scalars, and a few demos only run in serial.


Getting started
---------------

* :doc:`demos/demo_poisson` -- the recommended starting point: solve the
  Poisson equation with mixed Dirichlet/Neumann boundary conditions.
* :doc:`demos/demo_helmholtz` -- solve the Helmholtz equation with both
  real-valued and complex-valued formulations.
* :doc:`demos/demo_biharmonic` -- solve the biharmonic equation using an
  interior penalty discontinuous Galerkin method.


Interpolation, IO and visualisation
-----------------------------------

* :doc:`demos/demo_pyvista` -- visualise finite element functions with
  PyVista, including warp-by-scalar and warp-by-vector plots.
* :doc:`demos/demo_interpolation-io` -- interpolate into an
  :math:`H(\mathrm{curl})` Nédélec space and visualise it via a
  discontinuous Lagrange space.


Mixed and hybridised formulations
---------------------------------

* :doc:`demos/demo_mixed-poisson` -- solve the Poisson equation in mixed
  (flux, potential) form with a block-preconditioned iterative solver.
* :doc:`demos/demo_stokes` -- solve the Stokes equations with Taylor-Hood
  elements, comparing five block and monolithic solver strategies.
* :doc:`demos/demo_navier-stokes` -- solve the Navier-Stokes equations
  with a divergence-conforming discontinuous Galerkin method.
* :doc:`demos/demo_hdg` -- solve the Poisson equation with a hybridised
  discontinuous Galerkin (HDG) scheme, using a submesh of facets.
* :doc:`demos/demo_static-condensation` -- solve a mixed linear
  elasticity formulation with static condensation of the stress
  degrees-of-freedom, using a numba-generated kernel.


Time-dependent and nonlinear problems
-------------------------------------

* :doc:`demos/demo_cahn-hilliard` -- solve the time-dependent, nonlinear
  Cahn-Hilliard equation with a Newton solver.
* :doc:`demos/demo_navier-stokes` -- time-step the semi-implicit
  divergence-conforming Navier-Stokes scheme (see also *Mixed and
  hybridised formulations*).


Linear solvers, preconditioners and matrix-free methods
----------------------------------------------------------

* :doc:`demos/demo_elasticity` -- solve the linear elasticity equations
  using a smoothed aggregation algebraic multigrid solver.
* :doc:`demos/demo_pyamg` -- solve the Poisson and linearised elasticity
  equations using algebraic multigrid from `pyamg
  <https://github.com/pyamg/pyamg>`_ (serial only).
* :doc:`demos/demo_stokes` -- see *Mixed and hybridised formulations*:
  five Stokes solver configurations, from block-preconditioned to fully
  monolithic.
* :doc:`demos/demo_mixed-poisson` -- see *Mixed and hybridised
  formulations*: a block-preconditioned solver, including a Hypre AMS
  preconditioner for :math:`H(\mathrm{div})`.
* :doc:`demos/demo_poisson-matrix-free` -- solve the Poisson equation
  with a matrix-free conjugate gradient solver.
* :doc:`demos/demo_matrix-free-petsc` -- solve a blocked projection
  problem with a matrix-free PETSc ``SHELL`` operator.
* :doc:`demos/demo_types` -- solve the Poisson equation using different
  scalar types (single/double precision, real/complex) and SciPy sparse
  solvers.


Custom and advanced finite elements
--------------------------------------

* :doc:`demos/demo_lagrange-variants` -- create Lagrange elements with
  different node placements (equispaced versus Gauss--Lobatto--Legendre)
  using Basix.
* :doc:`demos/demo_tnt-elements` -- define a custom finite element (a
  tiniest tensor element) using Basix's custom element interface.


Mesh generation, partitioning and parallel data
---------------------------------------------------

* :doc:`demos/demo_gmsh` -- generate and tag meshes using the Gmsh
  Python interface.
* :doc:`demos/demo_partition` -- compare graph and geometric mesh
  partitioning strategies and measure partition quality.
* :doc:`demos/demo_comm-pattern` -- build and analyse the parallel
  communication pattern of a distributed mesh with NetworkX.
* :doc:`demos/demo_mixed-topology` -- solve a Helmholtz problem on a
  mesh with mixed cell topology (in development, serial only).


Electromagnetics
----------------

All demos in this section require DOLFINx to be built with complex
PETSc scalars.

* :doc:`demos/demo_half-loaded-waveguide` -- compute eigenmodes of a
  half-loaded rectangular waveguide using SLEPc.
* :doc:`demos/demo_scattering-boundary-conditions` -- simulate
  electromagnetic scattering from a wire using scattering boundary
  conditions.
* :doc:`demos/demo_pml` -- simulate electromagnetic scattering from a
  wire using a perfectly matched layer (PML).
* :doc:`demos/demo_axis` -- simulate axisymmetric electromagnetic
  scattering from a sphere using an axisymmetric PML.


..
   The following hidden toctree serves as the master site map for Sphinx.
   It ensures the left sidebar populates cleanly without duplicating items.

.. toctree::
   :hidden:
   :maxdepth: 1

   demos/demo_poisson.md
   demos/demo_helmholtz.md
   demos/demo_biharmonic.md
   demos/demo_mixed-poisson.md
   demos/demo_stokes.md
   demos/demo_navier-stokes.md
   demos/demo_hdg.md
   demos/demo_static-condensation.md
   demos/demo_cahn-hilliard.md
   demos/demo_half-loaded-waveguide.md
   demos/demo_scattering-boundary-conditions.md
   demos/demo_pml.md
   demos/demo_axis.md
   demos/demo_elasticity.md
   demos/demo_pyamg.md
   demos/demo_poisson-matrix-free.md
   demos/demo_matrix-free-petsc.md
   demos/demo_types.md
   demos/demo_lagrange-variants.md
   demos/demo_tnt-elements.md
   demos/demo_pyvista.md
   demos/demo_interpolation-io.md
   demos/demo_gmsh.md
   demos/demo_partition.md
   demos/demo_comm-pattern.md
   demos/demo_mixed-topology.md
