# Release notes

## v0.10.0

### Revised timer logic

**Authors**: [Garth N. Wells](https://github.com/garth-wells) and [Paul T. Kühner](https://github.com/schnellerhase)

Instead of using the [Boost timer library](https://www.boost.org/doc/libs/1_89_0/libs/timer/doc/index.html),
we have opted for the standard timing library [std::chrono](https://en.cppreference.com/w/cpp/header/chrono.html).
The switch is mainly due to some observed unaccuracies in timings with Boost.
This removes the notion of wall, system and user time.
See or {py:class}`Timer<dolfinx.common.Timer>` for examples of usage.


### Improved non-linear (Newton) solver

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd), [Jack Hale](https://github.com/jhale)
and [Garth N. Wells](https://github.com/garth-wells)

The FEniCS project has for the last 15 years had its own implementation of a Netwon solver.
We no longer see the need of providing this solver, as the {py:class}`PETSc SNES<petsc4py.PETSc.SNES>` solver,
and equivalent solver for C++ provides more features than our own implementation.

The previously shipped {py:class}`dolfinx.nls.petsc.NewtonSolver` is deprecated, in favor of
{py:class}`dolfinx.fem.petsc.NonlinearProblem`, which now integrates directly with {py:class}`petsc4py.PETSc.SNES`.

The non-linear problem object that was sent into {py:class}`dolfinx.nls.petsc.NewtonSolver` has been renamed
to {py:class}`NewtonSolverNonlinearProblem<dolfinx.fem.petsc.NewtonSolverNonlinearProblem>` and is also deprecated.

The new {py:class}`NonlinearProblem<dolfinx.fem.petsc.NonlinearProblem>` has additional support for blocked systems,
such as {py:attr}`NEST<petsc4py.PETSc.Mat.Type.NEST>` by supplying `kind="nest"` to its intializer. See the documentation for further
information.


### IO

#### VTKHDF5

**Authors**: Chris Richardson and 

As Kitware has stated that [VTKHDF](https://www.kitware.com/vtk-hdf-reader/) is the future format they want to support,
we have started the transistion to this format.
Currently, the following features have been implemented:
- Reading meshes: {py:func}`dolfinx.io.vtkhdf.read_mesh`. Supports mixed topology.
- Writing meshes: {py:func}`dolfinx.io.vtkhdf.write_mesh`. Supports mixed topology.
- Writing point data {py:func}`dolfinx.io.vtkhdf.write_point_data`.
  The point data should have the same ordering as the geometry nodes of the mesh.
- Writing cell data {py:func}`dolfinx.io.vtkhdf.write_cell_data`.

#### Remove Fides backend
As we unfortunately haven't seen an expanding set of features for the 
[Fides Reader](https://fides.readthedocs.io/en/latest/paraview/paraview.html)
in Paraview, we have decided to remove it from DOLFINx.
