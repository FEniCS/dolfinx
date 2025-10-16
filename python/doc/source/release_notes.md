# Release notes

## v0.10.0

### Interpolation

**Author**:  [Garth N. Wells](https://github.com/garth-wells)

- The {py:func}`dolfinx.fem.discrete_curl` operator has been added to DOLFINx, to cater to
[Hypre Auxiliary-space Divergence Solver](https://hypre.readthedocs.io/en/latest/solvers-ads.html)
- A {py:class}`petsc4py.PETSc.Mat` equivalent can be found under {py:func}`dolfinx.fem.petsc.discrete_curl`.


### Simplified demos

#### Usage of {py:class}`ufl.MixedFunctionSpace` and {py:func}`ufl.extract_blocks`

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd) and [Joe Dean](https://github.com/jpdean/)

Initially introduced as part of the [v0.9.0-release](https://fenicsproject.org/blog/v0.9.0/#extract-blocks),
usage of these two UFL-abstractions hasve been propegated into the demos, to make it even easier for users to
see examples of how to work with blocked problems.

*TODO*: Add profiling of blocked/mixed-element vs mixedfunction-space.

#### Usage of {py:class}`ufl.ZeroBaseForm`

**Author**:  [Garth N. Wells](https://github.com/garth-wells)
For a long time, it has not been possible to specify the right hand side of a linear PDE as empty.
This means that users often have had to resolve to adding `dolfinx.fem.Constant(mesh, 0.0)*v*ufl.dx`
to ensure that one can use the dolfinx form compilation functions.
With the introduction of {py:class}`ufl.ZeroBaseForm` this is no longer required.
The aforementioned workaround can now be reduced to `ufl.ZeroBaseForm((v, ))`, which avoid extra
assembly calls within DOLFINx.

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

#### GMSH

**Authors**: [Paul T. Kühner](https://github.com/schnellerhase),  [Jørgen S. Dokken](https://github.com/jorgensd),
[Henrik N.T. Finsberg](https://github.com/finsberg)

The GMSH interface to DOLFINx has received a major upgrade.
An **API**-breaking change is that the module `dolfinx.io.gmshio` has been renamed to {py:mod}`dolfinx.io.gmsh`.
Another API-breaking change is the return type of {py:func}`dolfinx.io.gmshio.model_to_mesh` and
{py:func}`dolfinx.io.read_from_msh`. Instead of returning the {py:class}`dolfinx.mesh.Mesh`, cell and facet
{py:class}`dolfinx.mesh.MeshTags`, it now returns a {py:class}`dolfinx.io.gmsh.MeshData` data-class,
that can contain {py:class}`dolfinx.mesh.MeshTags` of an sub-entity:
- Cell (codim 0)
- Facet (codim 1)
- Ridge (codim 2)
- Peak (codim 3)


#### VTKHDF5

**Authors**: [Chris Richardson](https://github.com/chrisrichardson) and [Jørgen S. Dokken](https://github.com/jorgensd) 

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

#### Pyvista

Pyvista no longer requires {py:func}`pyvista.start_xvfb` if one has installed `vtk` with OSMesa support.

### Mesh

**Authors**:  [Garth N. Wells](https://github.com/garth-wells)

One can no longer use `set_connectivity` or `set_index_map` to modify {py:class}`dolfinx.mesh.Topology`
objects. Any connectivity that is not `(tdim, 0)`, (`tdim`, `tdim`) or `(0, 0)` should be created withed
{py:meth}`dolfinx.mesh.Topology.create_connectivity`. The aforementioned connections should be attached
to the topology when calling {py:func}`dolfinx.cpp.mesh.create_topology`.

Meshtag attribute name: https://github.com/FEniCS/dolfinx/pull/3257


### DirichletBC

Simplify {py:meth}`dolfinx.fem.DirichletBC.set` https://github.com/FEniCS/dolfinx/pull/3505

## Linear Algebra

**Authors**: [Chris Richardson](https://github.com/chrisrichardson)

The native {py:class}`matrix-format<dolfinx.la.MatrixCSR>` now has a sparse matrix-vector multiplication
{py:meth}`dolfinx.la.MatrixCSR.mult`. Note that the {py:class}`dolfinx.la.Vector` that you multiply with should use the
{py:attr}`dolfinx.la.MatrixCSR.index_map(1)<dolfinx.la.MatrixCSR.index_map>` rather than the one stemming from the
{py:meth}`dolfinx.fem.FunctionSpace.dofmap.index_map<dolfinx.fem.DofMap.index_map>`.


## Documentation
**Authors**: [Paul T. Kühner](https://github.com/schnellerhase), [Garth N. Wells](https://github.com/garth-wells)
and [Jørgen S. Dokken](https://github.com/jorgensd)
- Several classes that have only been exposed through the {ref}`dolfinx_cpp_interface` have gotten proper
  Python classes. This includes:
  - {py:class}`dolfinx.graph.AdjacencyList`
  - {py:class}`dolfinx.fem.FiniteElement`
- Tons of typos, formatting fixes and improvements have been made.
- Usage of [intersphinx](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) and
  [sphinx-codeautolink](https://sphinx-codeautolink.readthedocs.io) to make the documentation more interactive.
  Most classes, functions and methods in any demo on the webpage can now redirect you to the relevant package API.



# Progress
At commit: https://github.com/FEniCS/dolfinx/commit/54b4e6aa08eecfb0bae3844a3db13c01c21b3312