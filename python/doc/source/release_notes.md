# Release notes

## v0.10.0

Since the 0.9.0 release, there has been 311 merged pull requests from 25 contributors.
Below follows a summary of the biggest changes to the Python-API from these pull requests.
In addition to the changes below, the ever-lasting quest of improving performance and squasing bugs continues.

### PETSc API

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd), [Francesco Ballarin](https://github.com/francesco-ballarin),
[Jack Hale](https://github.com/jhale) and [Garth N. Wells](https://github.com/garth-wells)

Mapping data between {py:class}`PETSc.Vec<petsc4py.PETSc.Vec` and {py:class}`dolfinx.fem.Function`s is now
trivial for blocked problems by using {py:func}`dolfinx.fem.petsc.assign`. 

Both solvers and assembly routines interfacing with PETSc has recieved a drastic make-over to
improve useability and maintenance, both for developers and end-users

#### Improved non-linear (Newton) solver

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

#### Improved {py:class}`dolfinx.fem.petsc.LinearProblem`

- The {py:class}`dolfinx.fem.petsc.LinearProblem` now support blocked problems, either specified manually or by using
  {py:func}`ufl.extract_blocks`. By changing the input-argument `kind`, the user can now decide if they want to use
  the DOLFINx blocked PETSc implementation (`kind="mpi"`) or the `kind=`{py:attr}`"nest"<petsc4py.PETSc.Mat.Type.NEST>`.
- The default behavior for non-blocked systems remains the same as before.
- The users can now also specify a (blocked) form for preconditioning through the `P` keyword argument in the constructor.

#### Assembly routines

In earlier versions of DOLFINx, there were three assembly routines for {py:class}`PETSc.Vec<petsc4py.PETSc.Vec>` and {py:class}`PETSc.Mat<petsc4py.PETSc.Mat>`:
- `assemble_*`
- `assemble_*_block`
- `assemble_*_nest`

This caused alot of duplicate logic in codes.
Therefore, we have unified all these assembly routines under {py:func}`dolfinx.fem.petsc.assemble_vector` and {py:func}`dolfinx.fem.petsc.assemble_matrix`.
The input keyword argument `kind` selects the relevant assembler routine.
See for instance the [Stokes demo](./demos/demo_stokes) for a detailed introduction.
Similar changes has been done to {py:func}`dolfinx.fem.petsc.apply_lifting`.

#### Linear algebra submodule

There is now a sub-module ({py:mod}`dolfinx.la.petsc`) containing PETSc LA operations.

#### Interpolation

The {py:func}`dolfinx.fem.discrete_curl` operator has been added to DOLFINx, to cater to
[Hypre Auxiliary-space Divergence Solver](https://hypre.readthedocs.io/en/latest/solvers-ads.html)
- A {py:class}`petsc4py.PETSc.Mat` equivalent can be found under {py:func}`dolfinx.fem.petsc.discrete_curl`.

### Simplified demos

#### Usage of {py:class}`ufl.MixedFunctionSpace` and {py:func}`ufl.extract_blocks`

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd) and [Joe Dean](https://github.com/jpdean/)

Initially introduced as part of the [v0.9.0-release](https://fenicsproject.org/blog/v0.9.0/#extract-blocks),
usage of these two UFL-abstractions hasve been propagated into the demos, to make it even easier for users to
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

### Form compiler and integral types
**Author**: [Susanne Claus](https://github.com/sclaus2), [Paul T. Kühner](https://github.com/schnellerhase),
and [Jørgen S. Dokken](https://github.com/jorgensd)
- The tabulation kernels now have an extra input, a `void*`, to make it possible to pass custom data for custom kernels.
- New {py:class}`dolfinx.fem.IntegralType` support
  - Vertex integrals: {py:obj}`ufl.dP`
  - Ridge integrals (codim=2); {py:obj}`uf.dr`
- One can now assemble the diagonal of a bilinear form into a vector by adding `form_compiler_options={"part":"diagonal"}`
  when calling {py:func}`dolfinx.fem.form`. Instead of calling {py:func}`dolfinx.fem.petsc.assemble_matrix` one should now call
  {py:func}`dolfinx.fem.petsc.assemble_vector`. This is useful for matrix-free solvers with Jacobi smoothing.

### Mesh

**Authors**: [Paul T. Kühner](https://github.com/schnellerhase), [Joe Dean](https://github.com/jpdean/),
[Garth N. Wells](https://github.com/garth-wells), [Jørgen S. Dokken](https://github.com/jorgensd) and [Chris Richardson](https://github.com/chrisrichardson)

- Uniform mesh refinement of all {py:class}`CellTypes<dolfinx.mesh.CellType>` is available through
{py:func}`dolfinx.mesh.uniform_refine`.
- Branching meshes (a mesh where a single facet is connected to more than two cells), such as T-joints (3 cells connected to a single facet) are now supported as input meshes to DOLFINx. To ensure proper
  partitioning in parallel, one should change the default option `max_facet_to_cell_links` to how many cells a facet
  can be attached to in {py:meth}`dolfinx.io.XDMFFile.read_mesh`, {py:func}`dolfinx.io.vtkhd.read_mesh` and
  {py:func}`dolfinx.mesh.create_mesh`.
- One can no longer use `set_connectivity` or `set_index_map` to modify {py:class}`dolfinx.mesh.Topology`
  objects. Any connectivity that is not `(tdim, 0)`, (`tdim`, `tdim`) or `(0, 0)` should be created with
 {py:meth}`dolfinx.mesh.Topology.create_connectivity`. The aforementioned connections should be attached
 to the topology when calling {py:func}`dolfinx.cpp.mesh.create_topology`.
- Mixed-dimensional support has been vastly improved by creating {py:class}`dolfinx.mesh.EntityMap`,
  which replaces the numpy arrays used as `entity_maps` in {py:func}`dolfinx.fem.form` in the previous release.
  This is a two-way map, meaning that the user no longer has to take care of creating the correct mapping.
  The two-way map from a sub-mesh to a parent mesh is returned as part of {py:func}`dolfinx.mesh.create_submesh`.


### Linear Algebra

**Authors**: [Chris Richardson](https://github.com/chrisrichardson)

The native {py:class}`matrix-format<dolfinx.la.MatrixCSR>` now has a sparse matrix-vector multiplication
{py:meth}`dolfinx.la.MatrixCSR.mult`. Note that the {py:class}`dolfinx.la.Vector` that you multiply with should use the
{py:attr}`dolfinx.la.MatrixCSR.index_map(1)<dolfinx.la.MatrixCSR.index_map>` rather than the one stemming from the
{py:meth}`dolfinx.fem.FunctionSpace.dofmap.index_map<dolfinx.fem.DofMap.index_map>`.


### Collision detection
**Author**: [Chris Richardson](https://github.com/chrisrichardson)
The collision detection algorithm {py:func}`dolfinx.geometry.compute_distance_gjk` now used multiprecision to ensure
proper collision detection. The algorithm has also been improve to work on co-planar convex hulls.

### Documentation
**Authors**: [Paul T. Kühner](https://github.com/schnellerhase), [Garth N. Wells](https://github.com/garth-wells),
 [Mehdi Slimani](https://github.com/ordinary-slim) and [Jørgen S. Dokken](https://github.com/jorgensd)
- Several classes that have only been exposed through the {ref}`dolfinx_cpp_interface` have gotten proper
  Python classes and functions. This includes:
  - {py:class}`dolfinx.graph.AdjacencyList`
  - {py:class}`dolfinx.fem.FiniteElement`
  - {py:func}`dolfinx.geometry.determine_point_ownership`
- Tons of typos, formatting fixes and improvements have been made.
- Usage of [intersphinx](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) and
  [sphinx-codeautolink](https://sphinx-codeautolink.readthedocs.io) to make the documentation more interactive.
  Most classes, functions and methods in any demo on the webpage can now redirect you to the relevant package API.


### Revised timer logic

**Authors**: [Garth N. Wells](https://github.com/garth-wells) and [Paul T. Kühner](https://github.com/schnellerhase)

Instead of using the [Boost timer library](https://www.boost.org/doc/libs/1_89_0/libs/timer/doc/index.html),
we have opted for the standard timing library [std::chrono](https://en.cppreference.com/w/cpp/header/chrono.html).
The switch is mainly due to some observed unaccuracies in timings with Boost.
This removes the notion of wall, system and user time.
See or {py:class}`Timer<dolfinx.common.Timer>` for examples of usage.


### IO

#### GMSH

**Authors**: [Paul T. Kühner](https://github.com/schnellerhase),  [Jørgen S. Dokken](https://github.com/jorgensd),
[Henrik N.T. Finsberg](https://github.com/finsberg) and [Pierric Mora](https://github.com/pierricmora)

The GMSH interface to DOLFINx has received a major upgrade.
- An **API**-breaking change is that the module `dolfinx.io.gmshio` has been renamed to {py:mod}`dolfinx.io.gmsh`.
- Another API-breaking change is the return type of {py:func}`dolfinx.io.gmshio.model_to_mesh` and
  {py:func}`dolfinx.io.read_from_msh`. Instead of returning the {py:class}`dolfinx.mesh.Mesh`, cell and facet
  {py:class}`dolfinx.mesh.MeshTags`, it now returns a {py:class}`dolfinx.io.gmsh.MeshData` data-class,
  that can contain {py:class}`dolfinx.mesh.MeshTags` of an sub-entity:
    - Cell (codim 0)
    - Facet (codim 1)
    - Ridge (codim 2)
    - Peak (codim 3)
- Additional checks and error handing for `Physical` tags from GMSH has been added to improve the user experience.

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

#### VTXWriter

**Authors**:  [Mehdi Slimani](https://github.com/ordinary-slim) and [Jørgen S. Dokken](https://github.com/jorgensd)

The writer does now support time-dependent DG-0 data, which can be written in the same file as a set of functions
from another (unique) function space.

#### XDMF

**Author**: [Massimilliano Leoni](https://github.com/mleoni-pf) and [Paul T. Kühner](https://github.com/schnellerhase)
- When using {py:meth}`dolfinx.io.XDMFFIle.read_meshtags` one can now specify the attribute name, if the grid has
multiple tags assigned to it. 
- Flushing data to file is now possible with {py:meth}`dolfinx.io.XDMFFile.flush`. This is useful when wanting to visualize
  long-running jobs in Paraview.

#### Remove Fides backend
As we unfortunately haven't seen an expanding set of features for the 
[Fides Reader](https://fides.readthedocs.io/en/latest/paraview/paraview.html)
in Paraview, we have decided to remove it from DOLFINx.

#### Pyvista

Pyvista no longer requires {py:func}`pyvista.start_xvfb` if one has installed `vtk` with OSMesa support.

