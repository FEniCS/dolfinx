# Release notes

## v0.11.0

Since the 0.10.0 release, there has been 177 merged pull requests from 27 contributors.
Below follows a summary of the biggest changes to the Python API.
A full diff can be found [here](https://github.com/FEniCS/dolfinx/compare/v0.10.0.post5...main).

### SuperLU_DIST interface

**Authors**: [Jack Hale](https://github.com/jhale) and [Chris Richardson](https://github.com/chrisrichardson)

After the support for Windows was added in v0.9.0, there has been a lack of parallel supported solvers to
solve the resulting linear problems. Furthermore, PETSc sometimes can feel complicated for simple problems presented
during teaching. In this release, we add support for using SuperLU_DIST with native sparse matrices
{py:class}`dolfinx.la.MatrixCSR`.

The following constructors and classes have been added:

- {py:func}`dolfinx.la.superlu_dist.superlu_dist_matrix`: Deep-copy all data from {py:class}`dolfinx.la.MatrixCSR` to
  SUPERLU_Dist matrix.
- {py:func}`dolfinx.la.superlu_dist.superlu_dist_solver`: Create a {py:class}`dolfinx.la.superlu_dist.SuperLUDistSolver`
  which you can use to call {py:meth}`set_option<dolfinx.la.superlu_dist.SuperLUDistSolver.set_option>`,
  {py:meth}`set_A<dolfinx.la.superlu_dist.SuperLUDistSolver.set_A>` or {py:meth}`solve<dolfinx.la.superlu_dist.SuperLUDistSolver.solve>`.
- {py:class}`dolfinx.fem.problems.LinearProblem`: An interface similar to the {py:class}`dolfinx.fem.petsc.LinearProblem`, i.e.
  it takes in {py:class}`ufl.Form` for the bilinear and linear forms, along with appropriate
  {py:class}`Dirichlet boundary conditions<dolfinx.fem.DirichletBC>`,
  solver options, and {py:class}`entity_maps<dolfinx.mesh.EntityMap>`.

### Built-in matrix support
  
**Authors**: [Chris Richardson](https://github.com/chrisrichardson)

- Adds {py:meth}`A.transpose()<dolfinx.la.MatrixCSR.transpose>`, {py:meth}`A.mult(x, y, transpose=True)<dolfinx.la.MatrixCSR.mult>`
  and {py:meth}`A.matmul(B)<dolfinx.la.MatrixCSR.matmul>` to the built- in matrices
- Templated matrices in the Python API for block size `[i, i], i=1,2,3`.
- Improved tests for square and rectangular matrices

### The 'real' element

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd) and [Matthew Scroggs](https://github.com/mscroggs)

The real-element has for a long time not been present in DOLFINx. This has mainly been due to the fact that the implementation
in legacy FEniCS introduced a ton of special casing within core functionality, which we thought was better to avoid in DOLFINx.
However, a prototype implementation of the real-element has been around for a few releases, and has now been implemented in the
core libraries. Users can now call

```python
import basix.ufl
import dolfinx.fem

el = basix.ufl.real_element(mesh.basix_cell(), dtype=dtype, value_shape=(N, ))
R = dolfinx.fem.functionspace(mesh, el)
```

to create a function space consisting of `N` values (of data type `dtype`, which can be a complex type).

Furthermore, to use this space alongside other spaces, for instance for Lagrange multipliers, users are
recommended to use {py:class}`ufl.MixedFunctionSpace(V, R, ...)<ufl.MixedFunctionSpace>` and
{py:func}`ufl.extract_blocks` to create blocked systems that can be used in {py:class}`dolfinx.fem.petsc.LinearProblem`
or {py:class}`dolfinx.fem.petsc.NonlinearProblem`.



### Threading

**Authors**: [Chris Richardson](https://github.com/chrisrichardson), [Jørgen S. Dokken](https://github.com/jorgensd)
and [Garth N. Wells](https://github.com/garth-wells)

For a long time, DOLFINx has been exclusively using MPI for the distribution of computational load.
However, with the computational landscape evolving to more and more heterogeneous systems,
the need for additional parallelisation methods are required. 
In this release, we introduce initial threading support using
[std::jthread`](https://en.cppreference.com/cpp/thread/jthread) in the following methods:
- {py:meth}`dolfinx.mesh.Topology.create_entities`
- {py:func}`dolfinx.geometry.compute_distances_gjk`
which both take an optional argument `num_threads` which specifies how many CPU threads should be used.
If set to 0, threads are not spawned.

### New and improved demos

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd), [Paul T. Kühner](https://github.com/schnellerhase)

A new demo showcasing how to use PETSc and matrix-free solvers can be found in
[demo_matrix-free-petsc](./demos/demo_matrix-free-petsc)

The [PML demo](./demos/demo_pml) now shows how to use one-sided interior facet
integrals with manual specification of integration entities.

The [Biharmonic demo](./demos/demo_biharmonic) has gone through a major revision, using:
- A more suitable choice of finite elements (as P2 yields sub-optimal convergence)
- Better choice of penalty parameter
- Change of boundary conditions from simply supported to clamped and explaining the effect of different BCs.
- Verify solution with the method of manufactured solutions and add relevant references

In general, demos now consistently use `tdim=mesh.topology.dim`, `fdim = tdim -1` and `gdim = mesh.geometry.dim` to
avoid confusion for new users.


### Further improvements in submesh support

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd)

A feature that for a long time has existed outside of the FEniCS core is the
{py:func}`dolfinx.mesh.transfer_meshtags_to_submesh`, which makes it possible to transfer a
meshtag from a parent mesh to a submesh. This function is now part of the core library.

Another feature introduced in this release is the usage of submesh quantities in {py:class}`dolfinx.fem.Expression`.
You can now pass {py:class}`Expressions<ufl.core.expr.Expr>` containing coefficients and constants from a submesh,
combined with geometric quantities, coefficients, and constants of the parent mesh.
For example

```python
V = fem.functionspace(mesh, el)
u = fem.Function(V, dtype=dtype)
# Populate `u` ...

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
exterior_facets = exterior_facet_indices(mesh.topology)
submesh, entity_map, _, _ = create_submesh(mesh, mesh.topology.dim - 1, exterior_facets)
u_sub = fem.Function(fem.functionspace(submesh, sub_el), dtype=dtype)
# Populate `u_sub` ...

quadrature_points, _ = basix.make_quadrature(basix.CellType.interval, qdegree)
quadrature_points = quadrature_points.astype(xtype)

n_h = ufl.FacetNormal(mesh)
f = u * n + u_sub * n

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
compiled_expr = fem.Expression(expr, quadrature_points, dtype=dtype, entity_maps=[entity_map])
```


### Extending GMSH and VTKHDF IO

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd)

With the [changes to mesh](#mesh) in v0.10.0 `max_facet_to_cell_links` was introduced to make
it possible to create meshes with joints, branches, etc.
This is now exposed in {py:func}`dolfinx.io.gmsh.model_to_mesh`.

Furthermore, new cell types are supported for the `vtkhdf` backend, including all
linear and quadratic VTK cell types.

The GMSH interface can now read in meshes on mixed-topology grids using {py:func}`dolfinx.io.gmsh.model_to_mesh`
or {py:func}`dolfinx.io.gmsh.read_from_msh`. There is currently no support for
reading {py:class}`entity tags<dolfinx.mesh.MeshTags>` on mixed-topology grids through this interface yet.


### Exposing tolerances for non-affine pull-backs

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd)

A set of users have had issues with non-affine geometries, in particular higher order grids,
and the tolerance and maximum number of iterations in the 
{py:meth}`dolfinx.fem.CoordinateElement.pull_back`, {py:meth}`dolfinx.fem.Function.interpolate_nonmatching` 
and {py:meth}`dolfinx.fem.Function.eval` yielding errors such as:
```bash
RuntimeError: Newton method failed to converge for non-affine geometry
```
If you see this error message, try to increase the `maxiter` or `tol`.


### Mixed topology meshes, prisms and pyramid cells
**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd) and [Chris Richardson](https://github.com/chrisrichardson)

For mixed topology meshes, there are many notions that do not align with the original design of DOLFINx.
Examples are the members `num_entity_dofs` and `num_entity_closure_dofs` of {py:class}`dolfinx.cpp.fem.ElementDofLayout`,
as prisms and pyramids do not have the same number of dofs per sub-entity.
They have been removed, and users should instead call
{py:meth}`len(ElementDofLayout.entity_dofs(dim, entity_index))<dolfinx.cpp.fem.ElementDofLayout.entity_dofs>`

Furthermore, {py:func}`dolfinx.fem.apply_lifting` and {py:func}`dolfinx.fem.assemble_scalar` now work for mixed-topology meshes.

### Meshes

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd) and [Jack Hale](https://github.com/jhale) 

- New function {py:func}`dolfinx.mesh.create_point_mesh` to create a point cloud mesh with no points shared between
  the different processes. Useful for reading in point measures or outputting data.
- Built in mesh-generators such as {py:func}`dolfinx.mesh.create_rectangle` now takes an optional
  argument `gdim` that embeds in a larger space. This simplifies the testing process for problems on manifolds.
- New function {py:func}`dolfinx.fem.interpolate_geometry` that allows users to create a new mesh either raising or lowering the
  polynomial order of the mesh geometry. The topology is shared between the old and new grid. It is also possible to
  switch the Lagrange-variant of the underlying coordinate element.

### Interpolation

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd) and [Garth N. Wells](https://github.com/garth-wells)

A crucial bug interpolating Piola-mapped elements from parent to a codim-0 submesh has been fixed for this release.


### Modernizing UFL

**Authors**: [Paul T. Kühner](https://github.com/schnellerhase)

UFL is a Python project that has been in development for almost 20 years, and
Python has gone through a massive modernization during this time.
One of the visually pleasing improvements is the use of `@property`-decorators.
{py:class}`ufl.AbstractCell` now uses properties for
{py:attr}`topological_dimension<ufl.AbstractCell.topological_dimension>` and
{py:attr}`cellname<ufl.AbstractCell.cellname>`, etc. while
{py:class}`ufl.Mesh` now has {py:attr}`geometric_dimension<ufl.Mesh.geometric_dimension>`.
See [UFL PR \#385](https://github.com/FEniCS/ufl/pull/385) for more details.



## v0.10.0

Since the 0.9.0 release, there have  been 311 merged pull requests from 25 contributors.
Below follows a summary of the biggest changes to the Python-API from these pull requests.
In addition to the changes below, the ever-lasting quest of improving performance and squashing bugs continues.

### PETSc API

**Authors**: [Jørgen S. Dokken](https://github.com/jorgensd), [Francesco Ballarin](https://github.com/francesco-ballarin),
[Jack Hale](https://github.com/jhale) and [Garth N. Wells](https://github.com/garth-wells)

Mapping data between {py:class}`PETSc.Vec<petsc4py.PETSc.Vec>` and {py:class}`dolfinx.fem.Function`s is now
trivial for blocked problems by using {py:func}`dolfinx.fem.petsc.assign`. 

Both solvers and assembly routines interfacing with PETSc has received a drastic make-over to
improve usability and maintenance, both for developers and end-users.

#### Improved non-linear (Newton) solver

The FEniCS project has for the last 15 years had its own implementation of a Newton solver.
We no longer see the need of providing this solver, as the {py:class}`PETSc SNES<petsc4py.PETSc.SNES>` solver,
and equivalent solver for C++ provides more features than our own implementation.

The previously shipped {py:class}`dolfinx.nls.petsc.NewtonSolver` is deprecated, in favor of
{py:class}`dolfinx.fem.petsc.NonlinearProblem`, which now integrates directly with {py:class}`petsc4py.PETSc.SNES`.

The non-linear problem object that was sent into {py:class}`dolfinx.nls.petsc.NewtonSolver` has been renamed
to {py:class}`NewtonSolverNonlinearProblem<dolfinx.fem.petsc.NewtonSolverNonlinearProblem>` and is also deprecated.

The new {py:class}`NonlinearProblem<dolfinx.fem.petsc.NonlinearProblem>` has additional support for blocked systems,
such as {py:attr}`NEST<petsc4py.PETSc.Mat.Type.NEST>` by supplying `kind="nest"` to its initializer. See the documentation for further
information.

#### Improved {py:class}`dolfinx.fem.petsc.LinearProblem`

- The {py:class}`dolfinx.fem.petsc.LinearProblem` now supports blocked problems, either specified manually or by using
  {py:func}`ufl.extract_blocks`. By changing the input-argument `kind`, the user can now decide if they want to use
  the DOLFINx blocked PETSc implementation (`kind="mpi"`) or the `kind=`{py:attr}`"nest"<petsc4py.PETSc.Mat.Type.NEST>`.
- The default behavior for non-blocked systems remains the same as before.
- The users can now also specify a (blocked) form for preconditioning through the `P` keyword argument in the constructor.

#### Assembly routines

In earlier versions of DOLFINx, there were three assembly routines for {py:class}`PETSc.Vec<petsc4py.PETSc.Vec>` and {py:class}`PETSc.Mat<petsc4py.PETSc.Mat>`:
- `assemble_*`
- `assemble_*_block`
- `assemble_*_nest`

This caused a lot of duplicate logic in codes.
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
usage of these two UFL-abstractions has been propagated into the demos, to make it even easier for users to
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
- Branching meshes (a mesh where a single facet is connected to more than two cells), such as T-joints (3 cells connected to a single facet)
  are now supported as input meshes to DOLFINx. To ensure proper partitioning in parallel, one should change the default
  option `max_facet_to_cell_links` to how many cells a facet
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
The switch is mainly due to some observed inaccuracies in timings with Boost.
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
we have started the transition to this format.
Currently, the following features have been implemented:
- Reading meshes: {py:func}`dolfinx.io.vtkhdf.read_mesh`. Supports mixed topology.
- Writing meshes: {py:func}`dolfinx.io.vtkhdf.write_mesh`. Supports mixed topology.
- Writing point data {py:func}`dolfinx.io.vtkhdf.write_point_data`.
  The point data should have the same ordering as the geometry nodes of the mesh.
- Writing cell data {py:func}`dolfinx.io.vtkhdf.write_cell_data`.

#### VTXWriter

**Authors**: [Mehdi Slimani](https://github.com/ordinary-slim) and [Jørgen S. Dokken](https://github.com/jorgensd)

The writer does now support time-dependent DG-0 data, which can be written in the same file as a set of functions
from another (unique) function space.

#### XDMF

**Author**: [Massimiliano Leoni](https://github.com/mleoni-pf) and [Paul T. Kühner](https://github.com/schnellerhase)
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

