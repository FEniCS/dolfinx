Change log
==========

dev
---

- Remove OpenMpAsssmebler
- Remove MPI communicator as argument in GenericVector::init functions
  (communicator should be passed via constructor)
- Remove ``Function::operator[+-*/]`` to prevent memory corruption problems
  (does not affect Python interface)
- Add ``PETScSNESSolver`` constructor accepting both communicator and type
- Expression("f[0]*f[1]", f=obj) notation now supported for non-scalar
  GenericFunction obj
- Expression("f", f=obj) notation now supports obj of MeshFunction types
  (only cell based)
- Fix MPI deadlock in case of instant compilation failure

2016.2.0 [2016-11-30]
---------------------

- Updates to XDMFFile interface, now fully supporting MeshFunction and
  MeshValueCollection with multiple named datasets in one file (useful for
  volume/boundary markers). Time series now only when a time is explicitly
  specified for each step. Full support for ASCII/XML XDMF.
- Improved X3DOM support
- Improved detection of UFC
- Add CMake option `-DDOLFIN_USE_PYTHON3` to create a Python 3 build
- Require CMake version 3.5 or higher
- Add pylit to generate demo doc from rst
- More careful checks of Dirichlet BC function spaces
- Change definition of FunctionSpace::component()
- Adaptive solving now works for tensor-valued unknowns
- Improve logging of PETSc errors; details logged at level TRACE

2016.1.0 [2016-06-23]
---------------------
- Remove support for 'uint'-valued MeshFunction (replaced by 'size_t')
- Major performance improvements and simplifications of the XDMF IO.
- Remove Zoltan graph partitioning interface
- Add new algorithm for computing mesh entiites. Typical speed-up of
  two with gcc and four with clang. Reduced memory usage for meshes
  with irregularly numbered cells.
- Remove STLMatrix, STLVector, MUMPSLUSolver and PastixLUSolver
  classes
- Remove PETScPreconditioner::set_near_nullspace and add
  PETScMatrix::set_near_nullspace
- Build system updates for VTK 7.0
- Remove XDMF from File interface. XDMF is XML based, and has many
  possibilities for file access, which are not accessible through the
  limited File interface and "<<" ">>" operators. Instead of File, use
  XDMFFile, and use XDMFFile.read() and XDMFFile.write() for
  I/O. Demos and tests have been updated to show usage.  XDMF now also
  supports ASCII I/O in serial, useful for compatibility with users
  who do not have the HDF5 library available.
- Require polynomial degree or finite element for Expressions in the
  Python interface (fixes Issue #355,
  https://bitbucket.org/fenics-project/dolfin/issues/355)
- Switch to Google Test framwork for C++ unit tests
- Fix bug when reading domain data from mesh file for a ghosted mesh
- Add interface for manipulating mesh geometry using (higher-order) FE
  functions: free functions set_coordinates, get_coordinates,
  create_mesh
- Fix bug when reading domain data from mesh file for a ghosted mesh.
- Remove reference versions of constructors for many classes that
  store a pointer/reference to the object passed to the
  constructor. This is an intrusive interface change for C++ users,
  but necessary to improve code maintainabilty and to improve memory
  safety. The Python interface is (virtually) unaffected.
- Remove class SubSpace. Using FunctionSpace::sub(...) instead
- Remove reference versions constructors of NonlinearVariationalSolver
- Remove setting of bounds from NonlinearVariationalSolver (was
  already available through NonlinearVariationalProblem)
- Update Trilinos support to include Amesos2, and better support from
  Python
- Rewrite interface of TensorLayout and SparsityPattern;
  local-to-global maps now handled using new IndexMap class;
  GenericSparsityPattern class removed
- Remove QT (was an optional dependency)
- PETScTAOSolver::solve() now returns a pair of number of iterations
  (std::size_t) and whether iteration converged (bool)
- Better quality refinement in 2D in Plaza algorithm, by choosing
  refinement pattern based on max/min edge ratio
- Removed refine_cell() method in CellTypes
- Enable marker refinement to work in parallel for 1D meshes too
- Add std::abort to Python exception hook to avoid parallel deadlocks
- Extend dof_to_vertex_map with unowned dofs, thus making
  dof_to_vertex_map an inverse of vertex_to_dof_map
- Clean-up in PyDOLFIN function space design, issue #576
- Deprecate MixedFunctionSpace and EnrichedFunctionSpace in favour of
  initialization by suitable UFL element
- Add experimental matplotlib-based plotting backend, see mplot demo
- Remove method argument of DirichletBC::get_boundary_values()
- Change return types of free functions adapt() to shared_ptr

1.6.0 [2015-07-28]
------------------
- Remove redundant pressure boundary condition in Stokes demos
- Require Point in RectangleMesh and BoxMesh constructors
- Remove BinaryFile (TimeSeries now requires HDF5)
- Add (highly experimental) support for Tpetra matrices and vectors
  from Trilinos, interfacing to Belos, Amesos2, IfPack2 and Muelu.
- Enable (highly experimental) support for Quadrilateral and
  Hexahedral meshes, including some I/O, but no assembly yet.
- Enable UMFPACK and CHOLMOD solvers with Eigen backend
- Add an MPI_Comm to logger, currently defaulted to MPI_COMM_WORLD
  allowing better control over output in parallel
- Experimental output of quadratic geometry in XDMF files, allows more
  exact visualisation of P2 Functions
- Remove GenericMatrix::compressed (see Issue #61)
- Deprecate and PETScKryloveSolver::set_nullspace() and add
  PETScMatrix::set_nullspace()
- Remove uBLAS backend
- Remove UmfpackLUSolver and CholmodSolver
- Add EigenMatrix/Vector::data()
- Remove GenericMatrix/Vector::data() and GenericMatrix/Vector::data()
  (to use backends that support data(), cast first to backend type,
  e.g.  A = A.as_backend_type()
- Remove cmake.local, replaced by fenics-install-component.sh
- Make interior facet integrals define - and + cells ordered by
  cell_domains value.
- Remove deprecated arguments *_domains from assemble() and Form().
- Change measure definition notation from dx[mesh_function] to
  dx(subdomain_data=mesh_function).
- Set locale to "C" before reading from file
- Change GenericDofMap::cell_dofs return type from const
  std::vector<..>& to ArrayView<const ..>
- Add ArrayView class for views into arrays
- Change fall back linear algebra backend to Eigen
- Add Eigen linear algebra backend
- Remove deprecated GenericDofMap::geometric_dim function (fixes Issue
  #443)
- Add quadrature rules for multimesh/cut-cell integration up to order
  6
- Implement MPI reductions and XML ouput of Table class
- list_timings() is now collective and returns MPI average across
  processes
- Add dump_timings_to_xml()
- Add enum TimingType { wall, user, system } for selecting wall-clock,
  user and system time in timing routines
- Bump required SWIG version to 3.0.3
- Increase default maximum iterations in NewtonSolver to 50.
- Deprecate Python free function homogenize(bc) in favour of member
  function DirichletBC::homogenize()

1.5.0 [2015-01-12]
------------------
- DG demos working in parallel
- Simplify re-use of LU factorisations
- CMake 3 compatibility
- Make underlying SLEPc object accessible
- Full support for linear algebra backends with 64-bit integers
- Add smoothed aggregation AMG elasticity demo
- Add support for slepc4py
- Some self-assignment fixes in mesh data structures
- Deprecated GenericDofMap::geometric_dimension()
- Experimental support for ghosted meshes (overlapping region in
  parallel)
- Significant memory reduction in dofmap storage
- Re-write dofmap construction with significant performance and
  scaling improvements in parallel
- Switch to local (process-wise) indexing for dof indices
- Support local (process-wise) indexing in linear algerbra backends
- Added support for PETSc 3.5, require version >= 3.3
- Exposed DofMap::tabulate_local_to_global_dofs,
  MeshEntity::sharing_processes in Python
- Added GenericDofmap::local_dimension("all"|"owned"|"unowned")
- Added access to SLEPc or slepc4py EPS object of SLEPcEigenSolver
  (requires slepc4py version >= 3.5.1)
- LinearOperator can now be accessed using petsc4py
- Add interface (PETScTAOSolver) for the PETSc nonlinear
  (bound-constrained) optimisation solver (TAO)
- Add GenericMatrix::nnz() function to return number of nonzero
  entries in matrix (fixes #110)
- Add smoothed aggregation algerbraic multigrid demo for elasticity
- Add argument 'function' to project, to store the result into a
  preallocated function
- Remove CGAL dependency and mesh generation, now provided by mshr
- Python 2.7 required
- Add experimental Python 3 support. Need swig version 3.0.3 or later
- Move to py.test, speed up unit tests and make tests more robust in
  parallel
- Repeated initialization of PETScMatrix is now an error
- MPI interface change: num_processes -> size, process_number -> rank
- Add optional argument project(..., function=f), to avoid superfluous
  allocation
- Remove excessive printing of points during extrapolation
- Clean up DG demos by dropping restrictions of Constants: c('+') -> c
- Fix systemassembler warning when a and L both provide the same
  subdomain data.
- Require mesh instead of cell argument to FacetArea, FacetNormal,
  CellSize, CellVolume, SpatialCoordinate, Circumradius,
  MinFacetEdgeLength, MaxFacetEdgeLength
- Remove argument reset_sparsity to assemble()
- Simplify assemble() and Form() signature: remove arguments mesh,
  coefficients, function_spaces, common_cell. These are now all found
  by inspecting the UFL form
- Speed up assembly of forms with multiple integrals depending on
  different functions, e.g. f*dx(1) + g*dx(2).
- Handle accessing of GenericVectors using numpy arrays in python
  layer instead of in hard-to-maintain C++ layer
- Add support for mpi groups in jit-compilation
- Make access to HDFAttributes more dict like
- Add 1st and 2nd order Rush Larsen schemes for the
  PointIntegralSolver
- Add vertex assembler for PointIntegrals
- Add support for assembly of custom_integral
- Add support for multimesh assembly, function spaces, dofmaps and
  functions
- Fix to Cell-Point collision detection to prevent Points inside the
  mesh from falling between Cells due to rounding errors
- Enable reordering of cells and vertices in parallel via SCOTCH and
  the Giibs-Poole-Stockmeyer algorithm
- Efficiency improvements in dof assignment in parallel, working on
  HPC up to 24000 cores
- Introduction of PlazaRefinement methods based on refinement of the
  Mesh skeleton, giving better quality refinement in 3D in parallel
- Basic support for 'ghost cells' allowing integration over interior
  facets in parallel

1.4.0 [2014-06-02]
------------------
- Feature: Add set_diagonal (with GenericVector) to GenericMatrix
- Fix many bugs associated with cell orientations on manifolds
- Force all global dofs to be ordered last and to be on the last
  process in parallel
- Speed up dof reordering of mixed space including global dofs by
  removing the latter from graph reordering
- Force all dofs on a shared facet to be owned by the same process
- Add FEniCS ('fenics') Python module, identical with DOLFIN Python
  module
- Add function Form::set_some_coefficients()
- Remove Boost.MPI dependency
- Change GenericMatrix::compresss to return a new matrix (7be3a29)
- Add function GenericTensor::empty()
- Deprecate resizing of linear algebra via the GenericFoo interfaces
  (fixes #213)
- Deprecate MPI::process_number() in favour of MPI::rank(MPI_Comm)
- Use PETSc built-in reference counting to manage lifetime of wrapped
  PETSc objects
- Remove random access function from MeshEntityIterator (fixes #178)
- Add support for VTK 6 (fixes #149)
- Use MPI communicator in interfaces. Permits the creation of
  distributed and local objects, e.g. Meshes.
- Reduce memory usage and increase speed of mesh topology computation

1.3.0 [2014-01-07]
------------------
- Feature: Enable assignment of sparse MeshValueCollections to
  MeshFunctions
- Feature: Add free function assign that is used for sub function
  assignment
- Feature: Add class FunctionAssigner that cache dofs for sub function
  assignment
- Fix runtime dependency on checking swig version
- Deprecate DofMap member methods vertex_to_dof_map and
  dof_to_vertex_map
- Add free functions: vertex_to_dof_map and dof_to_vertex_map, and
  correct the ordering of the map.
- Introduce CompiledSubDomain a more robust version of
  compiled_subdomains, which is now deprecated
- CMake now takes care of calling the correct generate-foo script if
  so needed.
- Feature: Add new built-in computational geometry library
  (BoundingBoxTree)
- Feature: Add support for setting name and label to an Expression
  when constructed
- Feature: Add support for passing a scalar GenericFunction as default
  value to a CompiledExpression
- Feature: Add support for distance queries for 3-D meshes
- Feature: Add PointIntegralSolver, which uses the MultiStageSchemes
  to solve local ODEs at Vertices
- Feature: Add RKSolver and MultiStageScheme for general time integral
  solvers
- Feature: Add support for assigning a Function with linear
  combinations of Functions, which lives in the same FunctionSpace
- Added Python wrapper for SystemAssembler
- Added a demo using compiled_extension_module with separate source
  files
- Fixes for NumPy 1.7
- Remove DOLFIN wrapper code (moved to FFC)
- Add set_options_prefix to PETScKrylovSolver
- Remove base class BoundarCondition
- Set block size for PETScMatrix when available from TensorLayout
- Add support to get block compressed format from STLMatrix
- Add detection of block structures in the dofmap for vector equations
- Expose PETSc GAMG parameters
- Modify SystemAssembler to support separate assembly of A and b

1.2.0 [2013-03-24]
------------------
- Fixes bug where child/parent hierarchy in Python were destroyed
- Add utility script dolfin-get-demos
- MeshFunctions in python now support iterable protocol
- Add timed VTK output for Mesh and MeshFunction in addtion to
  Functions
- Expose ufc::dofmap::tabulate_entity_dofs to GenericDofMap interface
- Expose ufc::dofmap::num_entity_dofs to GenericDofMap interface
- Allow setting of row dof coordinates in preconditioners (only works
  with PETSc backed for now)
- Expose more PETSc/ML parameters
- Improve speed to tabulating coordinates in some DofMap functions
- Feature: Add support for passing a Constant as default value to a
  CompiledExpression
- Fix bug in dimension check for 1-D ALE
- Remove some redundant graph code
- Improvements in speed of parallel dual graph builder
- Fix bug in XMDF output for cell-based Functions
- Fixes for latest version of clang compiler
- LocalSolver class added to efficiently solve cell-wise problems
- New implementation of periodic boundary conditions. Now incorporated
  into the dofmap
- Optional arguments to assemblers removed
- SymmetricAssembler removed
- Domains for assemblers can now only be attached to forms
- SubMesh can now be constructed without a CellFunction argument, if
  the MeshDomain contains marked celldomains.
- MeshDomains are propagated to a SubMesh during construction
- Simplify generation of a MeshFunction from MeshDomains: No need to
  call mesh_function with mesh
- Rename dolfin-config.cmake to DOLFINConfig.cmake
- Use CMake to configure JIT compilation of extension modules
- Feature: Add vertex_to_dof_map to DofMap, which map vertex indices
  to dolfin dofs
- Feature: Add support for solving on m dimensional meshes embedded in
  n >= m dimensions

1.1.0 [2013-01-08]
------------------
- Add support for solving singular problems with Krylov solvers (PETSc
  only)
- Add new typedef dolfin::la_index for consistent indexing with linear
  algebra backends.
- Change default unsigned integer type to std::size_t
- Add support to attaching operator null space to preconditioner
  (required for smoothed aggregation AMG)
- Add basic interface to the PETSc AMG preconditioner
- Make SCOTCH default graph partitioner (GNU-compatible free license,
  unlike ParMETIS)
- Add scalable construction of mesh dual graph for mesh partitioning
- Improve performance of mesh building in parallel
- Add mesh output to SVG
- Add support for Facet and cell markers to mesh converted from
  Diffpack
- Add support for Facet and cell markers/attributes to mesh converted
  from Triangle
- Change interface for auto-adaptive solvers: these now take the goal
  functional as a constructor argument
- Add memory usage monitor: monitor_memory_usage()
- Compare mesh hash in interpolate_vertex_values
- Add hash() for Mesh and MeshTopology
- Expose GenericVector::operator{+=,-=,+,-}(double) to Python
- Add function Function::compute_vertex_values not needing a mesh
  argument
- Add support for XDMF and HDF5
- Add new interface LinearOperator for matrix-free linear systems
- Remove MTL4 linear algebra backend
- Rename down_cast --> as_type in C++ / as_backend_type in Python
- Remove KrylovMatrix interface
- Remove quadrature classes
- JIT compiled C++ code can now include a dolfin namespace
- Expression string parsing now understand C++ namespace such as
  std::cosh
- Fix bug in Expression so one can pass min, max
- Fix bug in SystemAssembler, where mesh.init(D-1, D) was not called
  before assemble
- Fix bug where the reference count of Py_None was not increased
- Fix bug in reading TimeSeries of size smaller than 3
- Improve code design for Mesh FooIterators to avoid dubious down cast
- Bug fix in destruction of PETSc user preconditioners
- Add CellVolume(mesh) convenience wrapper to Python interface for UFL
  function
- Fix bug in producing outward pointing normals of BoundaryMesh
- Fix bug introduced by SWIG 2.0.5, where typemaps of templated
  typedefs are not handled correctly
- Fix bug introduced by SWIG 2.0.5, which treated uint as Python long
- Add check that sample points for TimeSeries are monotone
- Fix handling of parameter "report" in Krylov solvers
- Add new linear algebra backend "PETScCusp" for GPU-accelerated
  linear algebra
- Add sparray method in the Python interface of GenericMatrix,
  requires scipy.sparse
- Make methods that return a view of contiguous c-arrays, via a NumPy
  array, keep a reference from the object so it wont get out of scope
- Add parameter: "use_petsc_signal_handler", which enables/disable
  PETSc system signals
- Avoid unnecessary resize of result vector for A*b
- MPI functionality for distributing values between neighbours
- SystemAssembler now works in parallel with topological/geometric
  boundary search
- New symmetric assembler with ability for stand-alone RHS assemble
- Major speed-up of DirichletBC computation and mesh marking
- Major speed-up of assembly of functions and expressions
- Major speed-up of mesh topology computation
- Add simple 2D and 3D mesh generation (via CGAL)
- Add creation of mesh from triangulations of points (via CGAL)
- Split the SWIG interface into six combined modules instead of one
- Add has_foo to easy check what solver and preconditioners are
  available
- Add convenience functions for listing available
  linear_algebra_backends
- Change naming convention for cpp unit tests test.cpp -> Foo.cpp
- Added cpp unit test for GenericVector::operator{-,+,*,/}= for all la
  backends
- Add functionality for rotating meshes
- Add mesh generation based on NETGEN constructive solid geometry
- Generalize SparsityPattern and STLMatrix to support column-wise
  storage
- Add interfaces to wrap PaStiX and MUMPS direct solvers
- Add CoordinateMatrix class
- Make STLMatrix work in parallel
- Remove all tr1::tuple and use boost::tuple
- Fix wrong link in Python quick reference.

1.0.0 [2011-12-07]
------------------
- Change return value of IntervalCell::facet_area() 0.0 --> 1.0.
- Recompile all forms with FFC 1.0.0
- Fix for CGAL 3.9 on OS X
- Improve docstrings for Box and Rectangle
- Check number of dofs on local patch in extrapolation

1.0-rc2 [2011-11-28]
--------------------
- Fix bug in 1D mesh refinement
- Fix bug in handling of subdirectories for TimeSeries
- Fix logic behind vector assignment, especially in parallel

1.0-rc1 [2011-11-21]
--------------------
- 33 bugs fixed
- Implement traversal of bounding box trees for all codimensions
- Edit and improve all error messages
- Added [un]equality operator to FunctionSpace
- Remove batch compilation of Expression (Expressions) from Python
  interface
- Added get_value to MeshValueCollection
- Added assignment operator to MeshValueCollection

1.0-beta2 [2011-10-26]
----------------------
- Change search path of parameter file to
  ~/.fenics/dolfin_parameters.xml
- Add functions Parameters::has_parameter,
  Parameters::has_parameter_set
- Added option to store all connectivities in a mesh for TimeSeries
  (false by default)
- Added option for gzip compressed binary files for TimeSeries
- Propagate global parameters to Krylov and LU solvers
- Fix OpenMp assemble of scalars
- Make OpenMP assemble over sub domains work
- DirichletBC.get_boundary_values, FunctionSpace.collapse now return a
  dict in Python
- Changed name of has_la_backend to has_linear_algebra_backend
- Added has_foo functions which can be used instead of the HAS_FOO
  defines
- Less trict check on kwargs for compiled Expression
- Add option to not right-justify tables
- Rename summary --> list_timings
- Add function list_linear_solver_methods
- Add function list_lu_solver_methods
- Add function list_krylov_solver_methods
- Add function list_krylov_solver_preconditioners
- Support subdomains in SystemAssembler (not for interior facet
  integrals)
- Add option functionality apply("flush") to PETScMatrix
- Add option finalize_tensor=true to assemble functions
- Solver parameters can now be passed to solve
- Remove deprecated function Variable::disp()
- Remove deprecated function logging()
- Add new class MeshValueCollection
- Add new class MeshDomains replacing old storage of boundary markers
  as part of MeshData. The following names are no longer supported:
  - boundary_facet_cells
  - boundary_facet_numbers
  - boundary_indicators
  - material_indicators
  - cell_domains
  - interior_facet_domains
  - exterior_facet_domains
- Rename XML tag <meshfunction> --> <mesh_function>
- Rename SubMesh data "global_vertex_indices" -->
  "parent_vertex_indices"
- Get XML input/output of boundary markers working again
- Get FacetArea working again

1.0-beta [2011-08-11]
---------------------
- Print percentage of non-zero entries when computing sparsity
  patterns
- Use ufl.Real for Constant in Python interface
- Add Dirichlet boundary condition argument to Python project function
- Add remove functionality for parameter sets
- Added out typemap for vector of shared_ptr objects
- Fix typemap bug for list of shared_ptr objects
- Support parallel XML vector io
- Add support for gzipped XML output
- Use pugixml for XML output
- Move XML SAX parser to libxml2 SAX2 interface
- Simplify XML io
- Change interface for variational problems, class VariationalProblem
  removed
- Add solve interface: solve(a == L), solve(F == 0)
- Add new classes Linear/NonlinearVariationalProblem
- Add new classes Linear/NonlinearVariationalSolver
- Ad form class aliases ResidualForm and Jacobian form in wrapper code
- Default argument to variables in Expression are passed as kwargs in
  the Python interface
- Add has_openmp as utility function in Python interface
- Add improved error reporting using dolfin_error
- Use Boost to compute Legendre polynolials
- Remove ode code
- Handle parsing of unrecognized command-line parameters
- All const std::vector<foo>& now return a read-only NumPy array
- Make a robust macro for generating a NumPy array from data
- Exposing low level fem functionality to Python, by adding a Cell ->
  ufc::cell typemap
- Added ufl_cell as a method to Mesh in Python interface
- Fix memory leak in Zoltan interface
- Remove some 'new' for arrays in favour of std::vector
- Added cell as an optional argument to Constant
- Prevent the use of non contiguous NumPy arrays for most typemaps
- Point can now be used to evaluate a Function or Expression in Python
- Fixed dimension check for Function and Expression eval in Python
- Fix compressed VTK output for tensors in 2D

0.9.11 [2011-05-16]
-------------------
- Change license from LGPL v2.1 to LGPL v3 or later
- Moved meshconverter to dolfin_utils
- Add support for conversion of material markers for Gmsh meshes
- Add support for point sources (class PointSource)
- Rename logging --> set_log_active
- Add parameter "clear_on_write" to TimeSeries
- Add support for input/output of nested parameter sets
- Check for dimensions in linear solvers
- Add support for automated error control for variational problems
- Add support for refinement of MeshFunctions after mesh refinement
- Change order of test and trial spaces in Form constructors
- Make SWIG version >= 2.0 a requirement
- Recognize subdomain data in Assembler from both Form and Mesh
- Add storage for subdomains (cell_domains etc) in Form class
- Rename MeshData "boundary facet cells" --> "boundary_facet_cells"
- Rename MeshData "boundary facet numbers" -->
  "boundary_facet_numbers"
- Rename MeshData "boundary indicators" --> "boundary_indicators"
- Rename MeshData "exterior facet domains" -->
  "exterior_facet_domains"
- Updates for UFC 2.0.1
- Add FiniteElement::evaluate_basis_derivatives_all
- Add support for VTK output of facet-based MeshFunctions
- Change default log level from PROGRESS to INFO
- Add copy functions to FiniteElement and DofMap
- Simplify DofMap
- Interpolate vector values when reading from time series

0.9.10 [2011-02-23]
-------------------
- Updates for UFC 2.0.0
- Handle TimeSeries stored backward in time (automatic reversal)
- Automatic storage of hierarchy during refinement
- Remove directory/library 'main', merged into 'common'
- dolfin_init --> init, dolfin_set_precision --> set_precision
- Remove need for mesh argument to functional assembly when possible
- Add function set_output_stream
- Add operator () for evaluation at points for Function/Expression in
  C++
- Add abs() to GenericVector interface
- Fix bug for local refinement of manifolds
- Interface change: VariationalProblem now takes: a, L or F, (dF)
- Map linear algebra objects to processes consistently with mesh
  partition
- Lots of improvemenst to parallel assembly, dof maps and linear
  algebra
- Add lists supported_elements and supported_elements_for_plotting in
  Python
- Add script dolfin-plot for plotting meshes and elements from the
  command-line
- Add support for plotting elements from Python
- Add experimental OpenMP assembler
- Thread-safe fixed in Function class
- Make GenericFunction::eval thread-safe (Data class removed)
- Optimize and speedup topology computation (mesh.init())
- Add function Mesh::clean() for cleaning out auxilliary topology data
- Improve speed and accuracy of timers
- Fix bug in 3D uniform mesh refinement
- Add built-in meshes UnitTriangle and UnitTetrahedron
- Only create output directories when they don't exist
- Make it impossible to set the linear algebra backend to something
  illegal
- Overload value_shape instead of dim for userdefined Python
  Expressions
- Permit unset parameters
- Search only for BLAS library (not cblas.h)

0.9.9 [2010-09-01]
------------------
- Change build system to CMake
- Add named MeshFunctions: VertexFunction, EdgeFunction, FaceFunction,
  FacetFunction, CellFunction
- Allow setting constant boundary conditions directly without using
  Constant
- Allow setting boundary conditions based on string ("x[0] == 0.0")
- Create missing directories if specified as part of file names
- Allow re-use of preconditioners for most backends
- Fixes for UMFPACK solver on some 32 bit machines
- Provide access to more Hypre preconditioners via PETSc
- Updates for SLEPc 3.1
- Improve and implement re-use of LU factorizations for all backends
- Fix bug in refinement of MeshFunctions

0.9.8 [2010-07-01]
------------------
- Optimize and improve StabilityAnalysis.
- Use own implementation of binary search in ODESolution (takes
  advantage of previous values as initial guess)
- Improve reading ODESolution spanning multiple files
- Dramatic speedup of progress bar (and algorithms using it)
- Fix bug in writing meshes embedded higher dimensions to M-files
- Zero vector in uBLASVector::resize() to fix spurious bug in Krylov
  solver
- Handle named fields (u.rename()) in VTK output
- Bug fix in computation of FacetArea for tetrahedrons
- Add support for direct plotting of Dirichlet boundary conditions:
  plot(bc)
- Updates for PETSc 3.1
- Add relaxation parameter to NewtonSolver
- Implement collapse of renumbered dof maps (serial and parallel)
- Simplification of DofMapBuilder for parallel dof maps
- Improve and simplify DofMap
- Add Armadillo dependency for dense linear algebra
- Remove LAPACKFoo wrappers
- Add abstract base class GenericDofMap
- Zero small values in VTK output to avoid VTK crashes
- Handle MeshFunction/markers in homogenize bc
- Make preconditioner selectable in VariationalProblem (new parameter)
- Read/write meshes in binary format
- Add parameter "use_ident" in DirichletBC
- Issue error by default when solvers don't converge (parameter
  "error_on_convergence")
- Add option to print matrix/vector for a VariationalProblem
- Trilinos backend now works in parallel
- Remove Mesh refine members functions. Use free refine(...) functions
  instead
- Remove AdapativeObjects
- Add Stokes demo using the MINI element
- Interface change: operator+ now used to denote enriched function
  spaces
- Interface change: operator+ --> operator* for mixed elements
- Add option 'allow_extrapolation' useful when interpolating to
  refined meshes
- Add SpatialCoordinates demo
- Add functionality for accessing time series sample times:
  vector_times(), mesh_times()
- Add functionality for snapping mesh to curved boundaries during
  refinement
- Add functionality for smoothing the boundary of a mesh
- Speedup assembly over exterior facets by not using BoundaryMesh
- Mesh refinement improvements, remove unecessary copying in Python
  interface
- Clean PETSc and Epetra Krylov solvers
- Add separate preconditioner classes for PETSc and Epetra solvers
- Add function ident_zeros for inserting one on diagonal for zero rows
- Add LU support for Trilinos interface

0.9.7 [2010-02-17]
------------------
- Add support for specifying facet orientation in assembly over
  interior facets
- Allow user to choose which LU package PETScLUSolver uses
- Add computation of intersection between arbitrary mesh entities
- Random access to MeshEntitiyIterators
- Modify SWIG flags to prevent leak when using SWIG director feature
- Fix memory leak in std::vector<Foo*> typemaps
- Add interface for SCOTCH for parallel mesh partitioning
- Bug fix in SubDomain::mark, fixes bug in DirichletBC based on
  SubDomain::inside
- Improvements in time series class, recognizing old stored values
- Add FacetCell class useful in algorithms iterating over boundary
  facets
- Rename reconstruct --> extrapolate
- Remove GTS dependency

0.9.6 [2010-02-03]
------------------
- Simplify access to form compiler parameters, now integrated with
  global parameters
- Add DofMap member function to return set of dofs
- Fix memory leak in the LA interface
- Do not import cos, sin, exp from NumPy to avoid clash with UFL
  functions
- Fix bug in MTL4Vector assignment
- Remove sandbox (moved to separate repository)
- Remove matrix factory (dolfin/mf)
- Update .ufl files for changes in UFL
- Added swig/import/foo.i for easy type importing from dolfin modules
- Allow optional argument cell when creating Expression
- Change name of Expression argument cpparg --> cppcode
- Add simple constructor (dim0, dim1) for C++ matrix Expressions
- Add example demonstrating the use of cpparg (C++ code in Python)
- Add least squares solver for dense systems (wrapper for DGELS)
- New linear algebra wrappers for LAPACK matrices and vectors
- Experimental support for reconstruction of higher order functions
- Modified interface for eval() and inside() in C++ using Array
- Introduce new Array class for simplified wrapping of arrays in SWIG
- Improved functionality for intersection detection
- Re-implementation of intersection detection using CGAL

0.9.5 [2009-12-03]
------------------
- Set appropriate parameters for symmetric eigenvalue problems with
  SLEPc
- Fix for performance regression in recent uBLAS releases
- Simplify Expression interface: f = Expression("sin(x[0])")
- Simplify Constant interface: c = Constant(1.0)
- Fix bug in periodic boundary conditions
- Add simple script dolfin-tetgen for generating DOLFIN XML meshes
  from STL
- Make XML parser append/overwrite parameter set when reading
  parameters from file
- Refinement of function spaces and automatic interpolation of member
  functions
- Allow setting global parameters for Krylov solver
- Fix handling of Constants in Python interface to avoid repeated JIT
  compilation
- Allow simple specification of subdomains in Python without needing
  to subclass SubDomain
- Add function homogenize() for simple creation of homogeneous BCs
  from given BCs
- Add copy constructor and possibility to change value for DirichletBC
- Add simple wrapper for ufl.cell.n. FacetNormal(mesh) now works again
  in Python.
- Support apply(A), apply(b) and apply(b, x) in PeriodicBC
- Enable setting spectral transformation for SLEPc eigenvalue solver

0.9.4 [2009-10-12]
------------------
- Remove set, get and operator() methods from MeshFunction
- Added const and none const T &operator[uint/MeshEntity] to
  MeshFunction
- More clean up in SWIG interface files, remove global renames and
  ignores
- Update Python interface to Expression, with extended tests for value
  ranks
- Removed DiscreteFunction class
- Require value_shape and geometric_dimension in Expression
- Introduce new class Expression replacing user-defined Functions
- interpolate_vertex_values --> compute_vertex_values
- std::map<std::string, Coefficient> replaces generated CoefficientSet
  code
- Cleanup logic in Function class as a result of new Expression class
- Introduce new Coefficient base class for form coefficients
- Replace CellSize::min,max by Mesh::hmin,hmax
- Use MUMPS instead of UMFPACK as default direct solver in both serial
  and parallel
- Fix bug in SystemAssembler
- Remove support for PETSc 2.3 and support PETSc 3.0.0 only
- Remove FacetNormal Function. Use UFL facet normal instead.
- Add update() function to FunctionSpace and DofMap for use in
  adaptive mesh refinement
- Require mesh in constructor of functionals (C++) or argument to
  assemble (Python)

0.9.3 [2009-09-25]
------------------
- Add global parameter "ffc_representation" for form representation in
  FFC JIT compiler
- Make norm() function handle both vectors and functions in Python
- Speedup periodic boundary conditions and make work for mixed
  (vector-valued) elements
- Add possibilities to use any number numpy array when assigning
  matrices and vectors
- Add possibilities to use any integer numpy array for indices in
  matrices and vectors
- Fix for int typemaps in PyDOLFIN
- Split mult into mult and transpmult
- Filter out PETSc argument when parsing command-line parameters
- Extend comments to SWIG interface files
- Add copyright statements to SWIG interface files (not finished yet)
- Add typemaps for misc std::vector<types> in PyDOLFIN
- Remove dependencies on std_vector.i reducing SWIG wrapper code size
- Use relative %includes in dolfin.i
- Changed names on SWIG interface files dolfin_foo.i -> foo.i
- Add function interpolate() in Python interface
- Fix typmaps for uint in python 2.6
- Use TypeError instead of ValueError in typechecks in typmaps.i
- Add in/out shared_ptr<Epetra_FEFoo> typemaps for PyDOLFIN
- Fix JIT compiling in parallel
- Add a compile_extension_module function in PyDOLFIN
- Fix bug in Python vector assignment
- Add support for compressed base64 encoded VTK files (using zlib)
- Add support for base64 encoded VTK files
- Experimental support for parallel assembly and solve
- Bug fix in project() function, update to UFL syntax
- Remove disp() functions and replace by info(foo, true)
- Add fem unit test (Python)
- Clean up SystemAssembler
- Enable assemble_system through PyDOLFIN
- Add 'norm' to GenericMatrix
- Efficiency improvements in NewtonSolver
- Rename NewtonSolver::get_iteration() to NewtonSolver::iteration()
- Improvements to EpetraKrylovSolver::solve
- Add constructor Vector::Vector(const GenericVector& x)
- Remove SCons deprecation warnings
- Memory leak fix in PETScKrylovSolver
- Rename dolfin_assert -> assert and use C++ version
- Fix debug/optimise flags
- Remove AvgMeshSize, InvMeshSize, InvFacetArea from SpecialFunctions
- Rename MeshSize -> CellSize
- Rewrite parameter system with improved support for command-line
  parsing, localization of parameters (per class) and usability from
  Python
- Remove OutflowFacet from SpecialFunctions
- Rename interpolate(double*) --> interpolate_vertex_values(double*)
- Add Python version of Cahn-Hilliard demo
- Fix bug in assemble.py
- Permit interpolation of functions between non-matching meshes
- Remove Function::Function(std::string filename)
- Transition to new XML io
- Remove GenericSparsityPattern::sort
- Require sorted/unsorted parameter in SparsityPattern constructor
- Improve performance of SparsityPattern::insert
- Replace enums with strings for linear algebra and built-in meshes
- Allow direct access to Constant value
- Initialize entities in MeshEntity constructor automatically and
  check range
- Add unit tests to the memorycheck
- Add call to clean up libxml2 parser at exit
- Remove unecessary arguments in DofMap member functions
- Remove reference constructors from DofMap, FiniteElement and
  FunctionSpace
- Use a shared_ptr to store the mesh in DofMap objects
- Interface change for wrapper code: PoissonBilinearForm -->
  Poisson::BilinearForm
- Add function info_underline() for writing underlined messages
- Rename message() --> info() for "compatibility" with Python logging
  module
- Add elementwise multiplication in GeneriVector interface
- GenericVector interface in PyDOLFIN now support the sequence
  protocol
- Rename of camelCaps functions names: fooBar --> foo_bar Note:
  mesh.numVertices() --> mesh.num_vertices(), mesh.numCells() -->
  mesh.num_cells()
- Add slicing capabilities for GenericMatrix interface in PyDOLFIN
  (only getitem)
- Add slicing capabilities for GenericVector interface in PyDOLFIN
- Add sum to GenericVector interface

0.9.2 [2009-04-07]
------------------
- Enable setting parameters for Newton solver in VariationalProblem
- Simplified and improved implementation of C++ plotting, calling
  Viper on command-line
- Remove precompiled elements and projections
- Automatically interpolate user-defined functions on assignment
- Add new built-in function MeshCoordinates, useful in ALE simulations
- Add new constructor to Function class, Function(V, "vector.xml")
- Remove class Array (using std::vector instead)
- Add vector_mapping data to MeshData
- Use std::vector instead of Array in MeshData
- Add assignment operator and copy constructor for MeshFunction
- Add function mesh.move(other_mesh) for moving mesh according to
  matching mesh (for FSI)
- Add function mesh.move(u) for moving mesh according to displacement
  function (for FSI)
- Add macro dolfin_not_implemented()
- Add new interpolate() function for interpolation of user-defined
  function to discrete
- Make _function_space protected in Function
- Added access to crs data from python for uBLAS and MTL4 backend

0.9.1 [2009-02-17]
------------------
- Check Rectangle and Box for non-zero dimensions
- ODE solvers now solve the dual problem
- New class SubMesh for simple extraction of matching meshes for sub
  domains
- Improvements of multiprecision ODE solver
- Fix Function class copy constructor
- Bug fixes for errornorm(), updates for new interface
- Interface update for MeshData: createMeshFunction -->
  create_mesh_function etc
- Interface update for Rectangle and Box
- Add elastodynamics demo
- Fix memory leak in IntersectionDetector/GTSInterface
- Add check for swig version, in jit and compile functions
- Bug fix in dolfin-order script for gzipped files
- Make shared_ptr work across C++/Python interface
- Replace std::tr1::shared_ptr with boost::shared_ptr
- Bug fix in transfinite mean-value interpolation
- Less annoying progress bar (silent when progress is fast)
- Fix assignment operator for MeshData
- Improved adaptive mesh refinement (recursive Rivara) producing
  better quality meshes

0.9.0 [2009-01-05]
------------------
- Cross-platform fixes
- PETScMatrix::copy fix
- Some Trilinos fixes
- Improvements in MeshData class
- Do not use initial guess in Newton solver
- Change OutflowFacet to IsOutflowFacet and change syntax
- Used shared_ptr for underling linear algebra objects
- Cache subspaces in FunctionSpace
- Improved plotting, now support plot(grad(u)), plot(div(u)) etc
- Simple handling of JIT-compiled functions
- Sign change (bug fix) in increment for Newton solver
- New class VariationalProblem replacing LinearPDE and NonlinearPDE
- Parallel parsing and partitioning of meshes (experimental)
- Add script dolfin-order for ordering mesh files
- Add new class SubSpace (replacing SubSystem)
- Add new class FunctionSpace
- Complete redesign of Function class hierarchy, now a single Function
  class
- Increased use of shared_ptr in Function, FunctionSpace, etc
- New interface for boundary conditions, form not necessary
- Allow simple setting of coefficient functions based on names (not
  their index)
- Don't order mesh automatically, meshes must now be ordered
  explicitly
- Simpler definition of user-defined functions (constructors not
  necessary)
- Make mesh iterators const to allow for const-correct Mesh code

0.8.1 [2008-10-20]
------------------
- Add option to use ML multigrid preconditioner through PETSc
- Interface change for ODE solvers: uBLASVector --> double*
- Remove homotopy solver
- Remove typedef real, now using plain double instead
- Add various operators -=, += to GenericMatrix
- Don't use -Werror when compiling SWIG generated code
- Remove init(n) and init(m, n) from GenericVector/Matrix. Use resize
  and zero instead
- Add new function is_combatible() for checking compatibility of
  boundary conditions
- Use x as initial guess in Krylov solvers (PETSc, uBLAS, ITL)
- Add new function errornorm()
- Add harmonic ALE mesh smoothing
- Refinements of Graph class
- Add CholmodCholeskySlover (direct solver for symmetric matrices)
- Implement application of Dirichlet boundary conditions within
  assembly loop
- Improve efficiency of SparsityPattern
- Allow a variable number of smoothings
- Add class Table for pretty-printing of tables
- Add experimental MTL4 linear algebra backend
- Add OutflowFacet to SpecialFunctions for DG transport problems
- Remove unmaintained OpenDX file format
- Fix problem with mesh smoothing near nonconvex corners
- Simple projection of functions in Python
- Add file format: XYZ for use with Xd3d
- Add built-in meshes: UnitCircle, Box, Rectangle, UnitSphere

0.8.0 [2008-06-23]
------------------
- Fix input of matrix data from XML
- Add function normalize()
- Integration with VMTK for reading DOLFIN XML meshes produced by VMTK
- Extend mesh XML format to handle boundary indicators
- Add support for attaching arbitrarily named data to meshes
- Add support for dynamically choosing the linear algebra backend
- Add Epetra/Trilinos linear solvers
- Add setrow() to matrix interface
- Add new solver SingularSolver for solving singular (pressure)
  systems
- Add MeshSize::min(), max() for easy computation of smallest/largest
  mesh size
- LinearSolver now handles all backends and linear solvers
- Add access to normal in Function, useful for inflow boundary
  conditions
- Remove GMRES and LU classes, use solve() instead
- Improve solve() function, now handles both LU and Krylov +
  preconditioners
- Add ALE mesh interpolation (moving mesh according to new boundary
  coordinates)

0.7.3 [2008-04-30]
------------------
- Add support for Epetra/Trilinos
- Bug fix for order of values in interpolate_vertex_values, now
  according to UFC
- Boundary meshes are now always oriented with respect to outward
  facet normals
- Improved linear algebra, both in C++ and Python
- Make periodic boundary conditions work in Python
- Fix saving of user-defined functions
- Improve plotting
- Simple computation of various norms of functions from Python
- Evaluation of Functions at arbitrary points in a mesh
- Fix bug in assembling over exterior facets (subdomains were ignored)
- Make progress bar less annoying
- New scons-based build system replaces autotools
- Fix bug when choosing iterative solver from Python

0.7.2 [2008-02-18]
------------------
- Improve sparsity pattern generator efficiency
- Dimension-independent sparsity pattern generator
- Add support for setting strong boundary values for DG elements
- Add option setting boundary conditions based on geometrical search
- Check UMFPACK return argument for warnings/errors
- Simplify setting simple Dirichlet boundary conditions
- Much improved integration with FFC in PyDOLFIN
- Caching of forms by JIT compiler now works
- Updates for UFC 1.1
- Catch exceptions in PyDOLFIN
- Work on linear algebra interfaces GenericTensor/Matrix/Vector
- Add linear algebra factory (backend) interface
- Add support for 1D meshes
- Make Assembler independent of linear algebra backend
- Add manager for handling sub systems (PETSc and MPI)
- Add parallel broadcast of Mesh and MeshFunction
- Add experimental support for parallel assembly
- Use PETSc MPI matrices when running in parallel
- Add predefined functions FacetNormal and AvgMeshSize
- Add left/right/crisscross options for UnitSquare
- Add more Python demos
- Add support for Exodus II format in dolfin-convert
- Autogenerate docstrings for PyDOLFIN
- Various small bug fixes and improvements

0.7.1 [2007-08-31]
------------------
- Integrate FFC form language into PyDOLFIN
- Just-in-time (JIT) compilation of variational forms
- Conversion from from Diffpack grid format to DOLFIN XML
- Name change: BoundaryCondition --> DirichletBC
- Add support for periodic boundary conditions: class PeriodicBC
- Redesign default linear algebra interface (Matrix, Vector,
  KrylovSolver, etc)
- Add function to return Vector associated with a DiscreteFunction

0.7.0-1 [2007-06-22]
--------------------
- Recompile all forms with latest FFC release
- Remove typedefs SparseMatrix and SparseVector
- Fix includes in LinearPDE
- Rename DofMaps -> DofMapSet

0.7.0 [2007-06-20]
------------------
- Move to UFC interface for code generation
- Major rewrite, restructure, cleanup
- Add support for Brezzi-Douglas-Marini (BDM) elements
- Add support for Raviart-Thomas (RT) elements
- Add support for Discontinuous Galerkin (DG) methods
- Add support for mesh partitioning (through SCOTCH)
- Handle both UMFPACK and UFSPARSE
- Local mesh refinement
- Mesh smoothing
- Built-in plotting (through Viper)
- Cleanup log system
- Numerous fixes for mesh, in particular MeshFunction
- Much improved Python bindings for mesh
- Fix Python interface for vertex and cell maps in boundary
  computation

0.6.4 [2006-12-01]
------------------
- Switch from Python Numeric to Python NumPy
- Improved mesh Python bindings
- Add input/output support for MeshFunction
- Change Mesh::vertices() --> Mesh::coordinates()
- Fix bug in output of mesh to MATLAB format
- Add plasticty module (experimental)
- Fix configure test for Python dev (patch from Åsmund Ødegård)
- Add mesh benchmark
- Fix memory leak in mesh (data not deleted correctly in MeshTopology)
- Fix detection of curses libraries
- Remove Tecplot output format

0.6.3 [2006-10-27]
------------------
- Move to new mesh library
- Remove dolfin-config and move to pkg-config
- Remove unused classes PArray, PList, Table, Tensor
- Visualization of 2D solutions in OpenDX is now supported (3D
  supported before)
- Add support for evaluation of functionals
- Fix bug in Vector::sum() for uBLAS vectors

0.6.2-1 [2006-09-06]
--------------------
- Fix compilation error when using --enable-petsc
  (dolfin::uBLASVector::PETScVector undefined)

0.6.2 [2006-09-05]
------------------
- Finish chapter in manual on linear algebra
- Enable PyDOLFIN by default, use --disable-pydolfin to disable
- Disable PETSc by default, use --enable-petsc to enable
- Modify ODE solver interface for u0() and f()
- Add class ConvectionMatrix
- Readd classes LoadVector, MassMatrix, StiffnessMatrix
- Add matrix factory for simple creation of standard finite element
  matrices
- Collect static solvers in LU and GMRES
- Bug fixes for Python interface PyDOLFIN
- Enable use of direct solver for ODE solver (experimental)
- Remove demo bistable
- Restructure and cleanup linear algebra
- Use UMFPACK for LU solver with uBLAS matrix types
- Add templated wrapper class for different uBLAS matrix types
- Add ILU preconditioning for uBLAS matrices
- Add Krylov solver for uBLAS sparse matrices (GMRES and BICGSTAB)
- Add first version of new mesh library (NewMesh, experimental)
- Add Parametrized::readParameters() to trigger reading of values on
  set()
- Remove output of zeros in Octave matrix file format
- Use uBLAS-based vector for Vector if PETSc disabled
- Add wrappers for uBLAS compressed_matrix class
- Compute eigenvalues using SLEPc (an extension of PETSc)
- Clean up assembly and linear algebra
- Add function to solve Ax = b for dense matrices and dense vectors
- Make it possible to compile without PETSc (--disable-petsc)
- Much improved ODE solvers
- Complete multi-adaptive benchmarks reaction and wave
- Assemble boundary integrals
- FEM class cleaned up.
- Fix multi-adaptive benchmark problem reaction
- Small fixes for Intel C++ compiler version 9.1
- Test for Intel C++ compiler and configure appropriately
- Add new classes DenseMatrix and DenseVector (wrappers for ublas)
- Fix bug in conversion from Gmsh format

0.6.1 [2006-03-28]
------------------
- Regenerate build system in makedist script
- Update for new FFC syntax: BasisFunction --> TestFunction,
  TrialFunction
- Fixes for conversion script dolfin-convert
- Initial cleanups and fixes for ODE solvers
- Numerous small fixes to improve portability
- Remove dolfin:: qualifier on output << in Parameter.h
- Don't use anonymous classes in demos, gives errors with some
  compilers
- Remove KrylovSolver::solver()
- Fix bug in convection-diffusion demo (boundary condition for
  pressure), use direct solver
- LinearPDE and NewonSolver use umfpack LU solver by default (if
  available) when doing direct solve
- Set PETSc matrix type through Matrix constructor
- Allow linear solver and preconditioner type to be passed to
  NewtonSolver
- Fix bug in Stokes demos (wrong boundary conditions)
- Cleanup Krylov solver
- Remove KrylovSolver::setPreconditioner() etc. and move to
  constructors
- Remove KrylovSolver::setRtol() etc. and replace with parameters
- Fix remaining name changes: noFoo() --> numFoo()
- Add Cahn-Hilliard equation demo
- NewtonSolver option to use residual or incremental convergence
  criterion
- Add separate function to nls to test for convergence of Newton
  iterations
- Fix bug in dolfin-config (wrong version number)

0.6.0 [2006-03-01]
------------------
- Fix bug in XML output format (writing multiple objects)
- Fix bug in XML matrix output format (handle zero rows)
- Add new nonlinear PDE demo
- Restructure PDE class to use envelope-letter design
- Add precompiled finite elements for q <= 5
- Add FiniteElementSpec and factor function for FiniteElement
- Add input/output of Function to DOLFIN XML
- Name change: dof --> node
- Name change: noFoo() --> numFoo()
- Add conversion from gmsh format in dolfin-convert script
- Updates for PETSc 2.3.1
- Add new type of Function (constant)
- Simplify use of Function class
- Add new demo Stokes + convection-diffusion
- Add new demo Stokes (equal-order stabilized)
- Add new demo Stokes (Taylor-Hood)
- Add new parameter for KrylovSolvers: "monitor convergence"
- Add conversion script dolfin-convert for various mesh formats
- Add new demo elasticity
- Move poisson demo to src/demo/pde/poisson
- Move to Mercurial (hg) from CVS
- Use libtool to build libraries (including shared)

0.5.12 [2006-01-12]
-------------------
- Make Stokes solver dimension independent (2D/3D)
- Make Poisson solver dimension independent (2D/3D)
- Fix sparse matrix output format for MATLAB
- Modify demo problem for Stokes, add exact solution and compute error
- Change interface for boundary conditions: operator() --> eval()
- Add two benchmark problems for the Navier-Stokes solver
- Add support for 2D/3D selection in Navier-Stokes solver
- Move tic()/toc() to timing.h
- Navier-Stokes solver back online
- Make Solver a subclass of Parametrized
- Add support for localization of parameters
- Redesign of parameter system

0.5.11 [2005-12-15]
-------------------
- Add script monitor for monitoring memory usage
- Remove meminfo.h (not portable)
- Remove dependence on parameter system in log system
- Don't use drand48() (not portable)
- Don't use strcasecmp() (not portable)
- Remove sysinfo.h and class System (not portable)
- Don't include <sys/utsname.h> (not portable)
- Change ::show() --> ::disp() everywhere
- Clean out old quadrature classes on triangles and tetrahedra
- Clean out old sparse matrix code
- Update chapter on Functions in manual
- Use std::map to store parameters
- Implement class KrylovSolver
- Name change: Node --> Vertex
- Add nonlinear solver demos
- Add support for picking sub functions and components of functions
- Update interface for FiniteElement for latest FFC version
- Improve and restructure implementation of the Function class
- Dynamically adjust safety factor during integration
- Improve output Matrix::disp()
- Check residual at end of time step, reject step if too large
- Implement Vector::sum()
- Implement nonlinear solver
- New option for ODE solver: "save final solution" --> solution.data
- New ODE test problem: reaction
- Fixes for automake 1.9 (nobase_include_HEADERS)
- Reorganize build system, remove fake install and require make
  install
- Add checks for non-standard PETSc component HYPRE in NSE solver
- Make GMRES solver return the number of iterations
- Add installation script for Python interface
- Add Matrix Market format (Haiko Etzel)
- Automatically reinitialize GMRES solver when system size changes
- Implement cout << for class Vector

0.5.10 [2005-10-11]
-------------------
- Modify ODE solver interface: add T to constructor
- Fix compilation on AMD 64 bit systems (add -fPIC)
- Add new BLAS mode for form evaluation
- Change enum types in File to lowercase
- Change default file type for .m to Octave
- Add experimental Python interface PyDOLFIN
- Fix compilation for gcc 4.0

0.5.9 [2005-09-23]
------------------
- Add Stokes module
- Support for arbitrary mixed elements through FFC
- VTK output interface now handles time-dependent functions
  automatically
- Fix cout for empty matrix
- Change dolfin_start() --> dolfin_end()
- Add chapters to manual: about, log system, parameters, reference
  elements, installation, contributing, license
- Use new template fenicsmanual.cls for manual
- Add compiler flag -U__STRICT_ANSI__ when compiling under Cygwin
- Add class EigenvalueSolver

0.5.8 [2005-07-05]
------------------
- Add new output format Paraview/VTK (Garth N. Wells)
- Update Tecplot interface
- Move to PETSc 2.3.0
- Complete support for general order Lagrange elements in triangles
  and tetrahedra
- Add test problem in src/demo/fem/convergence/ for general Lagrange
  elements
- Make FEM::assemble() estimate the number of nonzeros in each row
- Implement Matrix::init(M, N, nzmax)
- Add Matrix::nz(), Matrix::nzsum() and Matrix::nzmax()
- Improve Mesh::disp()
- Add FiniteElement::disp() and FEM::disp() (useful for debugging)
- Remove old class SparseMatrix
- Change FEM::setBC() --> FEM::applyBC()
- Change Mesh::tetrahedrons --> Mesh::tetrahedra
- Implement Dirichlet boundary conditions for tetrahedra
- Implement Face::contains(const Point& p)
- Add test for shape dimension of mesh and form in FEM::assemble()
- Move src/demo/fem/ demo to src/demo/fem/simple/
- Add README file in src/demo/poisson/ (simple manual)
- Add simple demo program src/demo/poisson/
- Update computation of alignment of faces to match FFC/FIAT

0.5.7 [2005-06-23]
------------------
- Clean up ODE test problems
- Implement automatic detection of sparsity pattern from given matrix
- Clean up homotopy solver
- Implement automatic computation of Jacobian
- Add support for assembly of non-square systems (Andy Terrel)
- Make ODE solver report average number of iterations
- Make progress bar write first update at 0%
- Initialize all values of u before solution in multi-adaptive solver,
  not only components given by dependencies
- Allow user to modify and verify a converging homotopy path
- Make homotopy solver save a list of the solutions
- Add Matrix::norm()
- Add new test problem for CES economy
- Remove cast from Parameter to const char* (use std::string)
- Make solution data filename optional for homotopy solver
- Append homotopy solution data to file during solution
- Add dolfin::seed(int) for optionally seeding random number generator
- Remove dolfin::max,min (use std::max,min)
- Add polynomial-integer (true polynomial) form of general CES system
- Compute multi-adaptive efficiency index
- Updates for gcc 4.0 (patches by Garth N. Wells)
- Add Matrix::mult(const real x[], uint row) (temporary fix, assumes
  uniprocessor case)
- Add Matrix::mult(const Vector& x, uint row) (temporary fix, assumes
  uniprocessor case)
- Update shortcuts MassMatrix and StiffnessMatrix to new system
- Add missing friend to Face.h (reported by Garth N. Wells)

0.5.6 [2005-05-17]
------------------
- Implementation of boundary conditions for general order Lagrange
  (experimental)
- Use interpolation function automatically generated by FFC
- Put computation of map into class AffineMap
- Clean up assembly
- Use dof maps automatically generated by FFC (experimental)
- Modify interface FiniteElement for new version of FFC
- Update ODE homotopy test problems
- Add cross product to class Point
- Sort mesh entities locally according to ordering used by FIAT and
  FFC
- Add new format for dof maps (preparation for higher-order elements)
- Code cleanups: NewFoo --> Foo complete
- Updates for new version of FFC (0.1.7)
- Bypass log system when finalizing PETSc (may be out of scope)

0.5.5 [2005-04-26]
------------------
- Fix broken log system, curses works again
- Much improved multi-adaptive time-stepping
- Move elasticity module to new system based on FFC
- Add boundary conditions for systems
- Improve regulation of time steps
- Clean out old assembly classes
- Clean out old form classes
- Remove kernel module map
- Remove kernel module element
- Move convection-diffusion module to new system based on FFC
- Add iterators for cell neighbors of edges and faces
- Implement polynomial for of CES economy
- Rename all new linear algebra classes: NewFoo --> Foo
- Clean out old linear algebra
- Speedup setting of boundary conditions (add MAT_KEEP_ZEROED_ROWS)
- Fix bug for option --disable-curses

0.5.4 [2005-03-29]
------------------
- Remove option to compile with PETSc 2.2.0 (2.2.1 required)
- Make make install work again (fix missing includes)
- Add support for mixing multiple finite elements (through FFC)
- Improve functionality of homotopy solver
- Simple creation of piecewise linear functions (without having an
  element)
- Simple creation of piecewise linear elements
- Add support of automatic creation of simple meshes (unit cube, unit
  square)

0.5.3 [2005-02-26]
------------------
- Change to PETSc version 2.2.1
- Add flag --with-petsc=<path> to configure script
- Move Poisson's equation to system based on FFC
- Add support for automatic creation of homotopies
- Make all ODE solvers automatically handle complex ODEs: (M) z' =
  f(z,t)
- Implement version of mono-adaptive solver for implicit ODEs: M u' =
  f(u,t)
- Implement Newton's method for multi- and mono-adaptive ODE solvers
- Update PETSc wrappers NewVector, NewMatrix, and NewGMRES
- Fix initialization of PETSc
- Add mono-adaptive cG(q) and dG(q) solvers (experimental)
- Implementation of new assebly: NewFEM, using output from FFC
- Add access to mesh for nodes, cells, faces and edges
- Add Tecplot I/O interface; contributed by Garth N. Wells

0.5.2 [2005-01-26]
------------------
- Benchmarks for DOLFIN vs PETSc (src/demo/form and src/demo/test)
- Complete rewrite of the multi-adaptive ODE solver (experimental)
- Add wrapper for PETSc GMRES solver
- Update class Point with new operators
- Complete rewrite of the multi-adaptive solver to improve performance
- Add PETSc wrappers NewMatrix and NewVector
- Add DOLFIN/PETSc benchmarks

0.5.1 [2004-11-10]
------------------
- Experimental support for automatic generation of forms using FFC
- Allow user to supply Jacobian to ODE solver
- Add optional test to check if a dependency already exists (Sparsity)
- Modify sparse matrix output (Matrix::show())
- Add FGMRES solver in new format (patch from eriksv)
- Add non-const version of quick-access of sparse matrices
- Add linear mappings for simple computation of derivatives
- Add check of matrix dimensions for ODE sparsity pattern
- Include missing cmath in Function.cpp

0.5.0 [2004-08-18]
------------------
- First prototype of new form evaluation system
- New classes Jacobi, SOR, Richardson (preconditioners and linear
  solvers)
- Add integrals on the boundary (ds), partly working
- Add maps from boundary of reference cell
- Add evaluation of map from reference cell
- New Matrix functions: max, min, norm, and sum of rows and columns
  (erik)
- Derivatives/gradients of ElementFunction (coefficients f.ex.)
  implemented
- Enable assignment to all elements of a NewArray
- Add functions Boundary::noNodes(), noFaces(), noEdges()
- New class GaussSeidel (preconditioner and linear solver)
- New classes Preconditioner and LinearSolver
- Bug fix for tetrahedral mesh refinement (ingelstrom)
- Add iterators for Edge and Face on Boundary
- Add functionality to Map: bdet() and cell()
- Add connectivity face-cell and edge-cell
- New interface for assembly: Galerkin --> FEM
- Bug fix for PDE systems of size > 3

0.4.11 [2004-04-23]
-------------------
- Add multigrid solver (experimental)
- Update manual

0.4.10
------
- Automatic model reduction (experimental)
- Fix bug in ParticleSystem (divide by mass)
- Improve control of integration (add function ODE::update())
- Load/save parameters in XML-format
- Add assembly test
- Add simple StiffnessMatrix, MassMatrix, and LoadVector
- Change dK --> dx
- Change dx() --> ddx()
- Add support for GiD file format
- Add performance tests for multi-adaptivity (both stiff and
  non-stiff)
- First version of Newton for the multi-adaptive solver
- Test for Newton for the multi-adaptive solver

0.4.9
-----
- Add multi-adaptive solver for the bistable equation
- Add BiCGSTAB solver (thsv)
- Fix bug in SOR (thsv)
- Improved visual program for OpenDX
- Fix OpenDX file format for scalar functions
- Allow access to samples of multi-adaptive solution
- New patch from thsv for gcc 3.4.0 and 3.5.0
- Make progress step a parameter
- New function ODE::sparse(const Matrix& A)
- Access nodes, cells, edges, faces by id
- New function Matrix::lump()

0.4.8
-----
- Add support for systems (jansson and bengzon)
- Add new module wave
- Add new module wave-vector
- Add new module elasticity
- Add new module elasticity-stationary
- Multi-adaptive updates
- Fix compilation error in LogStream
- Fix local Newton iteration for higher order elements
- Init matrix to given type
- Add output of cG(q) and dG(q) weights in matrix format
- Fix numbering of frames from plotslab script
- Add png output for plotslab script
- Add script for running stiff test problems, plot solutions
- Fix bug in MeshInit (node neighbors of node)
- Modify output of sysinfo()
- Compile with -Wall -Werror -pedantic -ansi -std=c++98 (thsv)

0.4.7
-----
- Make all stiff test problems work
- Display status report also when using step()
- Improve adaptive damping for stiff problems (remove spikes)
- Modify Octave/Matlab format for solution data (speed improvement)
- Adaptive sampling of solution (optional)
- Restructure stiff test problems
- Check if value of right-hand side is valid
- Modify divergence test in AdaptiveIterationLevel1

0.4.6
-----
- Save vectors and matrices from Matlab/Octave (foufas)
- Rename writexml.m to xmlmesh.m
- Inlining of important functions
- Optimize evaluation of elements
- Optimize Lagrange polynomials
- Optimize sparsity: use stl containers
- Optimize choice of discrete residual for multi-adaptive solver
- Don't save solution in benchmark proble
- Improve computation of divergence factor for underdamped systems
- Don't check residual on first slab for fixed time step
- Decrease largest (default) time step to 0.1
- Add missing <cmath> in TimeStepper
- Move real into dolfin namespace

0.4.5
-----
- Rename function.h to enable compilation under Cygwin
- Add new benchmark problem for multi-adaptive solver
- Bug fix for ParticleSystem
- Initialization of first time step
- Improve time step regulation (threshold)
- Improve stabilization
- Improve TimeStepper interface (Ko Project)
- Use iterators instead of recursively calling TimeSlab::update()
- Clean up ODESolver
- Add iterators for elements in time slabs and element groups
- Add -f to creation of symbolic links

0.4.4
-----
- Add support for 3D graphics in Octave using Open Inventor (jj)

0.4.3
-----
- Stabilization of multi-adaptive solver (experimental)
- Improved non-support for curses (--disable-curses)
- New class MechanicalSystem for simulating mechanical systems
- Save debug info from primal and dual (plotslab.m)
- Fix bug in progress bar
- Add missing include file in Components.h (kakr)
- New function dolfin_end(const char* msg, ...)
- Move numerical differentiation to RHS
- New class Event for limited display of messages
- Fix bug in LogStream (large numbers in floating point format)
- Specify individual time steps for different components
- Compile without warnings
- Add -Werror to option enable-debug
- Specify individual methods for different components
- Fix bug in dGqMethods
- Fix bug (delete old block) in ElementData
- Add parameters for method and order
- New test problem reaction
- New class FixedPointIteration
- Fix bug in grid refinement

0.4.2
-----
- Fix bug in computation of residual (divide by k)
- Add automatic generation and solution of the dual problem
- Automatic selection of file names for primal and dual
- Fix bug in progress bar (TerminalLogger)
- Many updates of multi-adaptive solver
- Add class ODEFunction
- Update function class hierarchies
- Move functions to a separate directory
- Store multi-adaptive solution binary on disk with cache

0.4.1
-----
- First version of multi-adaptive solver working
- Clean up file formats
- Start changing from int to unsigned int where necessary
- Fix bool->int when using stdard in Parameter
- Add NewArray and NewList (will replace Array and List)

0.4.0
-----
- Initiation of the FEniCS project
- Change syntax of mesh files: grid -> mesh
- Create symbolic links instead of copying files
- Tanganyika -> ODE
- Add Heat module
- Grid -> Mesh
- Move forms and mappings to separate libraries
- Fix missing include of DirectSolver.h

0.3.12
------
- Adaptive grid refinement (!)
- Add User Manual
- Add function dolfin_log() to turn logging on/off
- Change from pointers to references for Node, Cell, Edge, Face
- Update writexml.m
- Add new grid files and rename old grid files

0.3.11
------
- Add configure option --disable-curses
- Grid refinement updates
- Make OpenDX file format work for grids (output)
- Add volume() and diameter() in cell
- New classes TriGridRefinement and TetGridRefinement
- Add iterators for faces and edges on a boundary
- New class GridHierarchy

0.3.10
------
- Use new boundary structure in Galerkin
- Make dolfin_start() and dolfin_end() work
- Make dolfin_assert() raise segmentation fault for plain text mode
- Add configure option --enable-debug
- Use autoreconf instead of scripts/preconfigure
- Rename configure.in -> configure.ac
- New class FaceIterator
- New class Face
- Move computation of boundary from GridInit to BoundaryInit
- New class BoundaryData
- New class BoundaryInit
- New class Boundary
- Make InitGrid compute edges
- Add test program for generic matrix in src/demo/la
- Clean up Grid classes
- Add new class GridRefinementData
- Move data from Cell to GenericCell
- Make GMRES work with user defined matrix, only mult() needed
- GMRES now uses only one function to compute residual()
- Change Matrix structure (a modified envelope/letter)
- Update script checkerror.m for Poisson
- Add function dolfin_info_aptr()
- Add cast to element pointer for iterators
- Clean up and improve the Tensor class
- New class: List
- Name change: List -> Table
- Name change: ShortList -> Array
- Make functions in GridRefinement static
- Make functions in GridInit static
- Fix bug in GridInit (eriksv)
- Add output to OpenDX format for 3D grids
- Clean up ShortList class
- Clean up List class
- New class ODE, Equation replaced by PDE
- Add Lorenz test problem
- Add new problem type for ODEs
- Add new module ode
- Work on multi-adaptive ODE solver (lots of new stuff)
- Work on grid refinement
- Write all macros in LoggerMacros in one line
- Add transpose functions to Matrix (Erik)

0.3.9
-----
- Update Krylov solver (Erik, Johan)
- Add new LU factorization and LU solve (Niklas)
- Add benchmark test in src/demo/bench
- Add silent logger

0.3.8
-----
- Make sure dolfin-config is regenerated every time
- Add demo program for cG(q) and dG(q)
- Add dG(q) precalc of nodal points and weights
- Add cG(q) precalc of nodal points and weights
- Fix a bug in configure.in (AC_INIT with README)
- Add Lagrange polynomials
- Add multiplication with transpose
- Add scalar products with rows and columns
- Add A[i][j] index operator for quick access to dense matrix

0.3.7
-----
- Add new Matlab-like syntax like A(i,all) = x or A(3,all) = A(4,all)
- Add dolfin_assert() macro enabled if debug is defined
- Redesign of Matrix/DenseMatrix/SparseMatrix to use Matrix as common
  interface
- Include missing cmath in Legendre.cpp and GaussianQuadrature.cpp

0.3.6
-----
- Add output functionality in DenseMatrix
- Add high precision solver to DirectSolver
- Clean up error messages in Matrix
- Make solvers directly accessible through Matrix and DenseMatrix
- Add quadrature (Gauss, Radau, and Lobatto) from Tanganyika
- Start merge with Tanganyika
- Add support for automatic documentation using doxygen
- Update configure scripts
- Add greeting at end of compilation

0.3.5
-----
- Define version number only in the file configure.in
- Fix compilation problem (missing depcomp)

0.3.4
-----
- Fix bugs in some of the ElementFunction operators
- Make convection-diffusion solver work again
- Fix bug in integration, move multiplication with the determinant
- Fix memory leaks in ElementFunction
- Add parameter to choose output format
- Make OctaveFile and MatlabFile subclasses of MFile
- Add classes ScalarExpressionFunction and VectorExpressionFunction
- Make progress bars work cleaner
- Get ctrl-c in curses logger
- Remove <Problem>Settings-classes and use dolfin_parameter()
- Redesign settings to match the structure of the log system
- Add vector functions: Function::Vector
- Add vector element functions: ElementFunction::Vector

0.3.3
-----
- Increased functionality of curses-based interface
- Add progress bars to log system

0.3.2
-----
- More work on grid refinement
- Add new curses based log system

0.3.1
-----
- Makefile updates: make install should now work properly
- KrylovSolver updates
- Preparation for grid refinement
- Matrix and Vector updates

0.3.0
-----
- Make poisson work again, other modules still not working
- Add output format for octave
- Fix code to compile with g++-3.2 -Wall -Werror
- New operators for Matrix
- New and faster GMRES solver (speedup factor 4)
- Changed name from SparseMatrix to Matrix
- Remove old unused code
- Add subdirectory math containing mathematical functions
- Better access for A(i,j) += to improve speed in assembling
- Add benchmark for linear algebra
- New definition of finite element
- Add algebra for function spaces
- Convert grids in data/grids to xml.gz
- Add iterators for Nodes and Cells
- Change from .hh to .h
- Add operators to Vector class (foufas)
- Add dependence on libxml2
- Change from .C to .cpp to make Jim happy.
- Change input/output functionality to streams
- Change to new data structure for Grid
- Change to object-oriented API at top level
- Add use of C++ namespaces
- Complete and major restructuring of the code
- Fix compilation error in src/config
- Fix name of keyword for convection-diffusion

0.2.11-1
--------
- Fix compilation error (`source`) on Solaris

0.2.11
------
- Automate build process to simplify addition of new modules
- Fix bug in matlab_write_field() (walter)
- Fix bug in SparseMatrix::GetCopy() (foufas)

0.2.10-1
--------
- Fix compilation errors on RedHat (thsv)

0.2.10
------
- Fix compilation of problems to use correct compiler
- Change default test problems to the ones in the report
- Improve memory management using mpatrol for tracking allocations
- Change bool to int for va_arg, seems to be a problem with gcc > 3.0
- Improve input / output support: GiD, Matlab, OpenDX

0.2.8
-----
- Navier-Stokes starting to work again
- Add Navier-Stokes 2d
- Bug fixes

0.2.7
-----
- Add support for 2D problems
- Add module convection-diffusion
- Add local/global fields in equation/problem
- Bug fixes
- Navier-Stokes updates (still broken)

0.2.6 [2002-02-19]
------------------
- Navier-Stokes updates (still broken)
- Output to matlab format

0.2.5
-----
- Add variational formulation with overloaded operators for systems
- ShapeFunction/LocalField/FiniteElement according to Scott & Brenner

0.2.4
-----
- Add boundary conditions
- Poisson seems to work ok

0.2.3
-----
- Add GMRES solver
- Add CG solver
- Add direct solver
- Add Poisson solver
- Big changes to the organisation of the source tree
- Add kwdist.sh script
- Bug fixes

0.2.2:
------
- Remove curses temporarily

0.2.1:
------
- Remove all PETSc stuff. Finally!
- Gauss-Seidel cannot handle the pressure equation

0.2.0:
------
- First GPL release
- Remove all of Klas Samuelssons proprietary grid code
- Adaptivity and refinement broken, include in next release
