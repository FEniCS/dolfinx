// Generated list of include files for PyDOLFIN

// DOLFIN headers included from common
%include "dolfin/swig/common_pre.i"
%include "dolfin/common/init.h"
%include "dolfin/common/types.h"
%include "dolfin/common/constants.h"
%include "dolfin/common/timing.h"
%include "dolfin/common/Array.h"
%include "dolfin/common/IndexSet.h"
%include "dolfin/common/Set.h"
%include "dolfin/common/Timer.h"
%include "dolfin/common/Variable.h"
%include "dolfin/common/Hierarchical.h"
%include "dolfin/common/MPI.h"
%include "dolfin/common/SubSystemsManager.h"
%include "dolfin/swig/common_post.i"

// DOLFIN headers included from parameter
%include "dolfin/swig/parameter_pre.i"
%include "dolfin/parameter/Parameter.h"
%include "dolfin/parameter/Parameters.h"
%include "dolfin/parameter/GlobalParameters.h"
%include "dolfin/swig/parameter_post.i"

// DOLFIN headers included from log
%include "dolfin/swig/log_pre.i"
%include "dolfin/log/log.h"
%include "dolfin/log/Event.h"
%include "dolfin/log/LogStream.h"
%include "dolfin/log/Progress.h"
%include "dolfin/log/Table.h"
%include "dolfin/log/LogLevel.h"
%include "dolfin/swig/log_post.i"

// DOLFIN headers included from la
%include "dolfin/swig/la_pre.i"
%include "dolfin/la/ublas.h"
%include "dolfin/la/GenericLinearSolver.h"
%include "dolfin/la/GenericLUSolver.h"
%include "dolfin/la/GenericTensor.h"
%include "dolfin/la/GenericMatrix.h"
%include "dolfin/la/GenericSparsityPattern.h"
%include "dolfin/la/GenericVector.h"
%include "dolfin/la/GenericSparsityPattern.h"
%include "dolfin/la/PETScObject.h"
%include "dolfin/la/PETScBaseMatrix.h"
%include "dolfin/la/uBLASFactory.h"
%include "dolfin/la/uBLASMatrix.h"
%include "dolfin/la/uBLASKrylovMatrix.h"
%include "dolfin/la/PETScMatrix.h"
%include "dolfin/la/PETScKrylovMatrix.h"
%include "dolfin/la/PETScPreconditioner.h"
%include "dolfin/la/EpetraLUSolver.h"
%include "dolfin/la/EpetraKrylovSolver.h"
%include "dolfin/la/EpetraMatrix.h"
%include "dolfin/la/EpetraVector.h"
%include "dolfin/la/PETScKrylovSolver.h"
%include "dolfin/la/PETScLUSolver.h"
%include "dolfin/la/CholmodCholeskySolver.h"
%include "dolfin/la/UmfpackLUSolver.h"
%include "dolfin/la/ITLKrylovSolver.h"
%include "dolfin/la/MTL4Matrix.h"
%include "dolfin/la/STLMatrix.h"
%include "dolfin/la/uBLASVector.h"
%include "dolfin/la/PETScVector.h"
%include "dolfin/la/MTL4Vector.h"
%include "dolfin/la/SparsityPattern.h"
%include "dolfin/la/EpetraSparsityPattern.h"
%include "dolfin/la/LinearAlgebraFactory.h"
%include "dolfin/la/DefaultFactory.h"
%include "dolfin/la/PETScUserPreconditioner.h"
%include "dolfin/la/PETScFactory.h"
%include "dolfin/la/EpetraFactory.h"
%include "dolfin/la/MTL4Factory.h"
%include "dolfin/la/STLFactory.h"
%include "dolfin/la/SLEPcEigenSolver.h"
%include "dolfin/la/TrilinosPreconditioner.h"
%include "dolfin/la/uBLASSparseMatrix.h"
%include "dolfin/la/uBLASDenseMatrix.h"
%include "dolfin/la/uBLASPreconditioner.h"
%include "dolfin/la/uBLASKrylovSolver.h"
%include "dolfin/la/uBLASILUPreconditioner.h"
%include "dolfin/la/Vector.h"
%include "dolfin/la/Matrix.h"
%include "dolfin/la/Scalar.h"
%include "dolfin/la/LinearSolver.h"
%include "dolfin/la/KrylovSolver.h"
%include "dolfin/la/LUSolver.h"
%include "dolfin/la/SingularSolver.h"
%include "dolfin/la/solve.h"
%include "dolfin/la/BlockVector.h"
%include "dolfin/la/BlockMatrix.h"
%include "dolfin/swig/la_post.i"

// DOLFIN headers included from nls
%include "dolfin/swig/nls_pre.i"
%include "dolfin/nls/NonlinearProblem.h"
%include "dolfin/nls/NewtonSolver.h"

// DOLFIN headers included from intersection
%include "dolfin/intersection/IntersectionOperator.h"
%include "dolfin/intersection/PrimitiveIntersector.h"
%include "dolfin/intersection/PrimitiveTraits.h"
%include "dolfin/intersection/MeshPrimitive.h"

// DOLFIN headers included from mesh
%include "dolfin/swig/mesh_pre.i"
%include "dolfin/mesh/CellType.h"
%include "dolfin/mesh/MeshTopology.h"
%include "dolfin/mesh/MeshGeometry.h"
%include "dolfin/mesh/MeshDomains.h"
%include "dolfin/mesh/MeshData.h"
%include "dolfin/mesh/Mesh.h"
%include "dolfin/mesh/MeshEntity.h"
%include "dolfin/mesh/MeshEntityIterator.h"
%include "dolfin/mesh/SubsetIterator.h"
%include "dolfin/mesh/Point.h"
%include "dolfin/mesh/Vertex.h"
%include "dolfin/mesh/Edge.h"
%include "dolfin/mesh/Face.h"
%include "dolfin/mesh/Facet.h"
%include "dolfin/mesh/Cell.h"
%include "dolfin/mesh/FacetCell.h"
%include "dolfin/mesh/MeshConnectivity.h"
%include "dolfin/mesh/MeshEditor.h"
%include "dolfin/mesh/DynamicMeshEditor.h"
%include "dolfin/mesh/MeshFunction.h"
%include "dolfin/mesh/MeshPartitioning.h"
%include "dolfin/mesh/MeshValueCollection.h"
%include "dolfin/mesh/MeshColoring.h"
%include "dolfin/mesh/MeshRenumbering.h"
%include "dolfin/mesh/LocalMeshData.h"
%include "dolfin/mesh/SubDomain.h"
%include "dolfin/mesh/SubMesh.h"
%include "dolfin/mesh/DomainBoundary.h"
%include "dolfin/mesh/BoundaryMesh.h"
%include "dolfin/mesh/UnitTetrahedron.h"
%include "dolfin/mesh/UnitCube.h"
%include "dolfin/mesh/UnitInterval.h"
%include "dolfin/mesh/Interval.h"
%include "dolfin/mesh/UnitTriangle.h"
%include "dolfin/mesh/UnitSquare.h"
%include "dolfin/mesh/UnitCircle.h"
%include "dolfin/mesh/Box.h"
%include "dolfin/mesh/Rectangle.h"
%include "dolfin/mesh/UnitSphere.h"
%include "dolfin/swig/mesh_post.i"

// DOLFIN headers included from refinement
%include "dolfin/refinement/refine.h"

// DOLFIN headers included from function
%include "dolfin/swig/function_pre.i"
%include "dolfin/function/GenericFunction.h"
%include "dolfin/function/Expression.h"
%include "dolfin/function/Function.h"
%include "dolfin/function/FunctionSpace.h"
%include "dolfin/function/SubSpace.h"
%include "dolfin/function/Constant.h"
%include "dolfin/function/SpecialFunctions.h"
%include "dolfin/swig/function_post.i"

// DOLFIN headers included from graph
%include "dolfin/graph/MatrixRenumbering.h"

// DOLFIN headers included from plot
%include "dolfin/plot/FunctionPlotData.h"

// DOLFIN headers included from math
%include "dolfin/math/basic.h"
%include "dolfin/math/Lagrange.h"
%include "dolfin/math/Legendre.h"

// DOLFIN headers included from quadrature
%include "dolfin/quadrature/Quadrature.h"
%include "dolfin/quadrature/GaussianQuadrature.h"
%include "dolfin/quadrature/GaussQuadrature.h"
%include "dolfin/quadrature/RadauQuadrature.h"
%include "dolfin/quadrature/LobattoQuadrature.h"
%include "dolfin/quadrature/BarycenterQuadrature.h"

// DOLFIN headers included from ale
%include "dolfin/ale/ALE.h"

// DOLFIN headers included from fem
%include "dolfin/swig/fem_pre.i"
%include "dolfin/fem/GenericDofMap.h"
%include "dolfin/fem/DofMap.h"
%include "dolfin/fem/Equation.h"
%include "dolfin/fem/FiniteElement.h"
%include "dolfin/fem/BasisFunction.h"
%include "dolfin/fem/BoundaryCondition.h"
%include "dolfin/fem/DirichletBC.h"
%include "dolfin/fem/PeriodicBC.h"
%include "dolfin/fem/PointSource.h"
%include "dolfin/fem/assemble.h"
%include "dolfin/fem/solve.h"
%include "dolfin/fem/Form.h"
%include "dolfin/fem/Assembler.h"
%include "dolfin/fem/SparsityPatternBuilder.h"
%include "dolfin/fem/SystemAssembler.h"
%include "dolfin/fem/LinearVariationalProblem.h"
%include "dolfin/fem/LinearVariationalSolver.h"
%include "dolfin/fem/NonlinearVariationalProblem.h"
%include "dolfin/fem/NonlinearVariationalSolver.h"
%include "dolfin/fem/OpenMpAssembler.h"
%include "dolfin/fem/VariationalProblem.h"
%include "dolfin/swig/fem_post.i"

// DOLFIN headers included from adaptivity
%include "dolfin/swig/adaptivity_pre.i"
%include "dolfin/adaptivity/GenericAdaptiveVariationalSolver.h"
%include "dolfin/adaptivity/AdaptiveLinearVariationalSolver.h"
%include "dolfin/adaptivity/AdaptiveNonlinearVariationalSolver.h"
%include "dolfin/adaptivity/GoalFunctional.h"
%include "dolfin/adaptivity/ErrorControl.h"
%include "dolfin/adaptivity/Extrapolation.h"
%include "dolfin/adaptivity/LocalAssembler.h"
%include "dolfin/adaptivity/SpecialFacetFunction.h"
%include "dolfin/adaptivity/TimeSeries.h"
%include "dolfin/adaptivity/adapt.h"
%include "dolfin/adaptivity/marking.h"
%include "dolfin/adaptivity/adaptivesolve.h"

// DOLFIN headers included from io
%include "dolfin/io/File.h"
%include "dolfin/swig/io_post.i"
