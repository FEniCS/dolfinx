// Generated list of include files for PyDOLFIN

// DOLFIN headers included from dolfin_parameter.h
%include "dolfin/Parameter.h"
%include "dolfin/ParameterSystem.h"

// DOLFIN headers included from dolfin_quadrature.h
%include "dolfin/Quadrature.h"
%include "dolfin/GaussQuadrature.h"
%include "dolfin/RadauQuadrature.h"
%include "dolfin/LobattoQuadrature.h"

// DOLFIN headers included from dolfin_la.h
%include "dolfin/AssemblyMatrix.h"
%include "dolfin/Matrix.h"
%include "dolfin/Vector.h"
%include "dolfin/GenericMatrix.h"
%include "dolfin/GenericVector.h"
%include "dolfin/GenericTensor.h"
%include "dolfin/DenseMatrix.h"
%include "dolfin/DenseVector.h"
%include "dolfin/SparseMatrix.h"
%include "dolfin/SparseVector.h"
%include "dolfin/LinearSolver.h"
%include "dolfin/KrylovMethod.h"
%include "dolfin/KrylovSolver.h"
%include "dolfin/GMRES.h"
%include "dolfin/LU.h"
%include "dolfin/LUSolver.h"
%include "dolfin/PETScKrylovMatrix.h"
%include "dolfin/PETScKrylovSolver.h"
%include "dolfin/PETScLinearSolver.h"
%include "dolfin/PETScLUSolver.h"
%include "dolfin/PETScManager.h"
%include "dolfin/PETScMatrix.h"
%include "dolfin/PETScPreconditioner.h"
%include "dolfin/PETScVector.h"
%include "dolfin/Preconditioner.h"
%include "dolfin/SLEPcEigenvalueSolver.h"
%include "dolfin/SparsityPattern.h"
%include "dolfin/ublas.h"
%include "dolfin/uBlasDenseMatrix.h"
%include "dolfin/uBlasDummyPreconditioner.h"
%include "dolfin/uBlasKrylovMatrix.h"
%include "dolfin/uBlasKrylovSolver.h"
%include "dolfin/uBlasLinearSolver.h"
%include "dolfin/uBlasLUSolver.h"
%include "dolfin/uBlasMatrix.h"
%include "dolfin/uBlasILUPreconditioner.h"
%include "dolfin/uBlasPreconditioner.h"
%include "dolfin/uBlasSparseMatrix.h"
%include "dolfin/uBlasVector.h"

// DOLFIN headers included from dolfin_main.h
%include "dolfin/init.h"
%include "dolfin/constants.h"

// DOLFIN headers included from dolfin_ode.h
%include "dolfin/ComplexODE.h"
%include "dolfin/Homotopy.h"
%include "dolfin/ODE.h"
%include "dolfin/cGqMethod.h"
%include "dolfin/dGqMethod.h"

// DOLFIN headers included from dolfin_pde.h
%include "dolfin/GenericPDE.h"
%include "dolfin/PDE.h"
%include "dolfin/LinearPDE.h"
%include "dolfin/NonlinearPDE.h"
%include "dolfin/TimeDependentPDE.h"

// DOLFIN headers included from dolfin_io.h
%include "dolfin/File.h"

// DOLFIN headers included from dolfin_graph.h
%include "dolfin/Graph.h"
%include "dolfin/GraphEditor.h"
%include "dolfin/UndirectedClique.h"
%include "dolfin/DirectedClique.h"

// DOLFIN headers included from dolfin_nls.h
%include "dolfin/NewtonSolver.h"
%include "dolfin/NonlinearProblem.h"

// DOLFIN headers included from dolfin_mesh.h
%include "dolfin/BoundaryComputation.h"
%include "dolfin/BoundaryMesh.h"
%include "dolfin/Cell.h"
%include "dolfin/CellType.h"
%include "dolfin/Edge.h"
%include "dolfin/Face.h"
%include "dolfin/Facet.h"
%include "dolfin/GTSInterface.h"
%include "dolfin/Interval.h"
%include "dolfin/LocalMeshRefinement.h"
%include "dolfin/LocalMeshCoarsening.h"
%include "dolfin/IntersectionDetector.h"
%include "dolfin/MeshConnectivity.h"
%include "dolfin/MeshData.h"
%include "dolfin/MeshEditor.h"
%include "dolfin/MeshEntity.h"
%include "dolfin/MeshEntityIterator.h"
%include "dolfin/MeshFunction.h"
%include "dolfin/MeshGeometry.h"
%include "dolfin/MeshHierarchy.h"
%include "dolfin/MeshHierarchyAlgorithms.h"
%include "dolfin/MeshOrdering.h"
%include "dolfin/Mesh.h"
%include "dolfin/MeshTopology.h"
%include "dolfin/Point.h"
%include "dolfin/SubDomain.h"
%include "dolfin/Tetrahedron.h"
%include "dolfin/TopologyComputation.h"
%include "dolfin/Triangle.h"
%include "dolfin/UniformMeshRefinement.h"
%include "dolfin/UnitCube.h"
%include "dolfin/UnitSquare.h"
%include "dolfin/Vertex.h"

// DOLFIN headers included from dolfin_math.h
%include "dolfin/basic.h"
%include "dolfin/Lagrange.h"
%include "dolfin/Legendre.h"

// DOLFIN headers included from dolfin_common.h
%include "dolfin/AdjacencyGraph.h"
%include "dolfin/Array.h"
%include "dolfin/List.h"
%include "dolfin/utils.h"
%include "dolfin/timing.h"

// DOLFIN headers included from dolfin_elements.h
%include "dolfin/ElementLibrary.h"

// DOLFIN headers included from dolfin_log.h
%include "dolfin/CursesLogger.h"
%include "dolfin/Event.h"
%include "dolfin/GenericLogger.h"
%include "dolfin/Logger.h"
%include "dolfin/LoggerMacros.h"
%include "dolfin/LogManager.h"
%include "dolfin/LogStream.h"
%include "dolfin/Progress.h"
%include "dolfin/TerminalLogger.h"

// DOLFIN headers included from dolfin_fem.h
%include "dolfin/assemble.h"
%include "dolfin/NewBoundaryCondition.h"
%include "dolfin/Form.h"
%include "dolfin/Assembler.h"

// DOLFIN headers included from dolfin_function.h
%include "dolfin/Function.h"
%include "dolfin/SpecialFunctions.h"

// DOLFIN headers included from dolfin_mf.h
%include "dolfin/ConvectionMatrix.h"
%include "dolfin/LoadVector.h"
%include "dolfin/MassMatrix.h"
%include "dolfin/MatrixFactory.h"
%include "dolfin/StiffnessMatrix.h"
