// Generated file to include docstrings
%include "dolfin_docstrings.i"

// Generated list of include files for PyDOLFIN

// DOLFIN headers included from common
%include "dolfin/common/types.h"
%include "dolfin/common/constants.h"
%include "dolfin/common/timing.h"
%include "dolfin/common/Array.h"
%include "dolfin/common/List.h"
%include "dolfin/common/simple_array.h"
%include "dolfin/common/Timer.h"
%include "dolfin/common/TimeDependent.h"
%include "dolfin/common/Variable.h"

// DOLFIN headers included from parameter
%include "dolfin/parameter/Parametrized.h"
%include "dolfin/parameter/Parameter.h"
%include "dolfin/parameter/parameters.h"

// DOLFIN headers included from log
%include "dolfin/log/log.h"
%include "dolfin/log/Event.h"
%include "dolfin/log/LogStream.h"
%include "dolfin/log/Progress.h"

// DOLFIN headers included from la
%include "dolfin/la/VectorNormType.h"
%include "dolfin/la/NormalizationType.h"
%include "dolfin/la/SolverType.h"
%include "dolfin/la/PreconditionerType.h"
%include "dolfin/la/GenericTensor.h"
%include "dolfin/la/GenericMatrix.h"
%include "dolfin/la/GenericVector.h"
%include "dolfin/la/PETScObject.h"
%include "dolfin/la/uBlasMatrix.h"
%include "dolfin/la/PETScMatrix.h"
%include "dolfin/la/EpetraMatrix.h"
%include "dolfin/la/AssemblyMatrix.h"
%include "dolfin/la/uBlasVector.h"
%include "dolfin/la/PETScVector.h"
%include "dolfin/la/EpetraVector.h"
%include "dolfin/la/GenericSparsityPattern.h"
%include "dolfin/la/SparsityPattern.h"
%include "dolfin/la/LinearAlgebraFactory.h"
%include "dolfin/la/DefaultFactory.h"
%include "dolfin/la/uBlasFactory.h"
%include "dolfin/la/PETScFactory.h"
%include "dolfin/la/EpetraFactory.h"
%include "dolfin/la/AssemblyFactory.h"
%include "dolfin/la/PETScKrylovSolver.h"
%include "dolfin/la/PETScLUSolver.h"
%include "dolfin/la/SLEPcEigenvalueSolver.h"
%include "dolfin/la/uBlasDenseMatrix.h"
%include "dolfin/la/uBlasKrylovSolver.h"
%include "dolfin/la/uBlasLUSolver.h"
%include "dolfin/la/uBlasPreconditioner.h"
%include "dolfin/la/uBlasILUPreconditioner.h"
%include "dolfin/la/Vector.h"
%include "dolfin/la/Matrix.h"
%include "dolfin/la/Scalar.h"
%include "dolfin/la/LinearSolver.h"
%include "dolfin/la/KrylovSolver.h"
%include "dolfin/la/LUSolver.h"
%include "dolfin/la/SingularSolver.h"
%include "dolfin/la/solve.h"

// DOLFIN headers included from elements
%include "dolfin/elements/ElementLibrary.h"
%include "dolfin/elements/ProjectionLibrary.h"

// DOLFIN headers included from function
%include "dolfin/function/Function.h"
%include "dolfin/function/SpecialFunctions.h"
%include "dolfin/function/ProjectL2.h"

// DOLFIN headers included from graph
%include "dolfin/graph/Graph.h"
%include "dolfin/graph/GraphEditor.h"
%include "dolfin/graph/GraphPartition.h"
%include "dolfin/graph/UndirectedClique.h"
%include "dolfin/graph/DirectedClique.h"

// DOLFIN headers included from io
%include "dolfin/io/File.h"

// DOLFIN headers included from main
%include "dolfin/main/init.h"
%include "dolfin/common/types.h"

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

// DOLFIN headers included from mesh
%include "dolfin/mesh/ALE.h"
%include "dolfin/mesh/ALEType.h"
%include "dolfin/mesh/MeshEntity.h"
%include "dolfin/mesh/MeshEntityIterator.h"
%include "dolfin/mesh/MeshTopology.h"
%include "dolfin/mesh/MeshGeometry.h"
%include "dolfin/mesh/MeshData.h"
%include "dolfin/mesh/MeshConnectivity.h"
%include "dolfin/mesh/MeshEditor.h"
%include "dolfin/mesh/MeshFunction.h"
%include "dolfin/mesh/Mesh.h"
%include "dolfin/mesh/MPIMeshCommunicator.h"
%include "dolfin/mesh/Vertex.h"
%include "dolfin/mesh/Edge.h"
%include "dolfin/mesh/Face.h"
%include "dolfin/mesh/Facet.h"
%include "dolfin/mesh/Cell.h"
%include "dolfin/mesh/Point.h"
%include "dolfin/mesh/SubDomain.h"
%include "dolfin/mesh/DomainBoundary.h"
%include "dolfin/mesh/BoundaryMesh.h"
%include "dolfin/mesh/UnitCube.h"
%include "dolfin/mesh/UnitInterval.h"
%include "dolfin/mesh/UnitSquare.h"
%include "dolfin/mesh/UnitCircle.h"
%include "dolfin/mesh/Box.h"
%include "dolfin/mesh/Rectangle.h"
%include "dolfin/mesh/UnitSphere.h"
%include "dolfin/mesh/IntersectionDetector.h"

// DOLFIN headers included from fem
%include "dolfin/fem/assemble.h"
%include "dolfin/fem/DofMap.h"
%include "dolfin/fem/DofMapSet.h"
%include "dolfin/fem/SubSystem.h"
%include "dolfin/fem/BoundaryCondition.h"
%include "dolfin/fem/DirichletBC.h"
%include "dolfin/fem/PeriodicBC.h"
%include "dolfin/fem/Form.h"
%include "dolfin/fem/Assembler.h"
%include "dolfin/fem/pAssembler.h"

// DOLFIN headers included from mf
%include "dolfin/mf/MatrixFactory.h"

// DOLFIN headers included from nls
%include "dolfin/nls/NewtonSolver.h"
%include "dolfin/nls/NonlinearProblem.h"

// DOLFIN headers included from ode
%include "dolfin/ode/ODE.h"
%include "dolfin/ode/ComplexODE.h"
%include "dolfin/ode/Homotopy.h"
%include "dolfin/ode/Method.h"
%include "dolfin/ode/cGqMethod.h"
%include "dolfin/ode/dGqMethod.h"
%include "dolfin/ode/ODESolution.h"

// DOLFIN headers included from pde
%include "dolfin/pde/LinearPDE.h"
%include "dolfin/pde/NonlinearPDE.h"

// DOLFIN headers included from plot
