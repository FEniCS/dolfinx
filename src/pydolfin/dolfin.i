%module(directors="1") dolfin

%{
#include <dolfin.h>

#include "dolfin_glue.h"

#include <string>
  
using namespace dolfin;
%}

%typemap(python,in) real = double; 
%typemap(python,out) real = double; 
%typemap(python,in) uint = int; 
%typemap(python,out) uint = int; 

%typemap(out) dolfin::Parameter {
  {
    // Custom typemap

    switch ( $1.type() )
    {
    case Parameter::type_real:
      
      $result = SWIG_From_double(*&($1));
      break;

    case Parameter::type_int:
      
      $result = SWIG_From_int((int)*&($1));
      break;
      
    case Parameter::type_bool:
      
      $result = SWIG_From_bool(*&($1));
      break;
      
    case Parameter::type_string:
      
      $result = SWIG_From_std_string(*&($1));
      break;
      
    default:
      dolfin_error("Unknown type for parameter.");
    }
  }
}


// Typemaps for dolfin::real array arguments in virtual methods
// probably not very safe
%typemap(directorin) dolfin::real [] {
  {
    // Custom typemap
    $input = SWIG_NewPointerObj((void *) $1_name, $1_descriptor, $owner);
  }
}

%typemap(directorin) dolfin::real const [] {
  {
    // Custom typemap
    $input = SWIG_NewPointerObj((void *) $1_name, $1_descriptor, $owner);
  }
}


%include "pointer.i"
%include "typemaps.i"
%include "std_string.i"

%include "carrays.i"

%array_functions(dolfin::real, realArray);
%array_functions(int, intArray);




%feature("director") Function;
%feature("director") BoundaryCondition;
%feature("director") ODE;
%feature("director") PDE;
%feature("director") TimeDependentPDE;

%ignore dolfin::dolfin_info;
%ignore dolfin::dolfin_info_aptr;


%import "dolfin.h"
%import "dolfin/constants.h"

#ifdef HAVE_PETSC_H
  %rename(Vector) PETScVector;
  %rename(Matrix) PETScSparseMatrix;
  %rename(KrylovSolver) PETScKrylovSolver;
#else
  %rename(Vector) DenseVector;
  %rename(Matrix) uBlasSparseMatrix;
  %rename(KrylovSolver) uBlasKrylovSolver;
#endif


%rename(set) glueset;
%rename(get) glueget;
%rename(increment) dolfin::VertexIterator::operator++;
%rename(increment) dolfin::CellIterator::operator++;
%rename(increment) dolfin::EdgeIterator::operator++;
%rename(fmono) dolfin::ODE::f(const real u[], real t, real y[]);
%rename(fmulti) dolfin::ODE::f(const real u[], real t, uint i);

%ignore dolfin::uBlasLUSolver::invert;

/* FIXME: Temporary, only works for PETSc data structures */

/*%rename(copy) dolfin::SparseVector::operator=;*/
/*%rename(__getitem__) dolfin::Vector::getval;*/
/*%rename(__setitem__) dolfin::Vector::setval;*/

%rename(__call__) dolfin::Function::operator();
%rename(__getitem__) dolfin::Function::operator[];

/* DOLFIN public interface */

/* FIXME: Order matters, why? */

/* main includes */

%include "dolfin/constants.h"
%include "dolfin/init.h"

/* math includes */

%include "dolfin/basic.h"

/* common includes */

%include "dolfin/Array.h"
%include "dolfin/List.h"
%include "dolfin/TimeDependent.h"
%include "dolfin/Variable.h"
%include "dolfin/utils.h"
%include "dolfin/timing.h"

/* log includes */

%include "dolfin/LoggerMacros.h"


/* settings includes */

%include "dolfin/Parameter.h"
//%include "dolfin/ParameterSystem.h"
//%include "dolfin/Parameter.h"
//%include "dolfin/Settings.h"
//%include "dolfin/ParameterList.h"
//%include "dolfin/SettingsMacros.h"
//%include "dolfin/SettingsManager.h"
//%include "dolfin/Settings.h"

/* io includes */

%include "dolfin/File.h"

/* la includes */

// Declare dummy uBlas classes

%inline %{
  namespace boost{ namespace numeric{ namespace ublas{}}}
%}

namespace dolfin {
  class ublas_dense_matrix {};
  class ublas_sparse_matrix {};
}

%include "dolfin/Matrix.h"
%include "dolfin/Vector.h"
%include "dolfin/GenericMatrix.h"
%include "dolfin/GenericVector.h"
%include "dolfin/DenseMatrix.h"
%include "dolfin/DenseVector.h"
%include "dolfin/SparseMatrix.h"
%include "dolfin/SparseVector.h"
%include "dolfin/LinearSolver.h"
%include "dolfin/KrylovSolver.h"
%include "dolfin/GMRES.h"
%include "dolfin/LU.h"
%include "dolfin/PETScEigenvalueSolver.h"
%include "dolfin/PETScKrylovMatrix.h"
%include "dolfin/PETScKrylovSolver.h"
%include "dolfin/PETScLinearSolver.h"
%include "dolfin/PETScLUSolver.h"
%include "dolfin/PETScManager.h"
%include "dolfin/PETScMatrix.h"
%include "dolfin/PETScPreconditioner.h"
%include "dolfin/PETScVector.h"
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

%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;

/* function includes */

%include "dolfin/Function.h"

/* form includes */

%include "dolfin/Form.h"
%include "dolfin/BilinearForm.h"
%include "dolfin/LinearForm.h"

/* mesh includes */

%include "dolfin/Mesh.h"
%include "dolfin/Boundary.h"
%include "dolfin/Point.h"
%include "dolfin/Vertex.h"
%include "dolfin/Edge.h"
%include "dolfin/Triangle.h"
%include "dolfin/Tetrahedron.h"
%include "dolfin/Cell.h"
%include "dolfin/Edge.h"
%include "dolfin/Face.h"
%include "dolfin/VertexIterator.h"
%include "dolfin/CellIterator.h"
%include "dolfin/EdgeIterator.h"
%include "dolfin/FaceIterator.h"
%include "dolfin/MeshIterator.h"
%include "dolfin/UnitSquare.h"
%include "dolfin/UnitCube.h"

%include "dolfin/MeshConnectivity.h"
%include "dolfin/MeshEditor.h"
%include "dolfin/MeshEntity.h"
%include "dolfin/MeshEntityIterator.h"
%include "dolfin/MeshGeometry.h"
%include "dolfin/MeshTopology.h"
%include "dolfin/NewMesh.h"
%include "dolfin/NewMeshData.h"
%include "dolfin/MeshFunction.h"
%include "dolfin/NewVertex.h"
%include "dolfin/NewEdge.h"
%include "dolfin/NewFace.h"
%include "dolfin/NewFacet.h"
%include "dolfin/NewCell.h"
%include "dolfin/TopologyComputation.h"
%include "dolfin/CellType.h"
%include "dolfin/Interval.h"
%include "dolfin/NewTriangle.h"
%include "dolfin/NewTetrahedron.h"
%include "dolfin/UniformMeshRefinement.h"
%include "dolfin/NewPoint.h"
%include "dolfin/BoundaryComputation.h"
%include "dolfin/BoundaryMesh.h"
%include "dolfin/NewUnitCube.h"
%include "dolfin/NewUnitSquare.h"

/* ode includes */

%include "dolfin/Dependencies.h"
/*%include "dolfin/Dual.h"*/
%include "dolfin/Homotopy.h"
%include "dolfin/HomotopyJacobian.h"
%include "dolfin/HomotopyODE.h"
%include "dolfin/Method.h"
%include "dolfin/MonoAdaptiveFixedPointSolver.h"
%include "dolfin/MonoAdaptiveJacobian.h"
%include "dolfin/MonoAdaptiveNewtonSolver.h"
%include "dolfin/MonoAdaptiveTimeSlab.h"
%include "dolfin/MonoAdaptivity.h"
%include "dolfin/MultiAdaptiveFixedPointSolver.h"
%include "dolfin/MultiAdaptivePreconditioner.h"
%include "dolfin/MultiAdaptiveNewtonSolver.h"
%include "dolfin/MultiAdaptiveTimeSlab.h"
%include "dolfin/MultiAdaptivity.h"
%include "dolfin/ODE.h"
%include "dolfin/ODESolver.h"
%include "dolfin/ParticleSystem.h"
%include "dolfin/Partition.h"
%include "dolfin/ReducedModel.h"
%include "dolfin/Sample.h"
%include "dolfin/TimeSlab.h"
%include "dolfin/TimeSlabJacobian.h"
//%include "dolfin/TimeSlabSolver.h"
%include "dolfin/TimeStepper.h"
%include "dolfin/cGqMethod.h"
%include "dolfin/dGqMethod.h"

/* pde */

%include "dolfin/TimeDependentPDE.h"

/* fem includes */

%include "dolfin/FEM.h"
%include "dolfin/FiniteElement.h"
%include "dolfin/AffineMap.h"
%include "dolfin/BoundaryValue.h"
%include "dolfin/BoundaryCondition.h"

/* glue */

%include "dolfin_glue.h"

/* modules */

/* %include "dolfin/ElasticityUpdatedSolver.h" */
