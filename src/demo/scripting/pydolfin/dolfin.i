%module(directors="1") dolfin

%{
#include <dolfin.h>
#include "PoissonTest.h"
#include "SettingsGlue.h"

#include <string>
  
using namespace dolfin;
%}

%typemap(python,in) real = double; 
%typemap(python,out) real = double; 
%typemap(python,in) uint = int; 
%typemap(python,out) uint = int; 

%typemap(out) Parameter {
  {
    // Custom typemap

    switch ( $1.type )
    {
    case Parameter::REAL:
      
      $result = SWIG_From_double((real)($1));
      break;

    case Parameter::INT:
      
      $result = SWIG_From_int((int)($1));
      break;
      
    case Parameter::BOOL:
      
      $result = SWIG_From_bool((bool)($1));
      break;
      
    case Parameter::STRING:
      
      $result = SWIG_From_std_string((std::string)($1));
      break;
      
    default:
      dolfin_error1("Unknown type for parameter \"%s\".", $1.identifier.c_str());
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




%include "typemaps.i"
%include "std_string.i"

%include "carrays.i"

%array_functions(dolfin::real, realArray);




%feature("director") Function;
%feature("director") BoundaryCondition;
%feature("director") ODE;

%ignore dolfin::dolfin_set;
%ignore dolfin::dolfin_set_aptr;

%import "dolfin.h"
%import "dolfin/constants.h"

%rename(dolfin_set) glueset;
%rename(increment) dolfin::NodeIterator::operator++;
%rename(increment) dolfin::CellIterator::operator++;
%rename(increment) dolfin::EdgeIterator::operator++;
%rename(fmono) dolfin::ODE::f(const real u[], real t, real y[]);
%rename(fmulti) dolfin::ODE::f(const real u[], real t, uint i);

%rename(PoissonBilinearForm) dolfin::Poisson::BilinearForm;
%rename(PoissonLinearForm) dolfin::Poisson::LinearForm;
%rename(PoissonBilinearFormTestElement) dolfin::Poisson::BilinearFormTestElement;
%rename(PoissonBilinearFormTrialElement) dolfin::Poisson::BilinearFormTrialElement;



/* DOLFIN public interface */

/* main includes */

%include "dolfin/constants.h"
%include "dolfin/init.h"

/* common includes */

/* System.h seems to be obsolete? */

%include "dolfin/Array.h"
%include "dolfin/List.h"
%include "dolfin/TimeDependent.h"
%include "dolfin/Variable.h"
%include "dolfin/meminfo.h"
%include "dolfin/sysinfo.h"
%include "dolfin/timeinfo.h"
%include "dolfin/utils.h"

/* settings includes */

//%include "dolfin/Parameter.h"
//%include "dolfin/Settings.h"
//%include "dolfin/ParameterList.h"
%include "dolfin/SettingsMacros.h"
//%include "dolfin/SettingsManager.h"
//%include "dolfin/Settings.h"

/* io includes */

%include "dolfin/File.h"

/* la includes */

%include "dolfin/Vector.h"
%include "dolfin/Matrix.h"
%include "dolfin/VirtualMatrix.h"
%include "dolfin/GMRES.h"
%include "dolfin/LinearSolver.h"
%include "dolfin/EigenvalueSolver.h"
%include "dolfin/Preconditioner.h"
%include "dolfin/PETScManager.h"

/* function includes */

%include "dolfin/Function.h"

/* fem includes */

%include "dolfin/FEM.h"
%include "dolfin/FiniteElement.h"
%include "dolfin/AffineMap.h"
%include "dolfin/BoundaryValue.h"
%include "dolfin/BoundaryCondition.h"

/* form includes */

%include "dolfin/Form.h"
%include "dolfin/BilinearForm.h"
%include "dolfin/LinearForm.h"

/* mesh includes */

%include "dolfin/Mesh.h"
%include "dolfin/Boundary.h"
%include "dolfin/Point.h"
%include "dolfin/Node.h"
%include "dolfin/Edge.h"
%include "dolfin/Triangle.h"
%include "dolfin/Tetrahedron.h"
%include "dolfin/Cell.h"
%include "dolfin/Edge.h"
%include "dolfin/Face.h"
%include "dolfin/NodeIterator.h"
%include "dolfin/CellIterator.h"
%include "dolfin/EdgeIterator.h"
%include "dolfin/FaceIterator.h"
%include "dolfin/MeshIterator.h"
%include "dolfin/UnitSquare.h"
%include "dolfin/UnitCube.h"

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
%include "dolfin/Regulator.h"
%include "dolfin/Sample.h"
%include "dolfin/TimeSlab.h"
%include "dolfin/TimeSlabJacobian.h"
%include "dolfin/TimeSlabSolver.h"
%include "dolfin/TimeStepper.h"
%include "dolfin/cGqMethod.h"
%include "dolfin/dGqMethod.h"

/* modules */

/* poisson */

%include "dolfin/PoissonSolver.h"
%include "PoissonTest.h"
%include "SettingsGlue.h"
