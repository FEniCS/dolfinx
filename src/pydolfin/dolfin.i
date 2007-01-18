%module(directors="1") dolfin

%feature("autodoc", "1");

%{
#include <dolfin.h>

#include "dolfin_glue.h"

#include <numpy/arrayobject.h>
#include <string>
  
using namespace dolfin;
%}

%init%{
  import_array();
%}

%typemap(in) real = double; 
%typemap(out) real = double; 
%typemap(in) uint = int; 
%typemap(out) uint = int; 

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


%include "cpointer.i"
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"
%include "stl.i"

%include "carrays.i"

%array_functions(dolfin::real, realArray);
%array_functions(int, intArray);

%pointer_class(int, intp);
%pointer_class(double, doublep);


//%feature("director") GenericVector;
%feature("director") Function;
%feature("director") BoundaryCondition;
%feature("director") ODE;
%feature("director") PDE;
%feature("director") TimeDependentPDE;

%ignore dolfin::dolfin_info;
%ignore dolfin::dolfin_info_aptr;


%import "dolfin.h"
%import "dolfin/constants.h"


// DOLFIN public interface 

// FIXME: Order matters, why? 

// main includes 

%include "dolfin/constants.h"
%include "dolfin/init.h"

// math includes 

%include "dolfin/basic.h"

// common includes 

%ignore dolfin::TimeDependent::TimeDependent(const real&);
%ignore dolfin::TimeDependent::sync(const real&);

%include "dolfin/Array.h"
%include "dolfin/List.h"
%include "dolfin/TimeDependent.h"
%include "dolfin/Variable.h"
%include "dolfin/utils.h"
%include "dolfin/timing.h"

%template(STLVectorUInt) std::vector<unsigned int>;
%template(ArrayUInt) dolfin::Array<unsigned int>;

%extend dolfin::TimeDependent {
  TimeDependent(double *t)
  {
    TimeDependent* td = new TimeDependent();
    td->sync(*t);
    return td;
  }
  void sync(double* t)
  {
    self->sync(*t);
  }
}

// log includes 

%include "dolfin/LoggerMacros.h"

// settings includes 

%rename(get) glueget;

%include "dolfin/Parameter.h"

%pythoncode
%{
def set(name, val):
  if(isinstance(val, bool)):
    glueset_bool(name, val)
  else:
    glueset(name, val)
%}

// io includes 

%include "dolfin/File.h"

// la includes 

%include "dolfin_la.i"

// function includes 
%include exception.i
%rename(__call__) dolfin::Function::operator();
%rename(__getitem__) dolfin::Function::operator[];

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::real values [] {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typemap(in) dolfin::real values [] {
    if PyArray_Check($input) {
        PyArrayObject *xa = (PyArrayObject*)($input);
        if (xa->descr->type == 'd')
            $1 = (double *)(*xa).data;
        else
            SWIG_exception(SWIG_ValueError, "numpy array of doubles expected");
    } else 
        SWIG_exception(SWIG_ValueError, "numpy array expected");
}

%include "dolfin/Function.h"

// form includes

%include "dolfin/Form.h"
%include "dolfin/BilinearForm.h"
%include "dolfin/LinearForm.h"

// DOLFIN mesh interface

%include "dolfin_mesh.i"

// ode includes

%include "dolfin_ode.i"

// pde 

%include "dolfin/TimeDependentPDE.h"

// DOLFIN FEM interface

%include "dolfin_fem.i"

// glue 

%include "dolfin_glue.h"

// modules 

%include "dolfin/ElasticityUpdatedSolver.h" 

