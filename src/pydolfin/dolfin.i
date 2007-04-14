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


%feature("director") Function;
%feature("director") BoundaryCondition;
%feature("director") ODE;
%feature("director") PDE;
%feature("director") TimeDependentPDE;

%ignore dolfin::dolfin_info;
%ignore dolfin::dolfin_info_aptr;


// common pre

%ignore dolfin::TimeDependent::TimeDependent(const real&);
%ignore dolfin::TimeDependent::sync(const real&);

// settings pre

%rename(oldget) dolfin::get;

// la pre

%include "dolfin_la_pre.i"

// log pre

%ignore dolfin::Logger;

// function pre

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

// mesh pre

%include "dolfin_mesh_pre.i"

// ode pre

%include "dolfin_ode_pre.i"

// DOLFIN interface

%import "dolfin/constants.h"
%include "dolfin_headers.h"
//%include "dolfin.h"

// common post

%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
%template(ArrayFunctionPtr) dolfin::Array<dolfin::Function *>;
%template(STLVectorUInt) std::vector<unsigned int>;
%template(ArrayUInt) dolfin::Array<unsigned int>;


// settings post

%pythoncode
%{
def set(name, val):
  if(isinstance(val, bool)):
    glueset_bool(name, val)
  else:
    glueset(name, val)
%}


/*
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
*/


// la post

%include "dolfin_la_post.i"

// mesh post

%include "dolfin_mesh_post.i"


// DOLFIN FEM interface

%include "dolfin_fem_post.i"

// glue 

%include "dolfin_glue.h"
