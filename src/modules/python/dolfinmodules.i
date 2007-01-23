%module(directors="1") dolfinmodules

%feature("autodoc", "1");

%{
#include <dolfin.h>

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
%import "dolfin.h"

// DOLFIN modules

%include "../elasticity-updated/dolfin/ElasticityUpdatedSolver.h"
