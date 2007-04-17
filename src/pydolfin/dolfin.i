%module(directors="1") dolfin

%feature("autodoc", "1");

%{
#include <dolfin.h>

#include <numpy/arrayobject.h>
  
using namespace dolfin;
%}

%init%{
  import_array();
%}

// Ignores
%include "ignores.i"

// Renames
%include "renames.i"

class Parametrized  {};



// Typemaps
%include "typemaps.i"

// Directors
%include "directors.i"


%include "cpointer.i"
%include "std_string.i"
%include "std_vector.i"
%include "stl.i"

%include "carrays.i"

%array_functions(dolfin::real, realArray);
%array_functions(int, intArray);

%pointer_class(int, intp);
%pointer_class(double, doublep);

// la pre

%include "dolfin_la_pre.i"

// log pre

// function pre

%include exception.i
%rename(__call__) dolfin::Function::operator();
%rename(__getitem__) dolfin::Function::operator[];


// mesh pre

%include "dolfin_mesh_pre.i"

// ode pre

%include "dolfin_ode_pre.i"

// DOLFIN interface

%import "dolfin/constants.h"
%include "dolfin_headers.h"

// common post

%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
%template(ArrayFunctionPtr) dolfin::Array<dolfin::Function *>;
%template(STLVectorUInt) std::vector<unsigned int>;
%template(ArrayUInt) dolfin::Array<unsigned int>;

// la post

%include "dolfin_la_post.i"

// mesh post

%include "dolfin_mesh_post.i"


// DOLFIN FEM interface

%include "dolfin_fem_post.i"

// glue 

%include "dolfin_glue.h"
