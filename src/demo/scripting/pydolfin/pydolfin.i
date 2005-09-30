%module pydolfin
%{
#include <dolfin.h>
#include <string>

namespace dolfin {

%}

%include "std_string.i"

%typemap(python,out) real = double; 
%typemap(python,in) uint = int; 

%import "dolfin.h"

%rename(increment) dolfin::NodeIterator::operator++;

%include "dolfin/Mesh.h"
%include "dolfin/Point.h"
%include "dolfin/File.h"
%include "dolfin/constants.h"
%include "dolfin/Node.h"
%include "dolfin/NodeIterator.h"

