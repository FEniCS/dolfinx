%module pydolfin
%{
#include <dolfin.h>
#include <string>

namespace dolfin {

%}

%import "dolfin.h"

%include "dolfin/Mesh.h"
%include "dolfin/Point.h"
%include "dolfin/File.h"
%include "dolfin/constants.h"
