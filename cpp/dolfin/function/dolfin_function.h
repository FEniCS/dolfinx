#pragma once

namespace dolfin
{
/*! \namespace dolfin::function
    \brief Functions tools, including FEM functions and pointwise defined functions

    This namespace provides classes for representing finite element functions, and
    coefficient functions that appear in forms.
*/
}

// DOLFIN function interface

#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionAXPY.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/SpecialFunctions.h>
