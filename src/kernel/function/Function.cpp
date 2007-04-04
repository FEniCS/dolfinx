// Copyright (C) 2003-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2003-11-28
// Last changed: 2007-04-04
//
// The class Function serves as the envelope class and holds a pointer
// to a letter class that is a subclass of GenericFunction. All the
// functionality is handled by the specific implementation (subclass).

#include <dolfin/UserFunction.h>
#include <dolfin/FunctionPointerFunction.h>
#include <dolfin/ConstantFunction.h>
#include <dolfin/DiscreteFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function()
  : Variable("u", "user-defined function"), f(0)
{
  f = new UserFunction(this);
}
//-----------------------------------------------------------------------------
Function::Function(FunctionPointer fp)
  : Variable("u", "user-defined function (by function pointer)"), f(0)
{
  f = new FunctionPointerFunction(fp);
}
//-----------------------------------------------------------------------------
Function::Function(real value)
  : Variable("u", "constant function"), f(0)
{
  f = new ConstantFunction(value);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, const Form& form, uint i)
  : Variable("u", "discrete function"), f(0)
{
  f = new DiscreteFunction(mesh, form, i);
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  if ( f )
    delete f;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real* coefficients,
                           const ufc::cell& cell,
                           const ufc::finite_element& finite_element)
{
  cout << "Interpolating Function" << endl;
  f->interpolate(coefficients, cell, finite_element);
}
//-----------------------------------------------------------------------------
void Function::eval(real* values, const real* coordinates)
{
  dolfin_error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
