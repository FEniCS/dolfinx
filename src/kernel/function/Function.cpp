// Copyright (C) 2003-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2003-11-28
// Last changed: 2007-04-12
//
// The class Function serves as the envelope class and holds a pointer
// to a letter class that is a subclass of GenericFunction. All the
// functionality is handled by the specific implementation (subclass).

#include <dolfin/UserFunction.h>
#include <dolfin/ConstantFunction.h>
#include <dolfin/DiscreteFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh)
  : Variable("u", "user-defined function"), f(0), _type(user)
{
  f = new UserFunction(mesh, this);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, real value)
  : Variable("u", "constant function"), f(0), _type(constant)
{
  f = new ConstantFunction(mesh, value);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, Vector& x, const Form& form, uint i)
  : Variable("u", "discrete function"), f(0), _type(discrete)
{
  f = new DiscreteFunction(mesh, x, form, i);
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  if ( f )
    delete f;
}
//-----------------------------------------------------------------------------
Function::Type Function::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
dolfin::uint Function::rank() const
{
  dolfin_assert(f);
  return f->rank();
}
//-----------------------------------------------------------------------------
dolfin::uint Function::dim(unsigned int i) const
{
  dolfin_assert(f);
  return f->dim(i);
}
//-----------------------------------------------------------------------------
Mesh& Function::mesh()
{
  dolfin_assert(f);
  return f->mesh;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real* values)
{
  dolfin_assert(f);
  f->interpolate(values);
}
//-----------------------------------------------------------------------------
void Function::interpolate(real* coefficients,
                           const ufc::cell& cell,
                           const ufc::finite_element& finite_element)
{
  dolfin_assert(f);
  f->interpolate(coefficients, cell, finite_element);
}
//-----------------------------------------------------------------------------
void Function::eval(real* values, const real* x)
{
  // Try scalar function if not overloaded
  values[0] = eval(x);
}
//-----------------------------------------------------------------------------
dolfin::real Function::eval(const real* x)
{
  dolfin_error("Missing eval() for user-defined function (must be overloaded).");
  return 0.0;
}
//-----------------------------------------------------------------------------
