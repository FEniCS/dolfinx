// Copyright (C) 2003-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2003-11-28
// Last changed: 2005-12-01
//
// The class Function serves as the envelope class and holds a pointer
// to a letter class that is a subclass of GenericFunction. All the
// functionality is handled by the specific implementation (subclass).

#include <dolfin/UserFunction.h>
#include <dolfin/FunctionPointerFunction.h>
#include <dolfin/DiscreteFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(uint vectordim)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(user), _cell(0)
{
  f = new UserFunction(this, vectordim);
}
//-----------------------------------------------------------------------------
Function::Function(FunctionPointer fp, uint vectordim)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(functionpointer), _cell(0)
{
  f = new FunctionPointerFunction(fp, vectordim);
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0)
{
  f = new DiscreteFunction(x);
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0)
{
  f = new DiscreteFunction(x, mesh);
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh, FiniteElement& element)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0)
{
  f = new DiscreteFunction(x, mesh, element);
}
//-----------------------------------------------------------------------------
Function::Function(const Function& f)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(f._type), _cell(0)
{
  switch ( f.type() )
  {
  case user:
    this->f = new UserFunction(*((UserFunction *) f.f));
    break;
  case functionpointer:
    this->f = new FunctionPointerFunction(*((FunctionPointerFunction *) f.f));
    break;
  case discrete:
    this->f = new DiscreteFunction(*((DiscreteFunction *) f.f));
    break;
  default:
    dolfin_error("Unknown function type.");
  }
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  delete f;
}
//-----------------------------------------------------------------------------
Function Function::operator[] (const uint i)
{
  // Create copy
  Function f(*this);

  // Restrict copy to sub function or component
  f.f->sub(i);

  return f;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Function& f)
{
  delete this->f;

  switch ( f.type() )
  {
  case user:
    this->f = new UserFunction(*((UserFunction *) f.f));
    break;
  case functionpointer:
    this->f = new FunctionPointerFunction(*((FunctionPointerFunction *) f.f));
    break;
  case discrete:
    this->f = new DiscreteFunction(*((DiscreteFunction *) f.f));
    break;
  default:
    dolfin_error("Unknown function type.");
  }

  return *this;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real coefficients[], AffineMap& map,
			   FiniteElement& element)
{
  // Save cell so it can be used by user-defined function
  _cell = &map.cell();
  
  // Delegate function call
  f->interpolate(coefficients, map, element);

  // Reset cell since it is no longer current
  _cell = 0;
}
//-----------------------------------------------------------------------------
Cell& Function::cell()
{
  if ( !_cell )
    dolfin_error("Current cell is unknown (only available during assembly).");
  
  return *_cell;
}
//-----------------------------------------------------------------------------
