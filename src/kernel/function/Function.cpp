// Copyright (C) 2003-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2003-11-28
// Last changed: 2006-12-12
//
// The class Function serves as the envelope class and holds a pointer
// to a letter class that is a subclass of GenericFunction. All the
// functionality is handled by the specific implementation (subclass).

#include <dolfin/ConstantFunction.h>
#include <dolfin/UserFunction.h>
#include <dolfin/FunctionPointerFunction.h>
#include <dolfin/DiscreteFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(real value)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(constant), _cell(0), _facet(-1)
{
  f = new ConstantFunction(value);
}
//-----------------------------------------------------------------------------
Function::Function(uint vectordim)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(user), _cell(0), _facet(-1)
{
  f = new UserFunction(this, vectordim);
}
//-----------------------------------------------------------------------------
Function::Function(FunctionPointer fp, uint vectordim)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(functionpointer), _cell(0), _facet(-1)
{
  f = new FunctionPointerFunction(fp, vectordim);
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  f = new DiscreteFunction(x);
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  f = new DiscreteFunction(x, mesh);
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh, FiniteElement& element)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  f = new DiscreteFunction(x, mesh, element);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, FiniteElement& element)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  f = new DiscreteFunction(mesh, element);
}
//-----------------------------------------------------------------------------
Function::Function(const Function& f)
  : Variable("u", "no description"), TimeDependent(),
    f(0), _type(f._type), _cell(0), _facet(-1)
{
  switch ( f._type )
  {
  case constant:
    this->f = new ConstantFunction(*((ConstantFunction *) f.f));
    break;
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
real Function::eval(const Point& p, uint i)
{
  dolfin_info("User-defined functions must overload real Function::eval(const Point& p, uint i).");
  dolfin_error("Missing eval() for user-defined function.");

  return 0.0;
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
  switch ( f._type )
  {
  case constant:
    delete this->f;
    this->f = new ConstantFunction(*((ConstantFunction *) f.f));
    break;
  case user:
    delete this->f;
    this->f = new UserFunction(*((UserFunction *) f.f));
    break;
  case functionpointer:
    delete this->f;
    this->f = new FunctionPointerFunction(*((FunctionPointerFunction *) f.f));
    break;
  case discrete:
    // Don't delete data if not necessary (don't want to recreate vector)
    if ( _type == discrete )
    {
      ((DiscreteFunction *) this->f)->copy(*((DiscreteFunction *) f.f));
    }
    else
    {
      delete this->f;
      this->f = new DiscreteFunction(*((DiscreteFunction *) f.f));
    }
    break;
  default:
    dolfin_error("Unknown function type.");
  }

  _type = f._type;

  return *this;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real coefficients[], Cell& cell,
                           AffineMap& map, FiniteElement& element)
{
  // Save cell so it can be used by user-defined function
  _cell = &cell;
  
  // Delegate function call
  f->interpolate(coefficients, cell, map, element);

  // Reset cell since it is no longer current
  _cell = 0;

  // Reset facet since it is no longer current
  _facet = -1;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real coefficients[], Cell& cell,
                           AffineMap& map, FiniteElement& element, uint facet)
{
  // Save cell so it can be used by user-defined function
  _cell = &cell;

  // Save facet so it can be used by user-defined function
  _facet = facet;
  
  dolfin_info("Interpolating function on cell = %d and facet = %d", cell.index(), facet);

  // Delegate function call
  f->interpolate(coefficients, cell, map, element);

  // Reset cell since it is no longer current
  _cell = 0;

  // Reset facet since it is no longer current
  _facet = -1;
}
//-----------------------------------------------------------------------------
void Function::interpolate(Function& fsource)
{
  // Delegate function call
  f->interpolate(fsource);
}
//-----------------------------------------------------------------------------
void Function::init(Mesh& mesh, FiniteElement& element)
{
  if ( _type != discrete )
  {
    delete f;
    f = new DiscreteFunction(mesh, element);
  }
  else
  {
    ((DiscreteFunction*) f)->init(mesh, element);
  }

  _type = discrete;
  _cell = 0;
  _facet = -1;
}
//-----------------------------------------------------------------------------
Cell& Function::cell()
{
  if ( !_cell )
    dolfin_error("Current cell is unknown (only available during assembly).");
  
  return *_cell;
}
//-----------------------------------------------------------------------------
dolfin::uint Function::facet()
{
  if ( !(_facet >= 0) )
  {
    dolfin_warning("Current facet is unknown (only available during assembly over facets), returning 0");
    return 0;
  }
//    dolfin_error("Current facet is unknown (only available during assembly over facets).");
  return static_cast<uint>(_facet);
}
//-----------------------------------------------------------------------------
