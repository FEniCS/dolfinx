// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Settings.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Grid.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/GenericFunction.h>
#include <dolfin/DofFunction.h>
#include <dolfin/ExpressionFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(Grid &grid, Vector &x) : _grid(grid)
{
  f = new DofFunction(x);

  rename("u", "A function");
}
//-----------------------------------------------------------------------------
Function::Function(Grid &grid, const char *name) : _grid(grid)
{
  function fp;
  Settings::get(name, &fp);
  
  f = new ExpressionFunction(fp);

  rename("u", "A function");
}
//-----------------------------------------------------------------------------
void Function::update(FunctionSpace::ElementFunction& v,
							 const FiniteElement& element,
							 const Cell& cell, real t) const
{
  // Update degrees of freedom for element function, assuming it belongs to
  // the local trial space of the finite element.

  // Set dimension of function space for element function
  v.init(element.dim());
   
  // Update coefficients
  f->update(v, element, cell, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (const Node& n, real t) const
{
  return (*f)(n, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (const Point& p, real t) const
{
  return (*f)(p, t);
}
//-----------------------------------------------------------------------------
const Grid& Function::grid() const
{
  return _grid;
}
//-----------------------------------------------------------------------------
