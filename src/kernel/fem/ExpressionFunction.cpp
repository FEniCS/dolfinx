// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/ExpressionFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ExpressionFunction::ExpressionFunction(function f)
{
  this->f = f;
}
//-----------------------------------------------------------------------------
void ExpressionFunction::update(FunctionSpace::ElementFunction &v,
										  const FiniteElement &element,
										  const Cell &cell,
										  real t) const
{
  for (FiniteElement::TrialFunctionIterator phi(element); !phi.end(); ++phi)
	 v.set(phi.index(), phi, phi.dof(cell, f, t));
}
//-----------------------------------------------------------------------------
real ExpressionFunction::operator() (const Node& n, real t)  const
{
  Point p = n.coord();
  return f(p.x, p.y, p.z, t);
}
//-----------------------------------------------------------------------------
real ExpressionFunction::operator() (const Point& p, real t) const
{
  return f(p.x, p.y, p.z, t);
}
//-----------------------------------------------------------------------------
