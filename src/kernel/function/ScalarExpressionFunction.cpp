// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementFunction.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/NewPDE.h>
#include <dolfin/ScalarExpressionFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ScalarExpressionFunction::ScalarExpressionFunction(function f) :
  ExpressionFunction(), f(f)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ScalarExpressionFunction::~ScalarExpressionFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ScalarExpressionFunction::operator() (const Node& n, real t)  const
{
  if ( f == 0 )
    return 0.0;
		 
  Point p = n.coord();
  return f(p.x, p.y, p.z, t);
}
//-----------------------------------------------------------------------------
real ScalarExpressionFunction::operator() (const Node& n, real t)
{
  if ( f == 0 )
    return 0.0;
		 
  Point p = n.coord();
  return f(p.x, p.y, p.z, t);
}
//-----------------------------------------------------------------------------
real ScalarExpressionFunction::operator() (const Point& p, real t) const
{
  if ( f == 0 )
    return 0.0;
  
  return f(p.x, p.y, p.z, t);
}
//-----------------------------------------------------------------------------
real ScalarExpressionFunction::operator() (const Point& p, real t)
{
  if ( f == 0 )
    return 0.0;
  
  return f(p.x, p.y, p.z, t);
}
//-----------------------------------------------------------------------------
real ScalarExpressionFunction::operator() (real x, real y, real z, real t) const
{
  if ( f == 0 )
    return 0.0;
  
  return f(x, y, z, t);
}
//-----------------------------------------------------------------------------
real ScalarExpressionFunction::operator() (real x, real y, real z, real t)
{
  if ( f == 0 )
    return 0.0;
  
  return f(x, y, z, t);
}
//-----------------------------------------------------------------------------
void ScalarExpressionFunction::update(FunctionSpace::ElementFunction& v,
				      const FiniteElement& element,
				      const Cell& cell,
				      real t) const
{
  for (FiniteElement::TrialFunctionIterator phi(element); !phi.end(); ++phi)
    v.set(phi.index(), phi, phi.dof(cell, *this, t));
}
//-----------------------------------------------------------------------------
void ScalarExpressionFunction::update(NewArray<real>& w, const Cell& cell, 
				      const NewFiniteElement& element) const
{
  // FIXME: time t ignored
  
  w.resize(element.spacedim());
  for (unsigned int i = 0; i < element.spacedim(); i++)
  {
    const Point& p = element.coord(i, cell);
    w[i] = f(p.x, p.y, p.z, 0.0);
  }
}
//-----------------------------------------------------------------------------
