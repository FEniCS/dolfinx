// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/Iteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Iteration::Iteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
		     unsigned int maxiter, real maxdiv, real maxconv, real tol) : 
  u(u), f(f), fixpoint(fixpoint), 
  maxiter(maxiter), maxdiv(maxdiv), maxconv(maxconv), tol(tol)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::~Iteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Iteration::init(NewArray<Element*>& elements)
{
  // Update initial data for elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Update initial data
    init(*element);
  }
}
//-----------------------------------------------------------------------------
void Iteration::init(Element& element)
{
  // Get initial value
  real u0 = u(element.index(), element.starttime());
  
  // Reset element
  element.update(u0);
}
//-----------------------------------------------------------------------------
void Iteration::reset(NewArray<Element*>& elements)
{
  // Reset elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Reset element
    reset(*element);
  }
}
//-----------------------------------------------------------------------------
void Iteration::reset(Element& element)
{
  // Get initial value
  real u0 = u(element.index(), element.starttime());

  // Reset element
  element.set(u0);
}
//-----------------------------------------------------------------------------
real Iteration::residual(TimeSlab& timeslab)
{
  return timeslab.elementResidualL2(fixpoint);
}
//-----------------------------------------------------------------------------
real Iteration::residual(NewArray<Element*>& elements)
{
  real r = 0.0;
  
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Compute discrete residual
    r += sqr(residual(*element));
  }

  return sqrt(r);
}
//-----------------------------------------------------------------------------
real Iteration::residual(Element& element)
{
  return fabs(element.computeElementResidual(f));
}
//-----------------------------------------------------------------------------
