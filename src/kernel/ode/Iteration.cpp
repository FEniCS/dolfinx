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
		     real maxdiv, real maxconv, real tol) : 
  u(u), f(f), fixpoint(fixpoint), maxdiv(maxdiv), maxconv(maxconv), tol(tol)
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
  element.reset(u0);
}
//-----------------------------------------------------------------------------
real Iteration::residual(TimeSlab& timeslab)
{
  return timeslab.computeMaxRd(fixpoint);
}
//-----------------------------------------------------------------------------
real Iteration::residual(NewArray<Element*> elements)
{
  real rmax = 0.0;
  
  // Compute maximum discrete residual
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Compute discrete residual
    rmax = std::max(rmax, residual(*element));
  }

  return rmax;
}
//-----------------------------------------------------------------------------
real Iteration::residual(Element& element)
{
  return fabs(element.computeDiscreteResidual(f));
}
//-----------------------------------------------------------------------------
real Iteration::computeConvergenceRate(const Iteration::Residuals& r)
{
  return r.r2 / (DOLFIN_EPS + r.r1);
}
//-----------------------------------------------------------------------------
real Iteration::computeDamping(real rho)
{
  if ( rho >= 0.0 || rho < -1.0 )
    return (1.0 + DOLFIN_SQRT_EPS) / (1.0 + rho);

  return 1.0;
}
//-----------------------------------------------------------------------------
unsigned int Iteration::computeDampingSteps(real rho)
{
  dolfin_assert(rho >= 0.0);
  return 1 + 2*ceil_int(log(1.0 + rho));
}
//-----------------------------------------------------------------------------
