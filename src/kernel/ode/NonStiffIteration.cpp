// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/NonStiffIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NonStiffIteration::NonStiffIteration(Solution& u, RHS& f,
				     FixedPointIteration & fixpoint, 
				     real maxdiv, real maxconv, real tol) :
  Iteration(u, f, fixpoint, maxdiv, maxconv, tol)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonStiffIteration::~NonStiffIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonStiffIteration::update(TimeSlab& timeslab)
{
  

}
//-----------------------------------------------------------------------------
void NonStiffIteration::update(NewArray<Element*>& elements)
{
  // Simple update of element list
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Iterate element
    fixpoint.iterate(*element);
  }
}
//-----------------------------------------------------------------------------
void NonStiffIteration::update(Element& element)
{
  // Simple update of element
  element.update(f);
}
//-----------------------------------------------------------------------------
void NonStiffIteration::stabilize(TimeSlab& timeslab, const Residuals& r)
{


}
//-----------------------------------------------------------------------------
void NonStiffIteration::stabilize(NewArray<Element*>& elements,
				  const Residuals& r)
{


}
//-----------------------------------------------------------------------------
void NonStiffIteration::stabilize(Element& element, const Residuals& r)
{
  // Compute local damping
  
  /*
    real dfdu = f.dfdu(element.index(), element.index(), element.endtime());
    real rho = - element.timestep() * dfdu;
    real alpha = computeDamping(rho);
  */


}
//-----------------------------------------------------------------------------
bool NonStiffIteration::converged(TimeSlab& timeslab, 
				  Residuals& r, unsigned int n)
{
  // Convergence handled locally when the slab contains only one element list
  if ( timeslab.leaf() )
    return n >= 1;

  // Compute maximum discrete residual
  r.r1 = r.r2;
  r.r2 = residual(timeslab);

  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::converged(NewArray<Element*>& elements, 
				  Residuals& r, unsigned int n)
{
  // Convergence handled locally when the list contains only one element
  if ( elements.size() == 1 )
    return n >= 1;
  
  // Compute maximum discrete residual
  r.r1 = r.r2;
  r.r2 = residual(elements);

  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::converged(Element& element, 
				  Residuals& r, unsigned int n)
{
  // Compute discrete residual
  r.r1 = r.r2;
  r.r2 = residual(element);

  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
void NonStiffIteration::report() const
{
  cout << "System is non-stiff, solution computed with "
       << "simple fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
