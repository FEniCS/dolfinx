// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/DiagonalIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DiagonalIteration::DiagonalIteration(Solution& u, RHS& f,
				     FixedPointIteration & fixpoint, 
				     real maxdiv, real maxconv, real tol) :
  Iteration(u, f, fixpoint, maxdiv, maxconv, tol)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DiagonalIteration::~DiagonalIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State DiagonalIteration::state() const
{
  return diagonal;
}
//-----------------------------------------------------------------------------
void DiagonalIteration::update(TimeSlab& timeslab)
{
  // Simple update of time slab
  timeslab.update(fixpoint);
}
//-----------------------------------------------------------------------------
void DiagonalIteration::update(NewArray<Element*>& elements)
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
void DiagonalIteration::update(Element& element)
{
  // Compute diagonal derivative
  real dfdu = f.dfdu(element.index(), element.index(), element.endtime());
  real rho = - element.timestep() * dfdu;
  real alpha = computeAlpha(rho);

  // Simple update of element
  element.update(f, alpha);
}
//-----------------------------------------------------------------------------
Iteration::State DiagonalIteration::stabilize(TimeSlab& timeslab,
					      const Residuals& r, Damping& d)
{
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return diagonal;

  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, need to use a stabilizing time step sequence.");
  
  // Check if we need to reset the element
  if ( r.r2 > r.r0 )
    timeslab.reset(fixpoint);

  // Compute damping
  computeDamping(r, d);

  // Change state
  return nonnormal;
}
//-----------------------------------------------------------------------------
Iteration::State DiagonalIteration::stabilize(NewArray<Element*>& elements,
					      const Residuals& r, Damping& d)
{
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return diagonal;

  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, trying parabolic damping.");
  
  // Check if we need to reset the element
  if ( r.r2 > r.r0 )
    reset(elements);

  // Compute damping
  computeDamping(r, d);

  // Change state
  return parabolic;
}
//-----------------------------------------------------------------------------
Iteration::State DiagonalIteration::stabilize(Element& element, 
					      const Residuals& r, Damping& d)
{
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return diagonal;
  
  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, don't know what to do.");
  dolfin_error("Local iterations did not converge.");

  // Change state (statement out of reach)
  return diagonal;
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::converged(TimeSlab& timeslab, 
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
bool DiagonalIteration::converged(NewArray<Element*>& elements, 
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
bool DiagonalIteration::converged(Element& element, 
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
void DiagonalIteration::report() const
{
  cout << "System appears to be diagonally stiff, solution computed with "
       << "diagonally damped fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
