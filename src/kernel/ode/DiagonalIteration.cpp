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
				     unsigned int maxiter,
				     real maxdiv, real maxconv, real tol) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol), alpha(1)
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
void DiagonalIteration::start(TimeSlab& timeslab)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DiagonalIteration::start(NewArray<Element*>& elements)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DiagonalIteration::start(Element& element)
{
  // Do nothing
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
  // Damped update of element
  element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void DiagonalIteration::stabilize(TimeSlab& timeslab,
				  const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DiagonalIteration::stabilize(NewArray<Element*>& elements,
				  const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DiagonalIteration::stabilize(Element& element, 
				  const Residuals& r, unsigned int n)
{
  // Compute diagonal damping
  real dfdu = f.dfdu(element.index(), element.index(), element.endtime());
  real rho = - element.timestep() * dfdu;
  alpha = computeAlpha(rho);
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::converged(TimeSlab& timeslab, 
				   Residuals& r, unsigned int n)
{
  // Convergence handled locally when the slab contains only one element list
  if ( timeslab.leaf() )
    return n > 0;
  
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(timeslab);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::converged(NewArray<Element*>& elements, 
				   Residuals& r, unsigned int n)
{
  // Convergence handled locally when the list contains only one element
  if ( elements.size() == 1 )
    return n > 0;
  
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(elements);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::converged(Element& element, 
				  Residuals& r, unsigned int n)
{
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(element);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::diverged(TimeSlab& timeslab, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;

  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, trying a stabilizing time step sequence.");
  
  // Reset time slab
  timeslab.reset(fixpoint);

  // Change state
  newstate = nonnormal;

  return true;
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::diverged(NewArray<Element*>& elements, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  
  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, trying adaptive damping.");
  
  // Reset element list
  reset(elements);

  // Change state
  newstate = adaptive;

  return true;
}
//-----------------------------------------------------------------------------
bool DiagonalIteration::diverged(Element& element, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  
  // Don't know what to do
  dolfin_error("Local iterations did not converge.");

  return true;
}
//-----------------------------------------------------------------------------
void DiagonalIteration::report() const
{
  cout << "System appears to be diagonally stiff, solution computed with "
       << "diagonally damped fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
real DiagonalIteration::computeAlpha(real rho) const
{
  if ( rho >= 0.0 || rho < -1.0 )
    return (1.0 + DOLFIN_SQRT_EPS) / (1.0 + rho);

  return 1.0;
}
//-----------------------------------------------------------------------------
