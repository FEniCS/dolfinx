// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/Element.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/AdaptiveIterationLevel1.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIterationLevel1::AdaptiveIterationLevel1(Solution& u, RHS& f,
						 FixedPointIteration & fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv, 
						 real tol) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
AdaptiveIterationLevel1::~AdaptiveIterationLevel1()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State AdaptiveIterationLevel1::state() const
{
  return stiff1;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::start(TimeSlab& timeslab)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::start(ElementGroup& group)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::update(TimeSlab& timeslab)
{
  // Simple update of time slab
  timeslab.update(fixpoint);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::update(ElementGroup& group)
{
  // Simple update of element list
  for (ElementIterator element(group); !element.end(); ++element)
    fixpoint.iterate(*element);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::update(Element& element)
{
  // Damped update of element
  element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::stabilize(TimeSlab& timeslab,
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::stabilize(ElementGroup& group,
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::stabilize(Element& element, 
					const Residuals& r, unsigned int n)
{
  // Compute diagonal damping
  real dfdu = f.dfdu(element.index(), element.index(), element.endtime());
  real rho = - element.timestep() * dfdu;

  if ( rho >= 0.0 || rho < -1.0 )
    alpha = (1.0 + DOLFIN_SQRT_EPS) / (1.0 + rho);
  else
    alpha = 1.0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::converged(TimeSlab& timeslab, 
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
bool AdaptiveIterationLevel1::converged(ElementGroup& group, 
					Residuals& r, unsigned int n)
{
  // Convergence handled locally when the list contains only one element
  if ( group.size() == 1 )
    return n > 0;
  
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(group);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::converged(Element& element, 
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
bool AdaptiveIterationLevel1::diverged(TimeSlab& timeslab, 
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
  newstate = stiff3;

  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::diverged(ElementGroup& group, 
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
  reset(group);

  // Change state
  newstate = stiff2;

  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::diverged(Element& element, 
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
void AdaptiveIterationLevel1::report() const
{
  cout << "System appears to be diagonally stiff, solution computed with "
       << "diagonally damped fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
