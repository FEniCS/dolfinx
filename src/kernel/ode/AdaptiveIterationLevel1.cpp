// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/Element.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/AdaptiveIterationLevel1.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIterationLevel1::AdaptiveIterationLevel1(Solution& u, RHS& f,
						 FixedPointIteration & fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv, 
						 real tol, unsigned int depth) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol, depth)
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
void AdaptiveIterationLevel1::start(ElementGroupList& list)
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
void AdaptiveIterationLevel1::update(ElementGroupList& list, Increments& d)
{
  // Iterate on each element group and compute the l2 norm of the increments
  real increment = 0.0;
  for (ElementGroupIterator group(list); !group.end(); ++group)
    increment += sqr(fixpoint.iterate(*group));

  d = sqrt(increment);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::update(ElementGroup& group, Increments& d)
{
  // Iterate on each element and compute the l2 norm of the increments
  real increment = 0.0;
  for (ElementIterator element(group); !element.end(); ++element)
    increment += sqr(fixpoint.iterate(*element));

  d = sqrt(increment);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::update(Element& element, Increments& d)
{
  // Damped update of element
  d = fabs(element.update(f, alpha));
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel1::stabilize(ElementGroupList& list,
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
bool AdaptiveIterationLevel1::converged(ElementGroupList& list, Residuals& r,
					const Increments& d, unsigned int n)
{
  /*
  // Convergence handled locally when the slab contains only one element group
  if ( list.size() <= 1 )
    return n > 0;
  
  // Compute residual
  r = residual(list);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
  */

  // First check increment
  if ( d.d2 > tol || n == 0 )
    return false;
  
  // If increment is small, then check residual
  return residual(list) < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::converged(ElementGroup& group, Residuals& r,
					const Increments& d, unsigned int n)
{
  /*
  // Convergence handled locally when the group contains only one element
  if ( group.size() == 1 )
    return n > 0;

  // Compute residual
  r = residual(group);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
  */
  
  return d.d2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::converged(Element& element, Residuals& r,
					const Increments& d, unsigned int n)
{
  /*
  // Compute residual
  r = residual(element);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
  */

  return d.d2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::diverged(ElementGroupList& list, 
				       const Residuals& r, const Increments& d,
				       unsigned int n, State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;
  
  /*
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  */
  
  // Check if the solution converges
  if ( d.d2 < maxconv * d.d1 )
    return false;

  // Notify change of strategy
  dolfin_info("Not enough to stabilize element iterations, need to stabilize time slab iterations.");
  
  // Reset group list
  reset(list);

  // Change state
  newstate = stiff3;
  
  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::diverged(ElementGroup& group, 
				       const Residuals& r, const Increments& d,
				       unsigned int n, State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  /*
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  */

  // Check if the solution converges
  if ( d.d2 < maxconv * d.d1 )
    return false;
  
  // Notify change of strategy
  dolfin_info("Not enough to stabilize element iterations, need to stabilize element group iterations.");
  
  // Reset element group
  reset(group);

  // Change state
  newstate = stiff2;

  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel1::diverged(Element& element, 
				       const Residuals& r, const Increments& d,
				       unsigned int n, State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  /*
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  */
  
  // Check if the solution converges
  if ( d.d2 < maxconv * d.d1 )
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
