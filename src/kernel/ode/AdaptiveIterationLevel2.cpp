// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/AdaptiveIterationLevel2.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIterationLevel2::AdaptiveIterationLevel2(Solution& u, RHS& f,
						 FixedPointIteration & fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv,
						 real tol, unsigned int depth) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol, depth)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
AdaptiveIterationLevel2::~AdaptiveIterationLevel2()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State AdaptiveIterationLevel2::state() const
{
  return stiff2;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::start(ElementGroupList& list)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::start(ElementGroup& group)
{
  // Initialize data for Gauss-Jacobi iteration
  initData(x1, dataSize(group));
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(ElementGroupList& list)
{
  // Simple update of time slab
  for (ElementGroupIterator group(list); !group.end(); ++group)
    fixpoint.iterate(*group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(ElementGroup& group)
{
  // Reset values
  x1.offset = 0;

  // Compute new values
  for (ElementIterator element(group); !element.end(); ++element)
    fixpoint.iterate(*element);
  
  // Copy values to elements
  copyData(x1, group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(Element& element)
{
  // Compute new values for element
  element.update(f, alpha, x1.values + x1.offset);
  x1.offset += element.size();
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(ElementGroupList& list,
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(ElementGroup& group,
					const Residuals& r, unsigned int n)
{
  // Stabilize if necessary
  if ( Iteration::stabilize(r, n) )
  {
    cout << "Need to stabilize time slab" << endl;
    
    // Compute divergence
    real rho = computeDivergence(group, r);
    
    // Compute alpha
    alpha = computeAlpha(rho);
    cout << "  alpha = " << alpha << endl;

    // Compute number of damping steps
    m = computeSteps(rho);
    j = m;
    cout << "  m     = " << m << endl;
    
    // Save residual at start of stabilizing iterations
    r0 = r.r2;
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(Element& element, 
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(ElementGroupList& list, 
					Residuals& r, unsigned int n)
{
  // Convergence handled locally when the slab contains only one element group
  if ( list.size() <= 1 )
    return n > 0;

  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(list);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(ElementGroup& group, 
					Residuals& r, unsigned int n)
{
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(group);
  
  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(Element& element, 
					Residuals& r, unsigned int n)
{
  // Iterate one time on each element
  return n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::diverged(ElementGroupList& list, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  cout << "Time slab residual: " << r.r1 << " --> " << r.r2 << endl;

  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  
  // Notify change of strategy
  dolfin_info("Not enough to stabilize element group iterations, need to stabilize time slab iterations.");
  
  // Check if we need to reset the group list
  if ( r.r2 > r.r0 )
    reset(list);

  // Change state
  newstate = stiff3;
  
  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::diverged(ElementGroup& group, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  // Don't check divergence for element group, since we want to handle
  // the stabilization ourselves (and not change state).
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::diverged(Element& element, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  // Don't check divergence for elements
  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::report() const
{
  cout << "System is stiff, solution computed with adaptively stabilized "
       << "fixed point iteration (on element group level)." << endl;
}
//-----------------------------------------------------------------------------
