// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
#include <dolfin/AdaptiveIterationLevel3.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIterationLevel3::AdaptiveIterationLevel3(Solution& u, RHS& f,
						 FixedPointIteration & fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv,
						 real tol, unsigned int depth) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol, depth)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
AdaptiveIterationLevel3::~AdaptiveIterationLevel3()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State AdaptiveIterationLevel3::state() const
{
  return stiff3;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(ElementGroupList& list)
{
  // Initialize data for Gauss-Jacobi iteration
  initData(x1, dataSize(list));

  // FIXME: remove
  m = 0;
  j = 0;
  alpha = 1.0;
  reset(list);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(ElementGroup& group)
{
  // Initialize data for Gauss-Jacobi iteration
  initData(x1, dataSize(group));
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(ElementGroupList& list)
{
  dolfin_assert(depth() == 1);

  // Reset values
  x1.offset = 0;

  // Initialize data for propagation of initial values
  ElementIterator element(list);
  initInitialData(element->starttime());

  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  for (ElementIterator element(list); !element.end(); ++element)
  {
    // Update initial value for element
    element->update(u0.values[element->index()]);
    
    // Compute new values for element
    update(*element);
    
    // Save end value as new initial value for this component
    u0.values[element->index()] = element->endval();
  }

  //for (ElementIterator element(list); !element.end(); ++element)
  //  init(*element);
  
  // Copy values to elements
  copyData(x1, list);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(ElementGroup& group)
{
  dolfin_assert(depth() == 1);

  // Reset values
  x1.offset = 0;
    
  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  for (ElementIterator element(group); !element.end(); ++element)
    update(*element);
  
  // Copy values to elements
  copyData(x1, group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(Element& element)
{
  // Compute new values for element
  element.update(f, alpha, x1.values + x1.offset);
  x1.offset += element.size();
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(ElementGroupList& list,
					const Residuals& r, unsigned int n)
{
  // Stabilize if necessary
  if ( Iteration::stabilize(r, n) )
  {
    cout << "Need to stabilize time slab" << endl;
    
    // Compute divergence
    real rho = computeDivergence(list, r);
    
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
void AdaptiveIterationLevel3::stabilize(ElementGroup& group,
					const Residuals& r, unsigned int n)
{
  // Stabilize if necessary
  if ( Iteration::stabilize(r, n) )
  {
    cout << "Need to stabilize element group" << endl;
    
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
void AdaptiveIterationLevel3::stabilize(Element& element, 
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(ElementGroupList& list, 
					Residuals& r, unsigned int n)
{
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(list);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(ElementGroup& group, 
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
bool AdaptiveIterationLevel3::converged(Element& element, 
					Residuals& r, unsigned int n)
{
  // We should not reach this statement
  dolfin_assert(false);

  // Iterate one time on each element
  return n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(ElementGroupList& list, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  // Don't check divergence for element group, since we want to handle
  // the stabilization ourselves (and not change state).
  return false;
  
  /*

  // Make at least two iterations
  if ( n < 2 )
    return false;
  
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  
  // Notify change of strategy
  dolfin_info("Adaptive damping is not enough, trying a stabilizing time step sequence.");
  
  // Check if we need to reset the group list
  if ( r.r2 > r.r0 )
    reset(list);

  // Change state
  newstate = stiff;
  
  return true;
  */
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(ElementGroup& group, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  // Don't check divergence for element groups
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(Element& element, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  // Don't check divergence for elements
  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::report() const
{
  cout << "System is stiff, solution computed with adaptively stabilized "
       << "fixed point iteration (on time slab level)." << endl;
}
//-----------------------------------------------------------------------------
