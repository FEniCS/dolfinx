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
						 FixedPointIteration& fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv,
						 real tol, unsigned int depth,
						 bool debug_iter) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol, depth, debug_iter)
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
void AdaptiveIterationLevel3::update(ElementGroupList& list, Increments& d)
{
  dolfin_assert(depth() == 1);
  
  //updateGaussJacobi(list, d);
  updateGaussSeidel(list, d);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::updateGaussJacobi(ElementGroupList& list,
						Increments& d)
{
  // Reset values
  x1.offset = 0;

  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  real increment = 0.0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    real di = fabs(element->update(f, alpha, x1.values + x1.offset));
    increment += di*di;
    x1.offset += element->size();
  }
  d = sqrt(increment);
  
  // Copy values to elements
  copyData(x1, list);

  // Update initial data for all elements
  for (ElementIterator element(list); !element.end(); ++element)
    init(*element);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::updateGaussSeidel(ElementGroupList& list,
						Increments& d)
{
  // Reset values
  x1.offset = 0;
  
  // Initialize data for propagation of initial values
  ElementIterator element(list);
  initInitialData(element->starttime());

  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  real increment = 0.0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    // Update initial value for element
    element->update(u0.values[element->index()]);

    // Compute new values for element
    real di = fabs(element->update(f, alpha, x1.values + x1.offset));
    increment += di*di;
    x1.offset += element->size();
    
    // Save end value as new initial value for this component
    u0.values[element->index()] = element->endval();
  }
  d = sqrt(increment);

  // Copy values to elements
  copyData(x1, list);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(ElementGroup& group, Increments& d)
{
  dolfin_assert(depth() == 1);

  // Reset values
  x1.offset = 0;
    
  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  real increment = 0.0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    real di = fabs(element->update(f, alpha, x1.values + x1.offset));
    increment += di*di;
    x1.offset += element->size();
  }
  d = sqrt(increment);

  // Copy values to elements
  copyData(x1, group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(Element& element, Increments& d)
{
  dolfin_error("Unreachable statement.");
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(ElementGroupList& list,
					const Residuals& r,
					const Increments& d, unsigned int n)
{
  // Stabilize if necessary
  if ( Iteration::stabilize(r, d, n) )
  {
    // Compute divergence
    real rho = computeDivergence(list, r, d);
    
    // Compute alpha
    alpha = computeAlpha(rho);

    // Compute number of damping steps
    m = computeSteps(rho);
    j = m;
    
    // Save increment at start of stabilizing iterations
    r0 = d.d2;
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(ElementGroup& group,
					const Residuals& r,
					const Increments& d, unsigned int n)
{
  // Stabilize if necessary
  if ( Iteration::stabilize(r, d, n) )
  {
    // Compute divergence
    real rho = computeDivergence(group, r, d);
    
    // Compute alpha
    alpha = computeAlpha(rho);

    // Compute number of damping steps
    m = computeSteps(rho);
    j = m;
    
    // Save increment at start of stabilizing iterations
    r0 = d.d2;
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(Element& element, 
					const Residuals& r,
					const Increments& d, unsigned int n)
{
  dolfin_error("Unreachable statement.");
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(ElementGroupList& list, Residuals& r,
					const Increments& d, unsigned int n)
{
  /*
  // Compute residual
  r = residual(list);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol;
  */

  // First check increment
  if ( (d.d2/alpha) > tol || n == 0 )
    return false;
  
  // If increment is small, then check residual
  return residual(list) < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(ElementGroup& group, Residuals& r,
					const Increments& d, unsigned int n)
{
  /*
  // Compute residual
  r = residual(group);
  
  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol && n > 0;
  */

  return (d.d2 / alpha) < tol && n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(Element& element, Residuals& r,
					const Increments& d, unsigned int n)
{
  dolfin_error("Unreachable statement.");
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(ElementGroupList& list, 
				       const Residuals& r, const Increments& d,
				       unsigned int n, State& newstate)
{
  // Don't check divergence for group lists
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(ElementGroup& group, 
				       const Residuals& r, const Increments& d,
				       unsigned int n, State& newstate)
{
  // Don't check divergence for element groups
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(Element& element, 
				       const Residuals& r, const Increments& d,
				       unsigned int n, State& newstate)
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
