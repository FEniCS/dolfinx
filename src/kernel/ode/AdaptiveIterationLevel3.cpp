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
  initData(x0, dataSize(list));
  initData(x1, dataSize(list));
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(ElementGroup& group)
{
  // Initialize data for Gauss-Jacobi iteration
  initData(x0, dataSize(group));
  initData(x1, dataSize(group));
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(Element& element)
{
  dolfin_error("Unreachable statement.");
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
  // Save values before iteration (to restore when diverging)
  copyData(list, x0);

  // Reset values
  x1.offset = 0;

  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  real increment = 0.0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    real di = fabs(element->update(f, alpha, x1.values + x1.offset));
    di /= alpha;
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
  // Save values before iteration (to restore when diverging)
  copyData(list, x0);

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
    di /= alpha;
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

  // Save values before iteration (to restore when diverging)
  copyData(group, x0);

  // Reset values
  x1.offset = 0;
    
  // Compute new values. Note that we skip the recursive iteration,
  // we directly update all elements without calling iterate on
  // all element groups contained in the group list.
  real increment = 0.0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    real di = fabs(element->update(f, alpha, x1.values + x1.offset));
    di /= alpha;
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
					const Increments& d, unsigned int n)
{
  // Compute residual (needed for j = 0)
  real r = (j == 0 ? residual(list) : 0.0);

  // Stabilize if necessary
  if ( Iteration::stabilize(d, n, r) )
  {
    // Compute divergence
    real rho = computeDivergence(list);
    
    // Compute alpha
    alpha = computeAlpha(rho);

    // Compute number of damping steps
    m = computeSteps(rho);
    j = m;
    
    // Save increment at start of stabilizing iterations
    d0 = d.d2;

    // Check if we should keep the solution
    if ( !_accept )
      copyData(x0, list);
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(ElementGroup& group,
					const Increments& d, unsigned int n)
{
  // Compute residual (needed for j = 0)
  real r = (j == 0 ? residual(group) : 0.0);

  // Stabilize if necessary
  if ( Iteration::stabilize(d, n, r) )
  {
    // Compute divergence
    real rho = computeDivergence(group);
    
    // Compute alpha
    alpha = computeAlpha(rho);

    // Compute number of damping steps
    m = computeSteps(rho);
    j = m;
    
    // Save increment at start of stabilizing iterations
    d0 = d.d2;

    // Restore the solution if necessary
    if ( !_accept )
      copyData(x0, group);
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(Element& element, 
					const Increments& d, unsigned int n)
{
  dolfin_error("Unreachable statement.");
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(ElementGroupList& list,
					const Increments& d, unsigned int n)
{
  // First check increment
  if ( d.d2 > tol || n == 0 )
    return false;
  
  // If increment is small, then check residual
  return residual(list) < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(ElementGroup& group,
					const Increments& d, unsigned int n)
{
  return d.d2 < tol && n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(Element& element,
					const Increments& d, unsigned int n)
{
  dolfin_error("Unreachable statement.");
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(ElementGroupList& list,
				       const Increments& d,
				       unsigned int n, State& newstate)
{
  // Don't check divergence for group lists, since we want to handle
  // the stabilization ourselves (and not change state).
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(ElementGroup& group, 
				       const Increments& d,
				       unsigned int n, State& newstate)
{
  // Don't check divergence for element groups, since we want to handle
  // the stabilization ourselves (and not change state).
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(Element& element, 
				       const Increments& d,
				       unsigned int n, State& newstate)
{
  dolfin_error("Unreachable statement.");
  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::report() const
{
  cout << "System is stiff, solution computed with adaptively stabilized "
       << "fixed point iteration (on time slab level)." << endl;
}
//-----------------------------------------------------------------------------
