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
  dolfin_error("Unreachable statement.");
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(ElementGroupList& list, Increments& d)
{
  // Iterate on each element group and compute the l2 norm of the increments
  real increment = 0.0;
  for (ElementGroupIterator group(list); !group.end(); ++group)
    increment += sqr(fixpoint.iterate(*group));

  d = sqrt(increment);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(ElementGroup& group, Increments& d)
{
  // Reset values
  x1.offset = 0;

  // Iterate on each element and compute the l2 norm of the increments
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
void AdaptiveIterationLevel2::update(Element& element, Increments& d)
{
  dolfin_error("Unreachable statement.");
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(ElementGroupList& list,
					const Increments& d, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(ElementGroup& group,
					const Increments& d, unsigned int n)
{
  // Stabilize if necessary
  if ( Iteration::stabilize(d, n) )
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
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(Element& element, 
					const Increments& d, unsigned int n)
{
  dolfin_error("Unreachable statement.");
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(ElementGroupList& list,
					const Increments& d, unsigned int n)
{
  // First check increment
  if ( d.d2 > tol || n == 0 )
    return false;
  
  // If increment is small, then check residual
  return residual(list) < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(ElementGroup& group,
					const Increments& d, unsigned int n)
{
  return d.d2 < tol && n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(Element& element,
					const Increments& d, unsigned int n)
{
  dolfin_error("Unreachable statement.");
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::diverged(ElementGroupList& list,
				       const Increments& d,
				       unsigned int n, State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( d.d2 < maxconv * d.d1 )
    return false;
  
  // Notify change of strategy
  dolfin_info("Not enough to stabilize element group iterations, need to stabilize time slab iterations.");
  
  // Reset group list
  reset(list);

  // Change state
  newstate = stiff3;
  
  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::diverged(ElementGroup& group, 
				       const Increments& d,
				       unsigned int n, State& newstate)
{
  // Don't check divergence for element groups, since we want to handle
  // the stabilization ourselves (and not change state).
  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::diverged(Element& element, 
				       const Increments& d,
				       unsigned int n, State& newstate)
{
  dolfin_error("Unreachable statement.");
  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::report() const
{
  cout << "System is stiff, solution computed with adaptively stabilized "
       << "fixed point iteration (on element group level)." << endl;
}
//-----------------------------------------------------------------------------
