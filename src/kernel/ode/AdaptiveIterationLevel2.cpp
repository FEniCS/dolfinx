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
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/AdaptiveIterationLevel2.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIterationLevel2::AdaptiveIterationLevel2(Solution& u, RHS& f,
						 FixedPointIteration & fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv,
						 real tol) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol), datasize(0)
{
  // method = gauss_seidel;
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
void AdaptiveIterationLevel2::start(ElementGroupList& groups)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::start(ElementGroup& group)
{
  // Compute total number of values in element group
  datasize = dataSize(group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(ElementGroupList& groups)
{
  // Simple update of time slab
  for (ElementGroupIterator group(groups); !group.end(); ++group)
    fixpoint.iterate(*group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(ElementGroup& group)
{
  // Choose update method
  if ( method == gauss_jacobi )
    updateGaussJacobi(group);
  else
    updateGaussSeidel(group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::update(Element& element)
{
  // Choose update method
  if ( method == gauss_jacobi )
    element.update(f, alpha, x1.values + x1.offset);
  else
    element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(ElementGroupList& groups,
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(ElementGroup& group,
					const Residuals& r, unsigned int n)
{
  // Make at least one iteration before stabilizing
  if ( n < 1 )
    return;

  // Compute divergence rate if necessary
  real rho = 0.0;
  if ( r.r2 > r.r1 && j == 0 )
    rho = computeDivergence(group, r);
  
  // Adaptive stabilization
  Iteration::stabilize(r, rho);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::stabilize(Element& element, 
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel2::converged(ElementGroupList& groups, 
					Residuals& r, unsigned int n)
{
  // Convergence handled locally when the slab contains only one element list
  if ( groups.size() <= 1 )
    return n > 0;

  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(groups);

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
bool AdaptiveIterationLevel2::diverged(ElementGroupList& groups, 
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
  dolfin_info("Adaptive damping is not enough, trying a stabilizing time step sequence.");
  
  // Check if we need to reset all element groups
  if ( r.r2 > r.r0 )
    fixpoint.reset(groups);

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
void AdaptiveIterationLevel2::updateGaussJacobi(ElementGroup& group)
{  
  // Initialize values
  initData(x1);

  // Compute new values
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Iterate element
    fixpoint.iterate(*element);
    
    // Increase offset
    x1.offset += element->size();
  }

  // Copy values to elements
  copyData(x1, group);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::updateGaussSeidel(ElementGroup& group)
{
  // Simple update of element list
  for (ElementIterator element(group); !element.end(); ++element)
    fixpoint.iterate(*element);
}
//-----------------------------------------------------------------------------
real AdaptiveIterationLevel2::computeDivergence(ElementGroup& group,
						const Residuals& r)
{
  // Successive residuals
  real r1 = r.r1;
  real r2 = r.r2;

  // Successive convergence factors
  real rho2 = r2 / r1;
  real rho1 = rho2;

  // Save current alpha and change alpha to 1 for divergence computation
  real alpha0 = alpha;
  alpha = 1.0;

  // Save solution values before iteration
  initData(x0);
  copyData(group, x0);

  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Update element group
    update(group);
    
    // Compute residual
    r1 = r2;
    r2 = residual(group);
  
    // Compute divergence
    rho1 = rho2;
    rho2 = r2 / (DOLFIN_EPS + r1);

    cout << "rho = " << rho2 << endl;

    // Check if the divergence factor has converged
    if ( abs(rho2-rho1) < 0.1 * rho1 )
    {
      dolfin_debug1("Computed divergence rate in %d iterations", n + 1);
      break;
    }
    
  }

  // Restore alpha
  alpha = alpha0;

  // Restore solution values
  copyData(x0, group);

  return rho2;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::initData(Values& values)
{
  // Reallocate data if necessary
  if ( datasize > values.size )
    values.init(datasize);

  // Reset offset
  values.offset = 0;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::copyData(ElementGroup& group,
				       Values& values)
{
  // Copy data from element list
  unsigned int offset = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Get values from element
    element->get(values.values + offset);

    // Increase offset
    offset += element->size();
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel2::copyData(Values& values, ElementGroup& group)
{
  // Copy data to elements list
  unsigned int offset = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Set values for element
    element->set(values.values + offset);

    // Increase offset
    offset += element->size();
  }
}
//-----------------------------------------------------------------------------
unsigned int AdaptiveIterationLevel2::dataSize(ElementGroup& group)
{
  // Compute number of values
  int size = 0;  
  for (ElementIterator element(group); !element.end(); ++element)
    size += element->size();
  
  return size;
}
//-----------------------------------------------------------------------------
