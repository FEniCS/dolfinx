// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/AdaptiveIterationLevel3.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIterationLevel3::AdaptiveIterationLevel3(Solution& u, RHS& f,
						 FixedPointIteration & fixpoint, 
						 unsigned int maxiter,
						 real maxdiv, real maxconv,
						 real tol) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol), datasize(0)
{
  // method = gauss_seidel;
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
void AdaptiveIterationLevel3::start(TimeSlab& timeslab)
{
  // Compute total number of values in time slab
  datasize = dataSize(timeslab);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(NewArray<Element*>& elements)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(TimeSlab& timeslab)
{
  // Choose update method
  if ( method == gauss_jacobi )
    updateGaussJacobi(timeslab);
  else
    updateGaussSeidel(timeslab);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::update(NewArray<Element*>& elements)
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
void AdaptiveIterationLevel3::update(Element& element)
{
  // Choose update method
  if ( method == gauss_jacobi )
  {
    element.update(f, alpha, x1.values + x1.offset);
    x1.offset += element.size();
  }
  else
    element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(TimeSlab& timeslab,
					const Residuals& r, unsigned int n)
{
  // Make at least one iteration before stabilizing
  if ( n < 1 )
    return;

  // Compute divergence rate if necessary
  real rho = 0.0;
  if ( r.r2 > r.r1 && j == 0 )
    rho = computeDivergence(timeslab, r);
  
  // Adaptive stabilization
  Iteration::stabilize(r, rho);
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(NewArray<Element*>& elements,
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::stabilize(Element& element, 
					const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(TimeSlab& timeslab, 
					Residuals& r, unsigned int n)
{
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(timeslab);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(NewArray<Element*>& elements, 
					Residuals& r, unsigned int n)
{
  // Iterate one time on each element list
  return n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::converged(Element& element, 
					Residuals& r, unsigned int n)
{
  // Iterate one time on each element
  return n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(TimeSlab& timeslab, 
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
  
  // Check if we need to reset the element
  if ( r.r2 > r.r0 )
    timeslab.reset(fixpoint);

  // Change state
  newstate = stiff;
  
  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIterationLevel3::diverged(NewArray<Element*>& elements, 
				       Residuals& r, unsigned int n,
				       Iteration::State& newstate)
{
  // Don't check divergence for element lists
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
void AdaptiveIterationLevel3::updateGaussJacobi(TimeSlab& timeslab)
{  


  /*
  // Initialize values
  initData(x1);

  // Compute new values
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Iterate element
    fixpoint.iterate(*element);
    
    // Increase offset
    x1.offset += element->size();
    }

    // Copy values to elements
    copyData(x1, elements);
  */

}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::updateGaussSeidel(TimeSlab& timeslab)
{
  // Simple update of element list
  /*
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Iterate element
    fixpoint.iterate(*element);
  }
  */
}
//-----------------------------------------------------------------------------
real AdaptiveIterationLevel3::computeDivergence(TimeSlab& timeslab,
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
  copyData(timeslab, x0);

  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Update time slab
    update(timeslab);
    
    // Compute residual
    r1 = r2;
    r2 = residual(timeslab);
  
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
  copyData(x0, timeslab);

  return rho2;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::initData(Values& values)
{
  // Reallocate data if necessary
  if ( datasize > values.size )
    values.init(datasize);

  // Reset offset
  values.offset = 0;
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::copyData(const TimeSlab& timeslab,
				       Values& values)
{

  /*
  // Copy data from element list
  unsigned int offset = 0;
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Get values from element
    element->get(values.values + offset);

    // Increase offset
    offset += element->size();
  }
  */
}
//-----------------------------------------------------------------------------
void AdaptiveIterationLevel3::copyData(const Values& values,
				       TimeSlab& timeslab) const
{
  /*
  // Copy data to elements list
  unsigned int offset = 0;
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Set values for element
    element->set(values.values + offset);

    // Increase offset
    offset += element->size();
  }
  */
}
//-----------------------------------------------------------------------------
unsigned int AdaptiveIterationLevel3::dataSize(const TimeSlab& timeslab) const
{
  // Compute number of values
  int size = 0;
  
  /*
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Add size of element
    size += element->size();
  }
  */
  
  return size;
}
//-----------------------------------------------------------------------------
