// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/AdaptiveIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveIteration::AdaptiveIteration(Solution& u, RHS& f,
				     FixedPointIteration & fixpoint, 
				     real maxdiv, real maxconv, real tol) :
  Iteration(u, f, fixpoint, maxdiv, maxconv, tol), method(gauss_jacobi)
{
  // method = gauss_seidel;
}
//-----------------------------------------------------------------------------
AdaptiveIteration::~AdaptiveIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State AdaptiveIteration::state() const
{
  return adaptive;
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::start(TimeSlab& timeslab)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::start(NewArray<Element*>& elements)
{
  // Initialize data for Gauss-Jacobi iteration
  if ( method == gauss_jacobi )
    initData(elements);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::update(TimeSlab& timeslab, const Damping& d)
{
  // Simple update of time slab
  timeslab.update(fixpoint);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::update(NewArray<Element*>& elements, const Damping& d)
{
  // Choose update method
  if ( method == gauss_jacobi )
    updateGaussJacobi(elements, d);
  else
    updateGaussSeidel(elements, d);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::update(Element& element, const Damping& d)
{
  //real alpha = 0.01;
  real alpha = 1.0;

  // Choose update method
  if ( method == gauss_jacobi )
    element.update(f, alpha, values.values + values.offset);
  else
    element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::stabilize(TimeSlab& timeslab,
				   const Residuals& r, Damping& d)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::stabilize(NewArray<Element*>& elements,
				   const Residuals& r, Damping& d)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::stabilize(Element& element, 
				  const Residuals& r, Damping& d)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::converged(TimeSlab& timeslab, 
				   Residuals& r, unsigned int n)
{
  // Convergence handled locally when the slab contains only one element list
  if ( timeslab.leaf() )
    return n >= 1;

  // Compute maximum discrete residual
  r.r1 = r.r2;
  r.r2 = residual(timeslab);

  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::converged(NewArray<Element*>& elements, 
				   Residuals& r, unsigned int n)
{
  // Compute maximum discrete residual
  r.r1 = r.r2;
  r.r2 = residual(elements);
  
  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;

  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::converged(Element& element, 
				  Residuals& r, unsigned int n)
{
  // Iterate one time on each element
  return n > 0;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::diverged(TimeSlab& timeslab, 
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
  cout << "checking time slab convergence" << endl;
  dolfin_info("Adaptive damping is not enough, trying a stabilizing time step sequence.");
  
  // Check if we need to reset the element
  if ( r.r2 > r.r0 )
    timeslab.reset(fixpoint);

  // Change state
  newstate = nonnormal;
  
  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::diverged(NewArray<Element*>& elements, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  cout << "Checking element list convergence: " << r.r1 << " --> " << r.r2 << endl;

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
    reset(elements);

  // Change state
  newstate = nonnormal;

  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::diverged(Element& element, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Don't check divergence for elements
  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::report() const
{
  cout << "System is stiff, solution computed with adaptively stabilized "
       << "fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::updateGaussJacobi(NewArray<Element*>& elements,
					  const Damping& d)
{  
  // Reset offset
  values.offset = 0;

  cout << "Number of elements in list: " << elements.size() << endl;

  // Compute new values
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Iterate element
    fixpoint.iterate(*element);
    
    /*
      for(unsigned int j = 0; j < element->order() + 1; j++)
      {
      dolfin_debug2("value(%d): %lf", j, element->value(j));
      dolfin_debug2("newvalue(%d): %lf", j, newvalues[j + offset]);
      } 
    */

    // Increase offset
    values.offset += element->size();
  }

  // Copy values to elements
  copyData(elements);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::updateGaussSeidel(NewArray<Element*>& elements,
					  const Damping& d)
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
void AdaptiveIteration::initData(const NewArray<Element*>& elements)
{
  // Compute number of values
  unsigned int newsize = dataSize(elements);

  cout << "Number of values: " << newsize << endl;

  // Reallocate data if necessary
  if ( newsize > values.size )
    values.init(newsize);

  // Reset offset
  values.offset = 0;
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::copyData(NewArray<Element*>& elements) const
{
  // Copy data to elements in list
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
}
//-----------------------------------------------------------------------------
unsigned int AdaptiveIteration::dataSize(const NewArray<Element*>& elements) const
{
  // Compute number of values
  int size = 0;
  
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Add size of element
    size += element->size();
  }
  
  return size;
}
//-----------------------------------------------------------------------------
AdaptiveIteration::Values::Values() : values(0), size(0), offset(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
AdaptiveIteration::Values::~Values()
{
  if ( values )
    delete [] values;
  values = 0;
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::Values::init(unsigned int size)
{
  dolfin_assert(size > 0);

  if ( values )
    delete [] values;
  
  values = new real[size];
  dolfin_assert(values);
  this->size = size;
  offset = 0;
}
//-----------------------------------------------------------------------------
