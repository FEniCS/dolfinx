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
  Iteration(u, f, fixpoint, maxdiv, maxconv, tol), newvalues(0), offset(0)
{
  // Do nothing
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
void AdaptiveIteration::update(TimeSlab& timeslab, const Damping& d)
{
  // Simple update of time slab
  timeslab.update(fixpoint);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::update(NewArray<Element*>& elements, const Damping& d)
{
  // Compute total number of values in elements
  int numvalues = 0;

  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    numvalues += element->order() + 1;
  }

  newvalues = new real[numvalues];

  // Simple update of element list

  offset = 0;
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Iterate element
    fixpoint.iterate(*element);
    // Simple update of element
    
    //update(*element);
    //element->update(f, 1, newvalues + offset);

    for(unsigned int j = 0; j < element->order() + 1; j++)
    {
      dolfin_debug2("value(%d): %lf", j, element->value(j));
      dolfin_debug2("newvalue(%d): %lf", j, newvalues[j + offset]);
    }

    offset += element->order() + 1;
  }
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::update(Element& element, const Damping& d)
{
  real alpha = 1;

  // Simple update of element
  element.update(f, alpha, newvalues + offset);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::stabilize(TimeSlab& timeslab,
				   const Residuals& r, Damping& d)
{
  // Compute damping
  computeDamping(r, d);
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::stabilize(NewArray<Element*>& elements,
				   const Residuals& r, Damping& d)
{
  // Compute damping
  computeDamping(r, d);
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
  // Convergence handled locally when the list contains only one element
  if ( elements.size() == 1 )
    return n >= 1;
  
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
  return true;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::diverged(TimeSlab& timeslab, 
				   Residuals& r, unsigned int n,
				  Iteration::State &newstate)
{
  // Check if the solution diverges
  if ( r.r2 >= maxconv * r.r1 )
  {
    dolfin_debug("Problem appears to be stiff, trying parabolic damping.");

    newstate = nonnormal;
    return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::diverged(NewArray<Element*>& elements, 
				  Residuals& r, unsigned int n,
				  Iteration::State &newstate)
{
  // Check if the solution diverges
  if ( r.r2 >= maxconv * r.r1 )
  {
    dolfin_debug("Problem appears to be stiff, trying parabolic damping.");

    newstate = nonnormal;
    return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool AdaptiveIteration::diverged(Element& element, 
				  Residuals& r, unsigned int n,
				  Iteration::State &newstate)
{
  // Check if the solution diverges
  if ( r.r2 >= maxconv * r.r1 )
  {
    dolfin_debug("Problem appears to be stiff, trying parabolic damping.");

    newstate = nonnormal;
    return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveIteration::report() const
{
  cout << "System is stiff, solution computed with adaptively stabilized "
       << "fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
