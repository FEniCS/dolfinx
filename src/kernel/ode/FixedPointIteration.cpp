// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/NonStiffIteration.h>
#include <dolfin/AdaptiveIterationLevel1.h>
#include <dolfin/AdaptiveIterationLevel2.h>
#include <dolfin/AdaptiveIterationLevel3.h>
#include <dolfin/FixedPointIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f) : u(u), f(f)
{
  maxiter       = dolfin_get("maximum iterations");
  maxdiv        = dolfin_get("maximum divergence");
  maxconv       = dolfin_get("maximum convergence");

  // FIXME: Convergence should be determined by the error control
  tol = 1e-10;

  // Assume that problem is non-stiff
  state = new NonStiffIteration(u, f, *this, maxiter, maxdiv, maxconv, tol);
}
//-----------------------------------------------------------------------------
FixedPointIteration::~FixedPointIteration()
{
  if ( state )
    delete state;
  state = 0;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(TimeSlab& timeslab)
{
  Iteration::Residuals r;
  Iteration::State newstate;
  bool retry = true;
  
  dolfin_start("Starting time slab iteration");
  
  while ( retry )
  {
    // Start iteration on time slab
    start(timeslab);
    
    // Fixed point iteration on the time slab  
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(timeslab, r, n) )
      {
	dolfin_end("Time slab iteration converged in %d iterations", n + 1);
	return true;
      }
      
      // Check divergence
      if ( retry = diverged(timeslab, r, n, newstate) )
      {
	changeState(newstate);
	break;
      }
      
      // Stabilize iteration
      stabilize(timeslab, r, n);
      
      // Update time slab
      update(timeslab);
    }
  }

  dolfin_end("Time slab iteration did not converge");

  return false;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(ElementGroup& group)
{
  Iteration::Residuals r;
  Iteration::State newstate;
  bool retry = true;
  
  while ( retry )
  {
    // Update initial data
    init(group);
    
    // Start iteration on element group
    start(group);
    
    dolfin_start("Starting element group iteration");
    
    // Fixed point iteration on the element group
    for (unsigned int n = 0; n < maxiter; n++)
    {
      /*
	for (unsigned int i = 0; i < elements.size(); i++)
	{
	// Get the element
	Element* element = elements[i];
	dolfin_assert(element);
	
	for(unsigned int j = 0; j < element->order() + 1; j++)
	{
	dolfin_debug3("value(%d, %d): %lf", i, j, element->value(j));
	}
	}
      */
      
      //dolfin_debug3("r0: %lf r1: %lf r2: %lf", r.r0, r.r1, r.r2);
      
      // Check convergence
      if ( converged(group, r, n) )
      {
	dolfin_end("Element group iteration converged in %d iterations", n + 1);
	return true;
      }
      
      //dolfin_debug3("r0: %lf r1: %lf r2: %lf", r.r0, r.r1, r.r2);
      
      // Check divergence
      if ( retry = diverged(group, r, n, newstate) )
      {
	dolfin_end("Changing state");
	changeState(newstate);
	break;
      }
    
      // Stabilize iteration
      stabilize(group, r, n);
      
      /*
	for (unsigned int i = 0; i < elements.size(); i++)
	{
	// Get the element
	Element* element = elements[i];
	dolfin_assert(element);
	
	for(unsigned int j = 0; j < element->order() + 1; j++)
	{
	dolfin_debug3("value(%d, %d): %lf", i, j, element->value(j));
	}
	}
      */
      
      // Update element group
      update(group);
    }
  }
  
  dolfin_end("Element group iteration did not converge");

  return false;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(Element& element)
{
  Iteration::Residuals r;
  Iteration::State newstate;
  bool retry = true;

  //dolfin_start("Starting element iteration");

  while ( retry )
  {
    // Start iteration on element
    start(element);
    
    // Fixed point iteration on the element
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(element, r, n) )
      {
	//dolfin_end("Element iteration converged in %d iterations", n + 1);
	//cout << "u(" << element.index() << "," << element.endtime() << ") = " << element.endval() << endl;
	return true;
      }
      
      //dolfin_debug3("r0: %lf r1: %lf r2: %lf", r.r0, r.r1, r.r2);
      
      // Check divergence
      if ( retry = diverged(element, r, n, newstate) )
      {
	changeState(newstate);
	break;
      }
    
      // Stabilize iteration
      stabilize(element, r, n);
      
      //dolfin_debug2("value(%d): %lf", element.index(), element.value((unsigned int)0));
      
      // Update element
      update(element);
      //dolfin_debug2("value(%d): %lf", element.index(), element.value((unsigned int)0));
      
    }
    break;
  }

  //dolfin_end("Element iteration did not converge");

  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(TimeSlab& timeslab)
{
  dolfin_assert(state);
  return state->residual(timeslab);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(ElementGroup& group)
{
  dolfin_assert(state);
  return state->residual(group);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(Element& element)
{
  dolfin_assert(state);
  return state->residual(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::init(ElementGroup& group)
{
  dolfin_assert(state);
  state->init(group);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::init(Element& element)
{
  dolfin_assert(state);
  state->init(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset(ElementGroup& group)
{
  dolfin_assert(state);
  state->reset(group);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset(Element& element)
{
  dolfin_assert(state);
  state->reset(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::report() const
{
  dolfin_assert(state);
  state->report();
}
//-----------------------------------------------------------------------------
void FixedPointIteration::start(TimeSlab& timeslab)
{
  dolfin_assert(state);
  state->start(timeslab);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::start(ElementGroup& group)
{
  dolfin_assert(state);
  state->start(group);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::start(Element& element)
{
  dolfin_assert(state);
  state->start(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(TimeSlab& timeslab)
{
  dolfin_assert(state);
  state->update(timeslab);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(ElementGroup& group)
{
  dolfin_assert(state);
  state->update(group);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(Element& element)
{
  dolfin_assert(state);
  u.debug(element, Solution::update);
  state->update(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab, 
				    const Iteration::Residuals& r,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(timeslab, r, n);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(ElementGroup& group, 
				    const Iteration::Residuals& r,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(group, r, n);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(Element& element, 
				    const Iteration::Residuals& r,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(element, r, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(TimeSlab& timeslab,
				    Iteration::Residuals& r, unsigned int n)
{
  dolfin_assert(state);
  return state->converged(timeslab, r, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(ElementGroup& group,
				    Iteration::Residuals& r, unsigned int n)
{
  dolfin_assert(state);
  return state->converged(group, r, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(Element& element, Iteration::Residuals& r, 
				    unsigned int n)
{ 
  dolfin_assert(state);
  return state->converged(element, r, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::diverged(TimeSlab& timeslab,
				   Iteration::Residuals& r, unsigned int n,
				   Iteration::State& newstate)
{
  dolfin_assert(state);
  return state->diverged(timeslab, r, n, newstate);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::diverged(ElementGroup& group,
				   Iteration::Residuals& r, unsigned int n,
				   Iteration::State& newstate)
{
  dolfin_assert(state);
  return state->diverged(group, r, n, newstate);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::diverged(Element& element, Iteration::Residuals& r, 
				   unsigned int n,
				   Iteration::State& newstate)
{ 
  dolfin_assert(state);
  return state->diverged(element, r, n, newstate);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::changeState(Iteration::State newstate)
{
  dolfin_assert(state);

  // Don't change state if it hasn't changed
  if ( newstate == state->state() )
    return;

  // Delete old state
  delete state;
  state = 0;

  // Initialize new state
  switch ( newstate ) {
  case Iteration::nonstiff:
    state = new NonStiffIteration(u, f, *this, maxiter, maxdiv, maxconv, tol);
    break;
  case Iteration::stiff1:
    state = new AdaptiveIterationLevel1(u, f, *this, maxiter, maxdiv, maxconv, tol);
    break;
  case Iteration::stiff2:
    state = new AdaptiveIterationLevel2(u, f, *this, maxiter, maxdiv, maxconv, tol);
    break;
  case Iteration::stiff3:
    state = new AdaptiveIterationLevel3(u, f, *this, maxiter, maxdiv, maxconv, tol);
    break;
  case Iteration::stiff:
    dolfin_error("Not implemented.");
    break;
  default:
    dolfin_error("Unknown state");
  }
}
//-----------------------------------------------------------------------------
