// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/NonStiffIteration.h>
#include <dolfin/AdaptiveIterationLevel1.h>
#include <dolfin/AdaptiveIterationLevel2.h>
#include <dolfin/AdaptiveIterationLevel3.h>
#include <dolfin/FixedPointIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f) : u(u), f(f)
{
  maxiter = dolfin_get("maximum iterations");
  maxdiv  = dolfin_get("maximum divergence");
  maxconv = dolfin_get("maximum convergence");

  // FIXME: Convergence should be determined by the error control
  tol = 1e-10;

  // Assume that the problem is non-stiff
  state = new NonStiffIteration(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
  //state = new AdaptiveIterationLevel3(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
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
  // Create a list of element groups from the time slab
  ElementGroupList list(timeslab);

  // Iterate on the list of element groups
  return iterate(list);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(ElementGroupList& list)
{
  Iteration::Residuals r;
  Iteration::State newstate;
  bool retry = true;

  //dolfin_start("Starting time slab iteration");

  while ( retry )
  {
    // Start iteration
    start(list);
    
    // Fixed point iteration on the group list
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(list, r, n) )
      {
	cout << "Time slab containing " << list.size()
	     << " elements converged in " << n << " iterations" << endl;
	//dolfin_end("Time slab iteration converged in %d iterations", n);
	end(list);
	return true;
      }

      //cout << "Time slab residual: " << r.r1 << " --> " << r.r2 << endl;
      
      // Check divergence
      if ( retry = diverged(list, r, n, newstate) )
      {
	changeState(newstate);
	break;
      }
      
      // Stabilize iteration
      stabilize(list, r, n);
      
      // Update group list
      update(list);
    }

    // End iteration
    end(list);
  }

  cout << "Time slab containing " << list.size()
       << " elements did not converge" << endl;
  //dolfin_end("Time slab did not converge");

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
    
    // Start iteration
    start(group);
    
    //dolfin_start("Starting element group iteration");
    
    // Fixed point iteration on the element group
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(group, r, n) )
      {
	//dolfin_end("Element group iteration converged in %d iterations", n);
	end(group);
	return true;
      }
      
      //cout << "Element group residual: " << r.r1 << " --> " << r.r2 << endl;

      // Check divergence
      if ( retry = diverged(group, r, n, newstate) )
      {
	changeState(newstate);
	break;
      }
    
      // Stabilize iteration
      stabilize(group, r, n);

      // Update element group
      update(group);
    }

    // End iteration
    end(group);
  }
  
  //dolfin_end("Element group iteration did not converge");

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
    // Start iteration
    start(element);
    
    // Fixed point iteration on the element
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(element, r, n) )
      {
	//dolfin_end("Element iteration converged in %d iterations", n + 1);
	end(element);
	return true;
      }
      
      // Check divergence
      if ( retry = diverged(element, r, n, newstate) )
      {
	changeState(newstate);
	break;
      }
    
      // Stabilize iteration
      stabilize(element, r, n);
      
      // Update element
      update(element);
    }

    // End iteration
    end(element);
  }

  return true;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset(ElementGroupList& list)
{
  dolfin_assert(state);
  state->reset(list);
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
void FixedPointIteration::stabilization(real& alpha, unsigned int& m) const
{
  dolfin_assert(state);
  state->stabilization(alpha, m);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::report() const
{
  dolfin_assert(state);
  state->report();
}
//-----------------------------------------------------------------------------
void FixedPointIteration::start(ElementGroupList& list)
{
  dolfin_assert(state);
  state->down();
  state->start(list);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::start(ElementGroup& group)
{
  dolfin_assert(state);
  state->down();
  state->start(group);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::start(Element& element)
{
  dolfin_assert(state);
  state->down();
  state->start(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::end(ElementGroupList& list)
{
  dolfin_assert(state);
  state->up();
  // Could be implemented if needed
  //state->end(list);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::end(ElementGroup& group)
{
  dolfin_assert(state);
  state->up();
  // Could be implemented if needed
  //state->end(group);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::end(Element& element)
{
  dolfin_assert(state);
  state->up();
  // Could be implemented if needed
  //state->end(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(ElementGroupList& list)
{
  dolfin_assert(state);
  state->update(list);
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
void FixedPointIteration::stabilize(ElementGroupList& list, 
				    const Iteration::Residuals& r,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(list, r, n);
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
bool FixedPointIteration::converged(ElementGroupList& list,
				    Iteration::Residuals& r, unsigned int n)
{
  dolfin_assert(state);
  return state->converged(list, r, n);
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
bool FixedPointIteration::diverged(ElementGroupList& list,
				   Iteration::Residuals& r, unsigned int n,
				   Iteration::State& newstate)
{
  dolfin_assert(state);
  return state->diverged(list, r, n, newstate);
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
void FixedPointIteration::changeState(Iteration::State newstate)
{
  dolfin_assert(state);

  // Don't change state if it hasn't changed
  if ( newstate == state->state() )
    return;

  // Save old depth
  unsigned int depth = state->depth();

  // Delete old state
  delete state;
  state = 0;

  // Initialize new state
  switch ( newstate ) {
  case Iteration::nonstiff:
    state = new NonStiffIteration(u, f, *this, maxiter, maxdiv, maxconv, tol, depth);
    break;
  case Iteration::stiff1:
    state = new AdaptiveIterationLevel1(u, f, *this, maxiter, maxdiv, maxconv, tol, depth);
    break;
  case Iteration::stiff2:
    state = new AdaptiveIterationLevel2(u, f, *this, maxiter, maxdiv, maxconv, tol, depth);
    break;
  case Iteration::stiff3:
    state = new AdaptiveIterationLevel3(u, f, *this, maxiter, maxdiv, maxconv, tol, depth);
    break;
  case Iteration::stiff:
    dolfin_error("Not implemented.");
    break;
  default:
    dolfin_error("Unknown state");
  }
}
//-----------------------------------------------------------------------------
