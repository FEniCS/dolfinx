// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/Adaptivity.h>
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
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f,
					 Adaptivity& adaptivity) : u(u), f(f)
{
  maxiter    = dolfin_get("maximum iterations");
  maxdiv     = dolfin_get("maximum divergence");
  maxconv    = dolfin_get("maximum convergence");
  debug_iter = dolfin_get("debug iterations");

  // FIXME: Convergence should be determined by the error control
  tol = dolfin_get("tolerance");
  tol *= 0.01;

  cout << "Fixed point tolerance: " << tol << endl;

  // Choose initial stiffness
  std::string stiffness = dolfin_get("stiffness");
  if ( stiffness == "non-stiff" )
    state = new NonStiffIteration(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
  else if ( stiffness == "stiff level 1" )
    state = new AdaptiveIterationLevel1(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
  else if ( stiffness == "stiff level 2" )
    state = new AdaptiveIterationLevel2(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
  else if ( stiffness == "stiff level 3" )
    state = new AdaptiveIterationLevel3(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
  else
  {
    dolfin_warning1("Unknown stiffness: %s, assuming problem is non-stiff.", stiffness.c_str());
    state = new NonStiffIteration(u, f, *this, maxiter, maxdiv, maxconv, tol, 0);
  }  
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
  Iteration::Increments d;
  bool retry = true;

  while ( retry )
  {
    // Start iteration
    start(list);

    // Fixed point iteration on the group list
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(list, d, n) )
      {
	end(list);
	return true;
      }
      
      // Check divergence
      Iteration::State newstate;
      if ( retry = diverged(list, d, n, newstate) )
      {
	changeState(newstate);
	d.reset();
	break;
      }
      
      // Stabilize iteration
      stabilize(list, d, n);
      
      // Write debug info
      if ( debug_iter )
	debug(list, d, n);

      // Update group list
      update(list, d);
    }

    // End iteration
    end(list);
  }

  return false;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::iterate(ElementGroup& group)
{
  Iteration::Increments d;
  bool retry = true;

  while ( retry )
  {
    // Update initial data
    init(group);
    
    // Start iteration
    start(group);

    // Fixed point iteration on the element group
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(group, d, n) )
      {
	end(group);
	return d.dtot;
      }
      
      // Check divergence
      Iteration::State newstate;
      if ( retry = diverged(group, d, n, newstate) )
      {
	changeState(newstate);
	d.reset();
	break;
      }
    
      // Stabilize iteration
      stabilize(group, d, n);

      // Write debug info
      if ( debug_iter )
	debug(group, d, n);

      // Update element group
      update(group, d);
    }

    // End iteration
    end(group);
  }

  return d.dtot;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::iterate(Element& element)
{
  Iteration::Increments d;
  bool retry = true;

  while ( retry )
  {
    // Start iteration
    start(element);

    // Fixed point iteration on the element
    for (unsigned int n = 0; n < maxiter; n++)
    {
      // Check convergence
      if ( converged(element, d, n) )
      {
	end(element);
	return d.dtot;
      }

      // Check divergence
      Iteration::State newstate;
      if ( retry = diverged(element, d, n, newstate) )
      {
	changeState(newstate);
	d.reset();
	break;
      }
    
      // Stabilize iteration
      stabilize(element, d, n);

      // Write debug info
      if ( debug_iter )
	debug(element, d, n);
      
      // Update element
      update(element, d);
    }

    // End iteration
    end(element);
  }

  // Return total increment
  return d.dtot;
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
void FixedPointIteration::update(ElementGroupList& list,
				 Iteration::Increments& d)
{
  dolfin_assert(state);
  state->update(list, d);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(ElementGroup& group,
				 Iteration::Increments& d)
{
  dolfin_assert(state);
  state->update(group, d);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(Element& element,
				 Iteration::Increments& d)
{
  dolfin_assert(state);
  u.debug(element, Solution::update);
  state->update(element, d);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(ElementGroupList& list, 
				    const Iteration::Increments& d,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(list, d, n);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(ElementGroup& group, 
				    const Iteration::Increments& d,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(group, d, n);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(Element& element, 
				    const Iteration::Increments& d,
				    unsigned int n)
{
  dolfin_assert(state);
  state->stabilize(element, d, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(ElementGroupList& list,
				    const Iteration::Increments& d,
				    unsigned int n)
{
  dolfin_assert(state);
  return state->converged(list, d, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(ElementGroup& group,
				    const Iteration::Increments& d, 
				    unsigned int n)
{
  dolfin_assert(state);
  return state->converged(group, d, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(Element& element,
				    const Iteration::Increments& d,
				    unsigned int n)
{ 
  dolfin_assert(state);
  return state->converged(element, d, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::diverged(ElementGroupList& list,
				   const Iteration::Increments& d,
				   unsigned int n, Iteration::State& newstate)
{
  dolfin_assert(state);
  return state->diverged(list, d, n, newstate);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::diverged(ElementGroup& group,
				   const Iteration::Increments& d,
				   unsigned int n, Iteration::State& newstate)
{
  dolfin_assert(state);
  return state->diverged(group, d, n, newstate);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::diverged(Element& element,
				   const Iteration::Increments& d,
				   unsigned int n, Iteration::State& newstate)
{ 
  dolfin_assert(state);
  return state->diverged(element, d, n, newstate);
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
void FixedPointIteration::debug(ElementGroupList& list,
				const Iteration::Increments& d,
				unsigned int n)
{
  // Don't write debug info if iteration wasn't accepted
  if ( !state->accept() )
    return;

  // Don't write debug info before the first iteration
  if ( n == 0 )
    return;

  real r = 0.0;
  real alpha = 0.0;
  unsigned int m = 0;

  // Get stabilization parameters
  dolfin_assert(state);
  state->stabilization(alpha, m);

  // Compute residual
  dolfin_assert(state);
  r = state->residual(list);

  cout << "debug1: " << r << " " << d.d2 << " " << alpha << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::debug(ElementGroup& group,
				const Iteration::Increments& d,
				unsigned int n)
{
  // Don't write debug info if iteration wasn't accepted
  if ( !state->accept() )
    return;

  // Don't write debug info before the first iteration
  if ( n == 0 )
    return;

  real r = 0.0;
  real alpha = 0.0;
  unsigned int m = 0;

  // Get stabilization parameters
  dolfin_assert(state);
  state->stabilization(alpha, m);

  // Compute residual
  dolfin_assert(state);
  r = state->residual(group);

  cout << "  debug2: " << r << " " << d.d2 << " " << alpha << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::debug(Element& element,
				const Iteration::Increments& d,
				unsigned int n)
{
  // Don't write debug info if iteration wasn't accepted
  if ( !state->accept() )
    return;

  // Don't write debug info before the first iteration
  if ( n == 0 )
    return;

  real r = 0.0;
  real alpha = 0.0;
  unsigned int m = 0;

  // Get stabilization parameters
  dolfin_assert(state);
  state->stabilization(alpha, m);

  // Compute residual
  dolfin_assert(state);
  r = state->residual(element);

  cout << "    debug3: " << r << " " << d.d2 << " " << alpha << endl;
}
//-----------------------------------------------------------------------------
