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
#include <dolfin/DiagonalIteration.h>
#include <dolfin/FixedPointIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f) : u(u), f(f)
{
  maxiter       = dolfin_get("maximum iterations");
  local_maxiter = dolfin_get("maximum local iterations");
  maxdiv        = dolfin_get("maximum divergence");
  maxconv       = dolfin_get("maximum convergence");

  // FIXME: Convergence should be determined by the error control
  tol = 1e-10;

  // Assume that problem is non-stiff
  state = new NonStiffIteration(u, f, *this, maxdiv, maxconv, tol);
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
  Iteration::Damping d;

  dolfin_start("Starting time slab iteration");

  // Fixed point iteration on the time slab  
  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Check convergence
    if ( converged(timeslab, r, n) )
    {
      dolfin_end("Time slab iteration converged");
      return true;
    }

    cout << "Time slab residual = " << r.r2 << endl;
  
    // Check stabilization
    if ( n >= 2 )
      stabilize(timeslab, r, d);
    
    // Update time slab
    update(timeslab);
  }


  dolfin_end("Time slab iteration did not converge");

  return false;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(NewArray<Element*>& elements)
{
  Iteration::Residuals r;
  Iteration::Damping d;
  
  // Update initial data
  init(elements);
  
  dolfin_start("Starting element list iteration");

  // Fixed point iteration on the element list
  for (unsigned int n = 0; n < local_maxiter; n++)
  {
    // Check convergence
    if ( converged(elements, r, n) )
    {
      dolfin_end("Element list iteration converged");
      return true;
    }
    
    cout << "Element list residual = " << r.r2 << endl;

    // Check stabilization
    if ( n >= 2 )
      stabilize(elements, r, d);

    // Update element list
    update(elements);
  }
  
  dolfin_end("Element list iteration did not converge");

  return false;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(Element& element)
{
  Iteration::Residuals r;
  Iteration::Damping d;

  dolfin_start("Starting element iteration");

  // Fixed point iteration on the element
  for (unsigned int n = 0; n < local_maxiter; n++)
  {
    // Check convergence
    if ( converged(element, r, n) )
    {
      dolfin_end("Element iteration converged");
      return true;
    }

    cout << "Element residual = " << r.r2 << endl;

    // Check stabilization
    if ( n >= 2 )
      stabilize(element, r, d);

    // Update element
    update(element);
  }

  dolfin_end("Element iteration did not converge");

  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(TimeSlab& timeslab)
{
  dolfin_assert(state);
  return state->residual(timeslab);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(NewArray<Element*> elements)
{
  dolfin_assert(state);
  return state->residual(elements);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(Element& element)
{
  dolfin_assert(state);
  return state->residual(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::init(NewArray<Element*>& elements)
{
  dolfin_assert(state);
  state->init(elements);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::init(Element& element)
{
  dolfin_assert(state);
  state->init(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset(NewArray<Element*>& elements)
{
  dolfin_assert(state);
  state->reset(elements);
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

  /*
  switch ( state ) {
  case diagonal:
    cout << "System appears to be diagonally stiff, solution computed with "
	 << "diagonally damped fixed point iteration." << endl;
    break;
  case parabolic:
    cout << "System appears to be parabolically stiff, solution computed with "
	 << "adaptively damped fixed point iteration." << endl;
    break;
  case nonnormal:
    cout << "System is stiff, solution computed with "
	 << "adaptively stabilizing time step sequence." << endl;
    break;
  default:
    dolfin_error("Unknown state");
  }
  */
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(TimeSlab& timeslab)
{
  dolfin_assert(state);
  state->update(timeslab);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(NewArray<Element*>& elements)
{
  dolfin_assert(state);
  state->update(elements);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(Element& element)
{
  u.debug(element, Solution::update);
  dolfin_assert(state);
  state->update(element);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab, 
				    const Iteration::Residuals& r,
				    Iteration::Damping& d)
{
  dolfin_assert(state);
  Iteration::State newstate = state->stabilize(timeslab, r, d);
  changeState(newstate);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(NewArray<Element*>& elements, 
				    const Iteration::Residuals& r,
				    Iteration::Damping& d)
{
  dolfin_assert(state);
  Iteration::State newstate = state->stabilize(elements, r, d);
  changeState(newstate);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(Element& element, 
				    const Iteration::Residuals& r,
				    Iteration::Damping& d)
{
  dolfin_assert(state);
  Iteration::State newstate = state->stabilize(element, r, d);
  changeState(newstate);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(TimeSlab& timeslab, Iteration::Residuals& r,
				    unsigned int n)
{
  dolfin_assert(state);
  return state->converged(timeslab, r, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(NewArray<Element*>& elements,
				    Iteration::Residuals& r, unsigned int n)
{
  dolfin_assert(state);
  return state->converged(elements, r, n);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(Element& element, Iteration::Residuals& r, 
				    unsigned int n)
{ 
  dolfin_assert(state);
  return state->converged(element, r, n);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::changeState(Iteration::State newstate)
{
  dolfin_assert(state);

  // Don't change if it hasn't changed
  if ( newstate == state->state() )
    return;

  dolfin_info("New state = %d", newstate);

  // Delete old state
  delete state;
  state = 0;

  // Initialize new state
  switch ( newstate ) {
  case Iteration::nonstiff:
    state = new NonStiffIteration(u, f, *this, maxdiv, maxconv, tol);
    break;
  case Iteration::diagonal:
    state = new DiagonalIteration(u, f, *this, maxdiv, maxconv, tol);
    break;
  case Iteration::parabolic:
    dolfin_error("Not implemented");
    break;
  case Iteration::nonnormal:
    dolfin_error("Not implemented");
    break;
  default:
    dolfin_error("Unknown state");
  }
}
//-----------------------------------------------------------------------------

/*
void FixedPointIteration::stabilizeNonStiff(TimeSlab& timeslab,
					    const Iteration::Residuals& r)
{
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return;

  // Notify change of strategy
  dolfin_info("Problem appears to be stiff, trying diagonal damping.");
  
  // Check if we need to reset the time slab
  if ( r.r2 > r.r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  // Change state
  state = diagonal;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeDiagonal(TimeSlab& timeslab,
					    const Iteration::Residuals& r)
{
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return;

  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, trying parabolic damping.");
  
  // Check if we need to reset the time slab
  if ( r.r2 > r.r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  // Change state
  state = parabolic;

  // Start without damping to find the correct value of the damping
  substate = undamped;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeParabolicUndamped(TimeSlab& timeslab,
						     const Iteration::Residuals& r)
{
  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return;
  
  // Compute stabilization
  real rho = computeConvergenceRate(r);
  alpha = computeDamping(rho);
  m = computeDampingSteps(rho);
  
  // Check if we need to reset the time slab
  if ( r.r2 > maxdiv * r.r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  cout << "Computing alpha = " << alpha << endl;
  cout << "Computing m = " << m << endl;

  // Use globally damped iterations
  substate = damped;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeParabolicDamped(TimeSlab& timeslab,
						   const Iteration::Residuals& r)
{
  // Decrease the remaining number of iterations with small alpha
  m--;

  cout << "Remaining number of small steps = " << m << endl;

  // Check if the solution converges
  if ( r.r2 < r.r1 && m > 0 )
    return;
  
  // Check if we're done
  if ( m == 0 )
  {
    alpha *= 2.0;
    substate = increasing;
  }
  
  // Check if the solution diverges
  if ( r.r2 > r.r1 )
  {
    // Decrease alpha
    alpha /= 2.0;
    
    // Check if we need to reset the time slab
    if ( r.r2 > maxdiv * r.r0 )
    {
      event_reset_timeslab();
      timeslab.reset(*this);
    }
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeParabolicIncreasing(TimeSlab& timeslab,
						       const Iteration::Residuals& r)
{
  // Increase alpha
  alpha *= 2.0;

  cout << "Increasing alpha to " << alpha << endl;

  // Check if the solution diverges
  if ( r.r2 > r.r1 )
  {    
    // Compute stabilization
    real rho = computeConvergenceRate(r);
    alpha = computeDamping(rho/alpha);
    m = computeDampingSteps(rho/alpha);
    
    // Check if we need to reset the time slab
    if ( r.r2 > maxdiv * r.r0 )
    {
      event_reset_timeslab();
      timeslab.reset(*this);
    }
    
    // Change state
    event_scalar_damping();
    substate = damped;
  }

  cout << "Adjusting alpha to " << alpha << endl;

  // Check if we're done
  if ( alpha >= 1.0 )
  {
    alpha = 1.0;
    substate = undamped;
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeNonNormal(TimeSlab& timeslab,
					     const Iteration::Residuals& r)
{
  cout << "Not implemented" << endl;
}

*/
