// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/FixedPointIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f) : u(u), f(f),
  event_diag_damping("System is diagonally stiff, trying diagonal damping.", 5),
  event_accelerating("Slow convergence, need to accelerate convergence.", 5),
  event_scalar_damping("System is stiff, damping is needed.", 5),
  event_reset_element("Element iterations diverged, resetting element.", 5),
  event_reset_timeslab("Iterations diverged, resetting time slab.", 5),
  event_nonconverging("Iterations did not converge, decreasing time step", 5)
{
  maxiter       = dolfin_get("maximum iterations");
  local_maxiter = dolfin_get("maximum local iterations");
  maxdiv        = dolfin_get("maximum divergence");
  maxconv       = dolfin_get("maximum convergence");

  // FIXME: Convergence should be determined by the error control
  tol = 1e-10;

  // Assume that problem is non-stiff
  state = nonstiff;

  substate = undamped;
  m = 0;
  alpha = 1.0;
}
//-----------------------------------------------------------------------------
FixedPointIteration::~FixedPointIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(TimeSlab& timeslab)
{
  // Time slab residuals
  Residuals r;

  dolfin_start("--- Starting time slab iteration ---");

  // Fixed point iteration on the time slab  
  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Check convergence
    if ( converged(timeslab, r, n) )
    {
      dolfin_end("--- Time slab iteration converged ---");
      return true;
    }

    cout << "Time slab residual = " << r.r2 << endl;
  
    // Check stabilization
    if ( n >= 2 )
      stabilize(timeslab, r);
    
    // Update time slab
    update(timeslab);
   }

  dolfin_end("Time slab iteration did not converge ---");

  return false;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(NewArray<Element*>& elements)
{
  // Element list residuals
  Residuals r;

  // Update initial data
  init(elements);

  dolfin_start("--- Starting element list iteration ---");
  
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
      stabilize(elements, r);

    // Update element list
    update(elements);
  }

  dolfin_end("Element list iteration did not converge ---");

  return false;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(Element& element)
{
  // Element residuals
  Residuals r;

  dolfin_start("--- Starting element iteration ---");

  // Fixed point iteration on the element
  for (unsigned int n = 0; n < local_maxiter; n++)
  {
    // Check convergence
    if ( converged(element, r, n) )
    {
      dolfin_end("--- Element iteration converged ---");
      return true;
    }

    cout << "Element residual = " << r.r2 << endl;

    // Check stabilization
    if ( n >= 2 )
      stabilize(element, r);

    // Update element
    update(element);
  }

  dolfin_end("--- Element iteration did not converge ---");

  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(TimeSlab& timeslab)
{
  return timeslab.computeMaxRd(*this);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(NewArray<Element*> elements)
{
  real rmax = 0.0;
  
  // Compute maximum discrete residual
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Compute discrete residual
    rmax = std::max(rmax, residual(*element));
  }

  return rmax;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::residual(Element& element)
{
  return fabs(element.computeDiscreteResidual(f));
}
//-----------------------------------------------------------------------------
void FixedPointIteration::init(NewArray<Element*>& elements)
{
  // Update initial data for elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Update initial data
    init(*element);
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset(NewArray<Element*>& elements)
{
  // Reset elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Reset element
    reset(*element);
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::report() const
{
  switch ( state ) {
  case nonstiff:
    cout << "System is non-stiff, solution computed with "
	 << "simple fixed point iteration." << endl;
    break;
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
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(TimeSlab& timeslab)
{
  // Update time slab. Note that since the element lists are stored
  // recursively within the time slab, we need help from the time slab
  // to call iterate() for each element list.

  timeslab.update(*this);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(NewArray<Element*>& elements)
{
  // Update list of elements

  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Iterate element
    iterate(*element);
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(Element& element)
{
  // Write a debug message
  u.debug(element, Solution::update);

  switch ( state ) {
  case nonstiff:
  
    updateUndamped(element);
    break;

  case diagonal:

    updateLocalDamping(element);
    break;

  case parabolic:
    
    switch ( substate ) {
    case undamped:
      updateUndamped(element);
      break;
    case damped:
      updateGlobalDamping(element);
      break;
    case increasing:
      updateGlobalDamping(element);
      break;
    default:
      dolfin_error("Unknown damping");
    }

    break;

  case nonnormal:

    updateLocalDamping(element);
    break;

  default:
    dolfin_error("Unknown state.");
  }
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(TimeSlab& timeslab, Residuals& r,
				    unsigned int n)
{
  // Compute maximum discrete residual
  r.r1 = r.r2;
  r.r2 = residual(timeslab);

  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(NewArray<Element*>& elements,
				    Residuals& r, unsigned int n)
{
  // Convergence is handled locally when we have only one element
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
bool FixedPointIteration::converged(Element& element, Residuals& r, 
				    unsigned int n)
{
  // Compute discrete residual
  r.r1 = r.r2;
  r.r2 = residual(element);

  // Save initial discrete residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab, const Residuals& r)
{
  switch ( state ) {
  case nonstiff:
    stabilizeNonStiff(timeslab, r);
    break;
  case diagonal:
    stabilizeDiagonal(timeslab, r);
    break;
  case parabolic:
    switch ( substate ) {
    case undamped:
      stabilizeParabolicUndamped(timeslab, r);
      break;
    case damped:
      stabilizeParabolicDamped(timeslab, r);
      break;
    case increasing:
      stabilizeParabolicIncreasing(timeslab, r);
      break;
    default:
      dolfin_error("Unknown damping");
    }
    break;
  case nonnormal:
    stabilizeNonNormal(timeslab, r);
    break;
  default:
    dolfin_error("Unknown state");
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(NewArray<Element*>& elements, 
				    const Residuals& r)
{
  dolfin_error("Not implemented");
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(Element& element, const Residuals& r)
{
  dolfin_error("Not implemented");
}
//-----------------------------------------------------------------------------
void FixedPointIteration::updateUndamped(Element& element)
{
  cout << "Simple update of element" << endl;

  // Update initial value for element
  real u0 = u(element.index(), element.starttime());
  element.update(u0);

  // Update element
  element.update(f);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::updateLocalDamping(Element& element)
{
  // Compute local damping
  real dfdu = f.dfdu(element.index(), element.index(), element.endtime());
  real rho = - element.timestep() * dfdu;
  real alpha = computeDamping(rho);
  
  // Update element
  element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::updateGlobalDamping(Element& element)
{
  cout << "Globally damped update of element" << endl;
  cout << "alpha = " << alpha << endl;

  // Update initial value for element
  real u0 = u(element.index(), element.endtime());
  element.update(u0);

  // Update element
  element.update(f, alpha);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeNonStiff(TimeSlab& timeslab,
					    const Residuals& r)
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
					    const Residuals& r)
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
						     const Residuals& r)
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
						   const Residuals& r)
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
						       const Residuals& r)
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
					     const Residuals& r)
{
  cout << "Not implemented" << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::init(Element& element)
{
  // Get initial value
  real u0 = u(element.index(), element.starttime());
  
  // Reset element
  element.update(u0);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset(Element& element)
{
  // Get initial value
  real u0 = u(element.index(), element.starttime());
  
  // Reset element
  element.reset(u0);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::computeConvergenceRate(const Residuals& r)
{
  return r.r2 / (DOLFIN_EPS + r.r1);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::computeDamping(real rho)
{
  if ( rho >= 0.0 || rho < -1.0 )
    return (1.0 + DOLFIN_SQRT_EPS) / (1.0 + rho);

  return 1.0;
}
//-----------------------------------------------------------------------------
unsigned int FixedPointIteration::computeDampingSteps(real rho)
{
  dolfin_assert(rho >= 0.0);
  return 1 + 2*ceil_int(log(1.0 + rho));
}
//-----------------------------------------------------------------------------
