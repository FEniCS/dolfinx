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

  // FIXME: Convergence should be determined by the error control
  tol = 1e-10;

  clear();
}
//-----------------------------------------------------------------------------
FixedPointIteration::~FixedPointIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::iterate(TimeSlab& timeslab)
{
  // Since only the time slab knows how its elements are stored, the fixed
  // point iteration needs help from the time slab to do the iteration.
  // The time slab will call FixedPointIteration::update() for each element
  // within the time slab.

  reset();

  while ( !converged(timeslab) )
  {
    // Check stabilization
    if ( n >= 2 )
      stabilize(timeslab);

    // Update time slab
    update(timeslab);

    // Check if we have done too many iterations
    if ( ++n >= maxiter )
    {
      event_nonconverging();
      reset();
      return false;
    }    
  }

  reset();
  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::update(Element& element)
{
  // Save end value for element
  real u1 = element.endval();

  // Get element data
  unsigned int index = element.index();
  real t0 = element.starttime();
  real t1 = element.endtime();

  // Update initial value for element
  real u0 = u(index, t0);
  element.update(u0);

  // Local variables
  real local_alpha = 1.0;
  real local_r0 = 0.0;
  real local_r1 = 0.0;
  real local_r2 = 0.0;

  // Start with damping if we have used it before
  bool local_damping = event_diag_damping.count() > 0;

  // Fixed point iteration on the element
  for (unsigned int i = 0; i < local_maxiter; i++)
  {
    // Compute local damping
    if ( local_damping )
    {
      real dfdu = f.dfdu(index, index, t1);
      real rho = element.timestep() * fabs(dfdu);
      local_alpha = computeDamping(rho);
    }
    
    // Update element
    u.debug(element, Solution::update);
    if ( alpha == 1.0 && local_alpha == 1.0 )
      element.update(f);
    else
      element.update(f, alpha*local_alpha);

    // Compute discrete residual
    local_r1 = local_r2;
    local_r2 = fabs(element.computeDiscreteResidual(f));
    if ( i == 0 )
      local_r0 = local_r2;

    // Check convergence
    if ( local_r2 < tol )
      break;

    if ( i > 0 )
    {
      // Check if local damping is needed
      if ( local_r2 > local_r1 )
      {
	event_diag_damping();
	local_damping = true;

	// Check if we need to reset the element
	if ( local_r2 > maxdiv * local_r0 )
	{
	  event_reset_element();
	  reset(element);
	}
      }
    }
  }
  
  // Return change in end value
  return fabs(element.endval() - u1);
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
void FixedPointIteration::update(TimeSlab& timeslab)
{
  // Update time slab
  d1 = d2;
  d2 = timeslab.update(*this);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab)
{
  switch ( state ) {
  case undamped:
    stabilizeUndamped(timeslab);
    break;
  case damped:
    stabilizeDamped(timeslab);
    break;
  case increasing:
    stabilizeIncreasing(timeslab);
    break;
  default:
    dolfin_error("Unknown state.");
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeUndamped(TimeSlab& timeslab)
{
  // Check if the solution converges
  if ( r2 < 0.5*r1 )
    return;

  // Determine type of damping
  if ( r2 < r1 && event_scalar_damping.count() == 0 )
    event_accelerating();
  else
    event_scalar_damping(); 
  
  // Compute stabilization
  real rho = computeConvergenceRate();
  alpha = computeDamping(rho);
  m = computeDampingSteps(rho);
  
  // Check if we need to reset the time slab
  if ( r2 > maxdiv * r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  // Change state
  state = damped;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeDamped(TimeSlab& timeslab)
{
  // Decrease the remaining number of iterations with small alpha
  m--;

  // Check if we're done
  if ( m == 0 )
  {
    alpha *= 2.0;
    state = increasing;
  }

  // Check if the solution diverges
  if ( r2 > r1 )
  {
    // Decrease alpha
    alpha /= 2.0;

    // Check if we need to reset the time slab
    if ( r2 > maxdiv * r0 )
    {
      event_reset_timeslab();
      timeslab.reset(*this);
    }
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeIncreasing(TimeSlab& timeslab)
{
  // Increase alpha
  alpha *= 2.0;

  // Check if the solution diverges
  if ( r2 > r1 )
  {    
    // Compute stabilization
    real rho = computeConvergenceRate();
    alpha = computeDamping(rho/alpha);
    m = computeDampingSteps(rho/alpha);
    
    // Check if we need to reset the time slab
    if ( r2 > maxdiv * r0 )
    {
      event_reset_timeslab();
      timeslab.reset(*this);
    }
    
    // Change state
    event_scalar_damping();
    state = damped;
  }

  // Check if we're done
  if ( alpha >= 1.0 )
  {
    alpha = 1.0;
    state = undamped;
  }
}
//-----------------------------------------------------------------------------
real FixedPointIteration::computeConvergenceRate()
{
  real rho = d2 / (DOLFIN_EPS + d1);
  
  if ( rho <= 1.0 )
    rho = r2 / (DOLFIN_EPS + r1);

  return rho;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::computeDamping(real rho)
{
  return (1.0 + DOLFIN_SQRT_EPS) / (1.0 + rho);
}
//-----------------------------------------------------------------------------
unsigned int FixedPointIteration::computeDampingSteps(real rho)
{
  return 1 + 2*ceil_int(log(1.0 + rho));
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(TimeSlab& timeslab)
{
  // Compute maximum discrete residual
  r1 = r2;
  r2 = timeslab.computeMaxRd(u, f);

  // Save initial discrete residual
  if ( n == 0 )
    r0 = r2;
  
  return r2 < tol;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset()
{
  state = undamped;

  n = 0;
  m = 0;

  alpha = 1.0;

  d1 = 0.0;
  d2 = 0.0;

  r0 = 0.0;
  r1 = 0.0;
  r2 = 0.0;
}
//-----------------------------------------------------------------------------
