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
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f) : 
  u(u), f(f),
  message_diagonal_damping("System is diagonally stiff, trying diagonal damping.", 5),
  message_accelerating("Slow convergence, need to accelerate convergence.", 5),
  message_scalar_damping("System is stiff, damping is needed.", 5),
  message_resetting_element("Element iterations diverged, resetting element.", 5),
  message_resetting_timeslab("Iterations diverged, resetting time slab.", 5),
  message_nonconverging("Iterations did not converge, decreasing time step", 5)
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
  // Since elements are stored recursively in the time slabs, the fixed
  // point iteration needs help from the time slabs to do the iteration.
  // The time slabs will call FixedPointIteration::update() on all elements
  // within the time slabs.

  reset();

  while ( !converged(timeslab) )
  {
    // Check stabilization
    if ( n >= 2 )
      stabilize(timeslab);

    // Update time slab
    update(timeslab);

    // Check if we have done too many iterations
    if ( n++ >= maxiter )
    {
      message_nonconverging.display();
      return false;
    }    
  }

  reset();

  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::update(Element& element)
{
  // Get initial value for element
  real u0 = u(element.index(), element.starttime());
  
  // Update value
  element.update(u0);    

  // Save end value for element
  real u1 = element.endval();

  // Local iteration number
  unsigned int local_n = 0;

  // Local alpha
  real local_alpha = 1.0;
  
  // Local discrete residuals
  real local_r0 = 0.0;
  real local_r1 = 0.0;
  real local_r2 = 0.0;

  // Fixed point iteration on the element
  while ( true )
  {
    // Update element
    if ( alpha == 1.0 && local_alpha == 1.0 )
      element.update(f);
    else
      element.update(f, alpha*local_alpha);

    // Compute discrete residual
    local_r1 = local_r2;
    local_r2 = fabs(element.computeDiscreteResidual(f));

    // Save initial discrete residual
    if ( n == 0 )
      local_r0 = local_r2;
    
    // Write debug info
    u.debug(element, Solution::update);
    
    // Check stabilization
    if ( local_n >= 2 )
    {
      real rho = local_r2 / (DOLFIN_EPS + local_r1);
      if ( rho > 0.5 )
      {
	// Compute diagonal damping
	message_diagonal_damping.display();
	local_alpha = 1.0 / (1.0 + rho/local_alpha);

	if ( local_r2 > maxdiv * r0 )
	{
	  // Need to reset element
	  message_resetting_element.display();
	  reset(element);
	}
      }
    }

    // Check if we have done too many iterations
    if ( local_n++ >= local_maxiter )
      break;

    // Check convergence
    if ( local_r2 < tol )
      break;

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
  else if ( r2 < r1 )
    if ( message_scalar_damping.count() == 0 )
      message_accelerating.display();
  else
    message_scalar_damping.display();
  
  // Compute stabilization
  real rho = computeConvergenceRate();
  alpha = computeDamping(rho);
  m = computeDampingSteps(rho);
  
  // Reset time slab to initial values 
  if ( r2 > maxdiv * r0 )
  {
    message_resetting_timeslab.display();
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

  // Adjust alpha if the solution diverges
  if ( r2 > r1 )
  {
    // Decrease alpha
    alpha /= 2.0;

    // Reset time slab to initial values 
    if ( r2 > maxdiv * r0 )
      timeslab.reset(*this);
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
    m = computeDampingSteps(rho);
    
    // Reset time slab to initial values 
    if ( r2 > maxdiv * r0 )
      timeslab.reset(*this);
    
    // Change state
    message_scalar_damping.display();
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
  dolfin_assert(rho > 0.0);
  return 0.99 / (1 + rho);    
}
//-----------------------------------------------------------------------------
unsigned int FixedPointIteration::computeDampingSteps(real rho)
{
  dolfin_assert(rho > 0.0);
  return 1 + 2*ceil_int(log(1.0 + rho));
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
