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

  cout << "-----------------------------------------------" << endl;
  dolfin_start("Starting fixed point iteration");

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
      cout << "Did not converge" << endl;

      event_nonconverging();
      reset();
      return false;
    }    

    cout << endl;
  }

  dolfin_end("Converged after %d iterations", n);
  cout << "-----------------------------------------------" << endl;

  reset();
  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::update(Element& element)
{
  switch ( state ) {
  case nonstiff:
   
    return updateUndamped(element);
    break;

  case diagonal:

    return updateLocalDamping(element);
    break;

  case parabolic:
    
    switch ( damping ) {
    case undamped:
      return updateUndamped(element);
      break;
    case damped:
      return updateGlobalDamping(element);
      break;
    case increasing:
      return updateGlobalDamping(element);
      break;
    default:
      dolfin_error("Unknown damping");
    }

    break;

  case nonnormal:

    return updateLocalDamping(element);
    break;

  default:
    dolfin_error("Unknown state.");
  }

  return 0.0;
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
  // Update time slab
  d1 = d2;
  d2 = timeslab.update(*this);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::updateUndamped(Element& element)
{
  cout << "Simple update of element" << endl;

  // Save end value for element
  real u1 = element.endval();

  // Update initial value for element
  real u0 = u(element.index(), element.starttime());
  element.update(u0);

  // Update element
  u.debug(element, Solution::update);
  element.update(f);
    
  // Return change in end value
  return fabs(element.endval() - u1);
}
//-----------------------------------------------------------------------------
real FixedPointIteration::updateLocalDamping(Element& element)
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

  // Fixed point iteration on the element
  for (unsigned int i = 0; i < local_maxiter; i++)
  {
    // Compute local damping
    real dfdu = f.dfdu(index, index, t1);
    real rho = - element.timestep() * dfdu;
    local_alpha = computeDamping(rho);

    // Update element
    u.debug(element, Solution::update);
    element.update(f, local_alpha);
    
    // Compute discrete residual
    local_r1 = local_r2;
    local_r2 = fabs(element.computeDiscreteResidual(f));
    if ( i == 0 )
      local_r0 = local_r2;

    // Check convergence
    if ( local_r2 < tol )
      break;

    // Check if iterations diverge
    if ( i > 2 && local_r2 > (maxdiv * local_r0) )
      dolfin_error("Local iterations did not converge");
  }
  
  // Return change in end value
  return fabs(element.endval() - u1); 
}
//-----------------------------------------------------------------------------
real FixedPointIteration::updateGlobalDamping(Element& element)
{
  cout << "Globally damped update of element" << endl;
  cout << "alpha = " << alpha << endl;

  // Save end value for element
  real u1 = element.endval();

  // Update initial value for element
  real u0 = u(element.index(), element.endtime());
  element.update(u0);

  // Update element
  u.debug(element, Solution::update);
  element.update(f, alpha);
    
  // Return change in end value
  return fabs(element.endval() - u1);
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab)
{
  switch ( state ) {
  case nonstiff:
    stabilizeNonStiff(timeslab);
    break;
  case diagonal:
    stabilizeDiagonal(timeslab);
    break;
  case parabolic:
    switch ( damping ) {
    case undamped:
      stabilizeParabolicUndamped(timeslab);
      break;
    case damped:
      stabilizeParabolicDamped(timeslab);
      break;
    case increasing:
      stabilizeParabolicIncreasing(timeslab);
      break;
    default:
      dolfin_error("Unknown damping");
    }
    break;
  case nonnormal:
    stabilizeNonNormal(timeslab);
    break;
  default:
    dolfin_error("Unknown state");
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeNonStiff(TimeSlab& timeslab)
{
  // Check if the solution converges
  if ( r2 < maxconv*r1 )
    return;

  // Notify change of strategy
  dolfin_info("Problem appears to be stiff, trying diagonal damping.");
  
  // Check if we need to reset the time slab
  if ( r2 > r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  // Reset the number of iterations
  n = 0;
  
  // Change state
  state = diagonal;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeDiagonal(TimeSlab& timeslab)
{
  // Check if the solution converges
  if ( r2 < maxconv*r1 )
    return;

  // Notify change of strategy
  dolfin_info("Diagonal damping is not enough, trying parabolic damping.");
  
  // Check if we need to reset the time slab
  if ( r2 > r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  // Reset the number of iterations
  n = 0;

  // Change state
  state = parabolic;

  // Start without damping to find the correct value of the damping
  damping = undamped;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeParabolicUndamped(TimeSlab& timeslab)
{
  // Check if the solution converges
  if ( r2 < maxconv*r1 )
    return;
  
  // Compute stabilization
  real rho = computeConvergenceRate();
  alpha = computeDamping(rho);
  m = computeDampingSteps(rho);
  
  // Check if we need to reset the time slab
  if ( r2 > maxdiv*r0 )
  {
    event_reset_timeslab();
    timeslab.reset(*this);  
  }

  cout << "Computing alpha = " << alpha << endl;
  cout << "Computing m = " << m << endl;

  // Use globally damped iterations
  damping = damped;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeParabolicDamped(TimeSlab& timeslab)
{
  // Decrease the remaining number of iterations with small alpha
  m--;

  cout << "Remaining number of small steps = " << m << endl;

  // Check if the solution converges
  if ( r2 < r1 && m > 0 )
    return;
  
  // Check if we're done
  if ( m == 0 )
  {
    alpha *= 2.0;
    damping = increasing;
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
void FixedPointIteration::stabilizeParabolicIncreasing(TimeSlab& timeslab)
{
  // Increase alpha
  alpha *= 2.0;

  cout << "Increasing alpha to " << alpha << endl;

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
    damping = damped;
  }

  cout << "Adjusting alpha to " << alpha << endl;

  // Check if we're done
  if ( alpha >= 1.0 )
  {
    alpha = 1.0;
    damping = undamped;
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeNonNormal(TimeSlab& timeslab)
{
  cout << "Not implemented" << endl;
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
bool FixedPointIteration::converged(TimeSlab& timeslab)
{
  // Compute maximum discrete residual
  r1 = r2;
  r2 = timeslab.computeMaxRd(u, f);

  // Save initial discrete residual
  if ( n == 0 )
    r0 = r2;
  
  cout << "global residual = " << r2 << endl;
  if ( n > 0 )
    cout << "convergence rate = " << r2 / r1 << endl;

  return r2 < tol;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::reset()
{
  damping = undamped;

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
