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
FixedPointIteration::FixedPointIteration(Solution&u, RHS& f) : u(u), f(f)
{
  maxiter = dolfin_get("maximum iterations");
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

  clear();

  cout << "-------------------------------------------------------" << endl;

  while ( !converged(timeslab) )
  {
    // Check convergence
    if ( n >= 2 )
      stabilize(timeslab);

    // Update time slab
    update(timeslab);

    // Check if we have done too many iterations
    if ( n++ >= maxiter )
      return false;
    
    cout << endl;

  }

  cout << "Converged in " << n << " iterations" << endl;

  clear();

  return true;
}
//-----------------------------------------------------------------------------
real FixedPointIteration::update(Element& element)
{
  // Get initial value for element
  real u0 = u(element.index(), element.starttime());
  
  // Update value
  element.update(u0);    
  
  real d = 0.0;
  if ( state == undamped )
    d = element.update(f);
  else
    d = element.update(f, alpha);
  
  // Write debug info
  u.debug(element, Solution::update);

  return fabs(d);
}
//-----------------------------------------------------------------------------
bool FixedPointIteration::converged(TimeSlab& timeslab)
{
  // FIXME: Convergence should be determined by the error control

  // Compute maximum discrete residual
  r1 = r2;
  r2 = timeslab.computeMaxRd(u, f);

  // Save initial discrete residual
  if ( n == 0 )
    r0 = r2;

  cout << "Checking convergence: " << r1 << " --> " << r2 << endl;

  return r2 < 1e-3;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(TimeSlab& timeslab)
{
  // Update time slab
  d1 = d2;
  d2 = timeslab.update(*this);
  
  cout << "Updating time slab: " << d1 << " --> " << d2 << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab)
{
  cout << "State = " << state << endl;

  switch ( state ) {
  case undamped:
    stabilizeUndamped(timeslab);
    break;
  case scalar_small:
    stabilizeScalarSmall(timeslab);
    break;
  case scalar_increasing:
    stabilizeScalarIncreasing(timeslab);
    break;
  case diagonal_small:
    stabilizeDiagonalSmall(timeslab);
    break;
  case diagonal_increasing:
    stabilizeDiagonalIncreasing(timeslab);
    break;
  default:
    dolfin_error("Unknown state.");
  }
 
  cout << "  Damping with alpha = " << alpha << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeUndamped(TimeSlab& timeslab)
{
  // Check if the solution converges
  if ( r2 < 0.5*r0 )
    return;

  cout << "  Need to stabilize: " << r1 << " --> " << r2 << endl;   
  
  // Compute stabilization
  real rho = computeConvergenceRate();
  alpha = computeDamping(rho);
  m = computeDampingSteps(rho);
  
  // Reset time slab to initial values 
  if ( r2 > 10.0 * r0 )
    timeslab.reset(u);
  
  // Change state
  state = scalar_small;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeScalarSmall(TimeSlab& timeslab)
{
  // Decrease the remaining number of iterations with small alpha
  m--;

  cout << "  m = " << m << endl;

  // Check if we're done
  if ( m == 0 )
  {
    alpha *= 2.0;
    state = scalar_increasing;
  }

  // Adjust alpha if the solution diverges
  if ( r2 > r1 )
  {
    // Decrease alpha
    alpha /= 2.0;

    // Reset time slab to initial values 
    if ( r2 > 10.0 * r0 )
      timeslab.reset(u);
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeScalarIncreasing(TimeSlab& timeslab)
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
    if ( r2 > 10.0 * r0 )
      timeslab.reset(u);
    
    // Change state
    state = scalar_small;
  }

  // Check if we're done
  if ( alpha >= 1.0 )
  {
    alpha = 1.0;
    state = undamped;
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeDiagonalSmall(TimeSlab& timeslab)
{
  dolfin_error("Diagonal damping not implemented.");
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeDiagonalIncreasing(TimeSlab& timeslab)
{
  dolfin_error("Diagonal damping not implemented.");
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
void FixedPointIteration::clear()
{
  n = 0;
  state = undamped;

  alpha = 1.0;
  m = 0;

  d1 = 0.0;
  d2 = 0.0;

  r0 = 0.0;
  r1 = 0.0;
  r2 = 0.0;
}
//-----------------------------------------------------------------------------
