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
  n = 0;
  maxiter = dolfin_get("maximum iterations");

  state = undamped;

  alpha = 1.0;
  m = 0;

  d0 = 0.0;
  d1 = 0.0;

  r0 = 0.0;
  r1 = 0.0;
}
//-----------------------------------------------------------------------------
FixedPointIteration::~FixedPointIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FixedPointIteration::iterate(TimeSlab& timeslab)
{
  // Since elements are stored recursively in the time slabs, the fixed
  // point iteration needs help from the time slabs to do the iteration.
  // The time slabs will call FixedPointIteration::update() on all elements
  // within the time slabs.

  n = 0;
  state = undamped;

  cout << "-------------------------------------------------------" << endl;

  while ( !converged(timeslab) )
  {
    // Update time slab
    update(timeslab);

    // Check convergence
    if ( n > 0 )
      stabilize(timeslab);

    // Check if we have done too many iterations
    if ( n++ >= maxiter )
    {
      dolfin_info("Solution did not converge.");
      dolfin_error1("Reached maximum number of iterations (%d).", maxiter);
    }

    cout << endl;
  }

  cout << "Converged in " << n << " iterations" << endl;
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
  r0 = r1;
  r1 = timeslab.computeMaxRd(u, f);

  cout << "--- Checking convergence: " << r0 << " --> " << r1 << endl;

  return r1 < 1e-3;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::update(TimeSlab& timeslab)
{
  // Update time slab
  d0 = d1;
  d1 = timeslab.update(*this);
  
  cout << "--- Updating time slab: " << d0 << " --> " << d1 << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilize(TimeSlab& timeslab)
{
  // Compute convergence rate
  dolfin_assert(d0 > DOLFIN_EPS); 
  real rho = d1 / d0;

  switch ( state ) {
  case undamped:
    stabilizeUndamped(timeslab, rho);
    break;
  case scalar_damping:
    stabilizeScalar(timeslab, rho);
    break;
  case diagonal_damping:
    stabilizeDiagonal(timeslab, rho);
    break;
  default:
    dolfin_error("Unknown state.");
  }
 
  cout << "Damping with alpha = " << alpha << endl;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeUndamped(TimeSlab& timeslab, real rho)
{
  // Don't stabilize if solution is converging
  if ( r1 < r0 || d1 < d0 )
    return;

  // Compute stabilization
  real c = 0.99;  
  alpha = c / (1 + rho);    
  
  // Compute number of iterations with small alpha
  m = 2*ceil_int(log(rho));
  
  // Reset time slab to initial values
  timeslab.reset(u);

  // Change state
  state = scalar_damping;
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeScalar(TimeSlab& timeslab, real rho)
{
  // Decrease the remaining number of iterations with small alpha
  if ( m > 0 )
    m--;

  cout << "m = " << m << endl;
  
  if ( r1 < r0 )
  {
    if ( m == 0 )
    {
      // Start increasing alpha after the first m iterations
      alpha *= 2.0;

      // If we reached alpha = 1, return to undamped state
      if ( alpha >= 1.0 )
      {
	alpha = 1.0;
	state = undamped;
      }
    }
  }
  else
  {
    dolfin_debug("hej");

    // Compensate for old alpha
    rho /= alpha;

    // Compute stabilization
    real c = 0.99;  
    alpha = c / (1.0 + rho);    
    
    // Compute number of iterations with small alpha
    m = 2*ceil_int(log(rho));

    // Reset time slab to initial values
    timeslab.reset(u);
  }
}
//-----------------------------------------------------------------------------
void FixedPointIteration::stabilizeDiagonal(TimeSlab& timeslab, real rho)
{
  dolfin_error("Diagonal damping not implemented.");
}
//-----------------------------------------------------------------------------
