// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-27
// Last changed: 2006-08-08

#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterSystem.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveFixedPointSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::MultiAdaptiveFixedPointSolver
(MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), f(0),
    num_elements(0), num_elements_mono(0), 
    maxiter_local(get("ODE maximum local iterations")),
    diagonal_newton_damping(get("ODE diagonal newton damping")), dfdu(0)
{
  // Initialize local array for quadrature
  f = new real[method.qsize()];
  for (unsigned int i = 0; i < method.qsize(); i++)
    f[i] = 0.0;
  
  // Initialize diagonal of Jacobian df/du for diagonal Newton damping
  if ( diagonal_newton_damping )
  {
    dfdu = new real[ts.N];
    for (uint i = 0; i < ts.N; i++)
      dfdu[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::~MultiAdaptiveFixedPointSolver()
{
  // Compute multi-adaptive efficiency index
  if ( num_elements > 0 )
  {
    const real alpha = num_elements_mono / static_cast<real>(num_elements);
    dolfin_info("Multi-adaptive efficiency index: %.3f", alpha);
  }

  // Delete local array
  if ( f ) delete [] f;
  
  // Delete diagonal of Jacobian
  if ( dfdu ) delete [] dfdu;
}
//-----------------------------------------------------------------------------
bool MultiAdaptiveFixedPointSolver::retry()
{
  // If we're already using damping, then we don't know what to do
  if ( diagonal_newton_damping )
    return false;

  // Otherwise, use damping
  dolfin_assert(dfdu == 0);
  diagonal_newton_damping = true;
  dfdu = new real[ts.N];
  for (uint i = 0; i < ts.N; i++)
    dfdu[i] = 0.0;

  // Reset system
  ts.reset();

  dolfin_info("Direct fixed-point iteration does not converge.");
  dolfin_info("Trying diagonally damped fixed-point iteration.");
  return true;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveFixedPointSolver::start()
{
  // Update diagonal of Jacobian if used
  if ( diagonal_newton_damping )
  {
    for (uint i = 0; i < ts.N; i++)
      dfdu[i] = ts.ode.dfdu(ts.u0, ts._a, i, i);
  }
}
//-----------------------------------------------------------------------------
void MultiAdaptiveFixedPointSolver::end()
{
  // Count the number of elements
  num_elements += ts.ne;
  num_elements_mono += ts.length() / ts.kmin * static_cast<real>(ts.ode.size());
}
//-----------------------------------------------------------------------------
real MultiAdaptiveFixedPointSolver::iteration(real tol, uint iter,
					      real d0, real d1)
{
  // Reset dof
  uint j = 0;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Reset maximum increment
  real increment_max = 0.0;

  // Keep track of the number of local iterations
  real num_local = 0.0;

  // Iterate over all sub slabs
  uint e0 = 0;
  uint e1 = 0;
  for (uint s = 0; s < ts.ns; s++)
  {
    // Cover all elements in current sub slab
    e1 = ts.coverSlab(s, e0);
    
    // Get data for sub slab
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;

    // Save current dof
    uint j0 = j;

    // Iterate on each sub slab
    real increment_local = 0.0;
    for (uint iter_local = 0; iter_local < maxiter_local; iter_local++)
    {
      // Reset current dof
      j = j0;

      // Iterate over all elements in current sub slab
      for (uint e = e0; e < e1; e++)
      {
	// Get element data
	const uint i = ts.ei[e];
	
	// Save old end-point value
	const real x1 = ts.jx[j + method.nsize() - 1];
	
	// Get initial value for element
	const int ep = ts.ee[e];
	const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0(i) );
	
	// Evaluate right-hand side at quadrature points of element
	if ( method.type() == Method::cG )
	  ts.cGfeval(f, s, e, i, a, b, k);
	else
	  ts.dGfeval(f, s, e, i, a, b, k);
	//cout << "f = "; Alloc::disp(f, method.qsize());
	
	// Update values on element using fixed-point iteration
	if ( diagonal_newton_damping )
	{
	  // FIXME: Parameter 0.5 most suited for cG(1)
	  const real alpha = 1.0 / (1.0 - 0.5*k*dfdu[i]);
	  method.update(x0, f, k, ts.jx + j, alpha);
	}
	else
	{
	  method.update(x0, f, k, ts.jx + j);
	}
	//cout << "x = "; Alloc::disp(ts.jx + j, method.nsize());

	// Compute increment
	const real increment = std::abs(ts.jx[j + method.nsize() - 1] - x1);
	
	// Update sub slab increment
	increment_local = std::max(increment_local, increment);
	
	// Update maximum increment
	increment_max = std::max(increment_max, increment);

	// Update dof
	j += method.nsize();
      }

      // Update counter of local iterations
      num_local += static_cast<real>(e1 - e0) / static_cast<real>(ts.ne);
      
      // Check if we are done
      if ( increment_local < tol )
      {
	
	break;
      }
    }

    // Step to next sub slab
    e0 = e1;
  }
  
  // Add to common counter of local iterations
  num_local_iterations += static_cast<uint>(num_local + 0.5);

  return increment_max;
}
//-----------------------------------------------------------------------------
dolfin::uint MultiAdaptiveFixedPointSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
