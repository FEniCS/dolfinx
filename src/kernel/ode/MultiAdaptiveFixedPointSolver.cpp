// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveFixedPointSolver.h>
#include <dolfin/Vector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::MultiAdaptiveFixedPointSolver
(MultiAdaptiveTimeSlab& timeslab) : TimeSlabSolver(timeslab), ts(timeslab), f(0),
				    num_elements(0), num_elements_mono(0),
				    num_iterations_local(0.0)
{
  f = new real[method.qsize()];
  for (unsigned int i = 0; i < method.qsize(); i++)
    f[i] = 0.0;

  maxiter_local = dolfin_get("maximum local iterations");
}
//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::~MultiAdaptiveFixedPointSolver()
{
  // Compute multi-adaptive efficiency index
  const real alpha = num_elements_mono / static_cast<real>(num_elements);
  dolfin_info("Multi-adaptive efficiency index: %.3f.", alpha);

  // Compute average number of local iterations
  if ( num_timeslabs > 0 )
  {
    const real n = num_iterations_local / static_cast<real>(num_iterations);
    dolfin_info("Average number of local iterations per step:  %.2f.", n);
  }

  // Delete local array
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveFixedPointSolver::start()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiAdaptiveFixedPointSolver::end()
{
  // Count the number of elements
  num_elements += ts.ne;
  num_elements_mono += ts.length() / ts.kmin * static_cast<real>(ts.ode.size());
}
//-----------------------------------------------------------------------------
real MultiAdaptiveFixedPointSolver::iteration(uint iter, real tol)
{
  // Reset dof
  uint j = 0;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Reset maximum increment
  real increment_max = 0.0;

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
	const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );
	
	// Evaluate right-hand side at quadrature points of element
	if ( method.type() == Method::cG )
	  ts.cGfeval(f, s, e, i, a, b, k);
	else
	  ts.dGfeval(f, s, e, i, a, b, k);
	//cout << "f = "; Alloc::disp(f, method.qsize());
	
	// Update values on element using fixed point iteration
	method.update(x0, f, k, ts.jx + j);
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
      num_iterations_local += static_cast<real>(e1 - e0) / static_cast<real>(ts.ne);
      
      // Check if we are done
      if ( increment_local < tol )
	break;
    }

    // Step to next sub slab
    e0 = e1;
  }

  return increment_max;
}
//-----------------------------------------------------------------------------
dolfin::uint MultiAdaptiveFixedPointSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
// DEBUG
//
//   Vector Rd(ts.ne), Rdprev(ts.ne);
//
//   Vector Rho(ts.ne), krecommend(ts.ne), comp(ts.ne);
//   real rho;
//
//   rho = Rd.norm() / Rdprev.norm();
//
//   cout << "rho: " << rho << endl;
//
//   // Compute Rho
//   for (uint e = 0; e < ts.ne; e++)
//   {
//     Rho(e) = fabs(Rd(e)) / Rdprev.norm(Vector::linf);
//     //krecommend(e) = 1.0 / (2.0 * Rho(e));
//     if(Rho(e) < 0.01)
//     {
//       krecommend(e) = 1;
//     }
//     else if(Rho(e) >= 0.01 && Rho(e) < 0.5)
//     {
//       krecommend(e) = 0;
//     }
//     else
//     {
//       krecommend(e) = -1;
//     }
//     comp(e) = ts.ei[e];
//   }
//  
//   cout << "comp: " << endl;
//   comp.disp();
//
//   cout << "Rho: " << endl;
//   Rho.disp();
//
//   cout << "krecommend: " << endl;
//   krecommend.disp();
