// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
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
				    num_elements(0), num_elements_mono(0)
{
  f = new real[method.qsize()];
  for (unsigned int i = 0; i < method.qsize(); i++)
    f[i] = 0.0;
}
//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::~MultiAdaptiveFixedPointSolver()
{
  // Compute multi-adaptive efficiency index
  const real alpha = num_elements_mono / static_cast<real>(num_elements);
  dolfin_info("Multi-adaptive efficiency index: %.3f.", alpha);

  // Delete local array
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveFixedPointSolver::end()
{
  num_elements += ts.ne;
  num_elements_mono += ts.length() / ts.kmin * static_cast<real>(ts.ode.size());
}
//-----------------------------------------------------------------------------
real MultiAdaptiveFixedPointSolver::iteration()
{
  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Reset maximum increment
  real max_increment = 0.0;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Cover all elements in current sub slab
    s = ts.cover(s, e);

    // Get element data
    const uint i = ts.ei[e];
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;

    // Save old end-point value
    const real x1 = ts.jx[j + method.nsize() - 1];

    // Get initial value for element
    const int ep = ts.ee[e];
    const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );

    //for (uint iter = 0; iter < 2; iter++)
    // {
    
    // Evaluate right-hand side at quadrature points of element
    if ( method.type() == Method::cG )
      ts.cGfeval(f, s, e, i, a, b, k);
    else
      ts.dGfeval(f, s, e, i, a, b, k);
    //cout << "f = "; Alloc::disp(f, method.qsize());
    
    // Update values on element using fixed point iteration
    method.update(x0, f, k, ts.jx + j);
    //cout << "x = "; Alloc::disp(ts.jx + j, method.nsize());

    //}

    // Compute increment
    const real increment = fabs(ts.jx[j + method.nsize() - 1] - x1);

    // Update maximum increment
    if ( increment > max_increment )
    {
      cout << "  i = " << i << ": " << increment << endl;
      max_increment = increment;
    }

    // Update dof
    j += method.nsize();
  }

  return max_increment;
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
