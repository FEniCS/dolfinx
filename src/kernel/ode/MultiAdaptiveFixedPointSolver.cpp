// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Alloc.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveFixedPointSolver.h>
#include <dolfin/NewVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::MultiAdaptiveFixedPointSolver
(MultiAdaptiveTimeSlab& timeslab) : TimeSlabSolver(timeslab), ts(timeslab), f(0)
{
  f = new real[method.qsize()];
}
//-----------------------------------------------------------------------------
MultiAdaptiveFixedPointSolver::~MultiAdaptiveFixedPointSolver()
{
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
real MultiAdaptiveFixedPointSolver::iteration()
{
  NewVector Rd(ts.ne), Rdprev(ts.ne);

  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  ts.elast = -1;

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

    // Get initial value for element
    const int ep = ts.ee[e];
    const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );

    // Evaluate right-hand side at quadrature points of element
    ts.feval(f, s, e, i, a, b, k);
    //cout << "f = "; Alloc::disp(f, method.qsize());

    // Update values on element using fixed point iteration
    const real increment = method.update(x0, f, k, ts.jx + j);
    //cout << "x = "; Alloc::disp(ts.jx + j, method.nsize());
    
    // Update maximum increment
    if ( increment > max_increment )
      max_increment = increment;
    
    Rd(e) = increment;
    Rdprev(e) = ts.er[e];

    ts.er[e] = increment;

    // Update dof
    j += method.nsize();
  }

  // Debug

//   NewVector Rho(ts.ne), krecommend(ts.ne), comp(ts.ne);
//   real rho;

//   rho = Rd.norm() / Rdprev.norm();

//   cout << "rho: " << rho << endl;

//   // Compute Rho
//   for (uint e = 0; e < ts.ne; e++)
//   {
//     Rho(e) = fabs(Rd(e)) / Rdprev.norm(NewVector::linf);
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

//   cout << "comp: " << endl;
//   comp.disp();

//   cout << "Rho: " << endl;
//   Rho.disp();

//   cout << "krecommend: " << endl;
//   krecommend.disp();


  return max_increment;
}
//-----------------------------------------------------------------------------
