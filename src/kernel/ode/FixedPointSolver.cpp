// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Alloc.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/NewMethod.h>
#include <dolfin/FixedPointSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FixedPointSolver::FixedPointSolver(NewTimeSlab& timeslab, const NewMethod& method)
  : TimeSlabSolver(timeslab, method), f(0)
{
  f = new real[method.qsize()];
}
//-----------------------------------------------------------------------------
FixedPointSolver::~FixedPointSolver()
{
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
real FixedPointSolver::iteration()
{
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
    
    // Update dof
    j += method.nsize();
  }

  return max_increment;
}
//-----------------------------------------------------------------------------
