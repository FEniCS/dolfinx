// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveNewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveNewtonSolver::MultiAdaptiveNewtonSolver
(MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), A(timeslab, ode, method), f(0)
{
  // Initialize local array
  f = new real[method.qsize()];
}
//-----------------------------------------------------------------------------
MultiAdaptiveNewtonSolver::~MultiAdaptiveNewtonSolver()
{
  // Delete local array
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::start()
{
  // Get size of system
  int nj = static_cast<int>(ts.nj);

  // Initialize increment vector
  dx.init(nj);

  // Initialize right-hand side
  b.init(nj);

  // Initialize Jacobian matrix
  A.init(dx, dx);

  // Recompute Jacobian
  A.update();

  //debug();
  //A.disp();
}
//-----------------------------------------------------------------------------
real MultiAdaptiveNewtonSolver::iteration()
{
  // Evaluate b = -F(x) at current x
  beval();
  
  // Solve linear system F for dx
  solver.solve(A, dx, b);
   
  // Get array containing the increments (assumes uniprocessor case)
  real* dxvals = dx.array();

  /*
  real* bvals = b.array();
  for (uint j = 0; j < ts.nj; j++)
    dxvals[j] = bvals[j];
  b.restore(bvals);
  */

  // Update solution x -> x - dx
  for (uint j = 0; j < ts.nj; j++)
    ts.jx[j] += dxvals[j];

  // Compute maximum increment
  real max_increment = 0.0;
  for (uint j = 0; j < ts.nj; j++)
  {
    const real increment = fabs(dxvals[j]);
    if ( increment > max_increment )
      max_increment = increment;
  }

  // Restore array
  dx.restore(dxvals);

  return max_increment;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::beval()
{
  // Get array of values for b (assumes uniprocessor case)
  real* bvals = b.array();

  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  ts.elast = -1;

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
    method.update(x0, f, k, bvals + j);
    
    // Subtract current values
    for (uint n = 0; n < method.nsize(); n++)
      bvals[j + n] -= ts.jx[j + n];

    // Update dof
    j += method.nsize();
  }

  // Restor array
  b.restore(bvals);
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  NewMatrix B(n, n);
  NewVector F1(n), F2(n);

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.jx[j];
    real dx = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(xj));
		  
    ts.jx[j] -= 0.5*dx;
    beval();
    for (uint i = 0; i < n; i++)
      F1(i) = -b(i);

    ts.jx[j] = xj + 0.5*dx;
    beval();
    for (uint i = 0; i < n; i++)
      F2(i) = -b(i);

    ts.jx[j] = xj;

    for (uint i = 0; i < n; i++)
    {
      real dFdx = (F2(i) - F1(i)) / dx;
      if ( fabs(dFdx) > DOLFIN_EPS )
	B(i, j) = dFdx;
    }
  }

  B.disp();
}
//-----------------------------------------------------------------------------
