// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveNewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::MonoAdaptiveNewtonSolver
(MonoAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), A(timeslab)
{

}
//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::~MonoAdaptiveNewtonSolver()
{

}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::start()
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
  A.update(ts);

  //debug();
  //A.disp();
}
//-----------------------------------------------------------------------------
real MonoAdaptiveNewtonSolver::iteration()
{
  // Evaluate b = -F(x) at current x
  beval();
  
  // Solve linear system F for dx
  solver.solve(A, dx, b);
   
  // Get array containing the increments (assumes uniprocessor case)
  real* dxvals = dx.array();

  /*  

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

  */

  // Restore array
  dx.restore(dxvals);

  return 0.0;

  //return max_increment;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::beval()
{
  // Get array of values for b (assumes uniprocessor case)
  real* bvals = b.array();


  // Restore array
  b.restore(bvals);
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::debug()
{
  /*
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
  */
}
//-----------------------------------------------------------------------------
