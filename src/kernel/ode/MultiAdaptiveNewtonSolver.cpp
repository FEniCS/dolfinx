// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/Matrix.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveNewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveNewtonSolver::MultiAdaptiveNewtonSolver
(MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), A(timeslab), f(0), mpc(A)
{
  // Initialize local array
  f = new real[method.qsize()];

  // Set preconditioner
  solver.setPreconditioner(mpc);
}
//-----------------------------------------------------------------------------
MultiAdaptiveNewtonSolver::~MultiAdaptiveNewtonSolver()
{
  // Compute multi-adaptive efficiency index
  const real alpha = num_elements_mono / static_cast<real>(num_elements);
  dolfin_info("Multi-adaptive efficiency index: %.3f.", alpha);
  
  // Delete local array
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::end()
{
  num_elements += ts.ne;
  num_elements_mono += ts.length() / ts.kmin * static_cast<real>(ts.ode.size());
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
real MultiAdaptiveNewtonSolver::iteration(uint iter, real tol)
{
  cout << "MultiAdaptiveNewtonSolver::iteration()" << endl;

  // Evaluate b = -F(x) at current x
  beval();
  
  // Solve linear system F for dx
  solver.solve(A, dx, b);
   
  // Get array containing the increments (assumes uniprocessor case)
  real* dxx = dx.array();

  // Update solution x -> x + dx
  for (uint j = 0; j < ts.nj; j++)
    ts.jx[j] += dxx[j];

  // Compute maximum increment
  real max_increment = 0.0;
  for (uint j = 0; j < ts.nj; j++)
  {
    const real increment = fabs(dxx[j]);
    if ( increment > max_increment )
      max_increment = increment;
  }

  // Restore array
  dx.restore(dxx);

  return max_increment;
}
//-----------------------------------------------------------------------------
dolfin::uint MultiAdaptiveNewtonSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::beval()
{
  // Get array of values for b (assumes uniprocessor case)
  real* bb = b.array();

  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Cover all elements in current sub slab
    s = ts.coverNext(s, e);

    // Get element data
    const uint i = ts.ei[e];
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;

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
    method.update(x0, f, k, bb + j);
    
    // Subtract current values
    for (uint n = 0; n < method.nsize(); n++)
      bb[j + n] -= ts.jx[j + n];

    // Update dof
    j += method.nsize();
  }

  // Restore array
  b.restore(bb);
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  Matrix B(n, n);
  Vector F1(n), F2(n);

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.jx[j];
    real dx = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * std::abs(xj));
		  
    ts.jx[j] -= 0.5*dx;
    beval();
    F1 = b; // Should be -b

    ts.jx[j] = xj + 0.5*dx;
    beval();
    F2 = b; // Should be -b

    ts.jx[j] = xj;

    for (uint i = 0; i < n; i++)
    {
      real dFdx = (F1(i) - F2(i)) / dx;
      if ( fabs(dFdx) > DOLFIN_EPS )
	B(i, j) = dFdx;
    }
  }

  B.disp();
}
//-----------------------------------------------------------------------------
