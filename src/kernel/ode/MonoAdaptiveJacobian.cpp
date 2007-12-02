// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2006-08-21

#include <dolfin/dolfin_math.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveJacobian.h>
#include <dolfin/timing.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveJacobian::MonoAdaptiveJacobian(MonoAdaptiveTimeSlab& timeslab,
					   bool implicit, bool piecewise)
  : TimeSlabJacobian(timeslab), ts(timeslab),
    implicit(implicit), piecewise(piecewise), xx(timeslab.N), yy(timeslab.N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MonoAdaptiveJacobian::~MonoAdaptiveJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint MonoAdaptiveJacobian::size(uint dim) const
{
  return ts.x.size();
}
//-----------------------------------------------------------------------------
void MonoAdaptiveJacobian::mult(const uBlasVector& x, uBlasVector& y) const
{
  // Start with y = x, accounting for the derivative dF_j/dx_j = 1
  if ( !implicit )
    y = x;

  // Compute size of time step
  const real a = ts.starttime();
  const real k = ts.length();

  // Compute product y = Mx for each stage for implicit system
  if ( implicit )
  {
    // Iterate over stages
    for (uint n = 0; n < method.nsize(); n++)
    {
      const uint noffset = n * ts.N;

      // Copy values to xx
      ts.copy(x, noffset, xx, 0, ts.N);

      // Do multiplication
      if ( piecewise )
      {
	ode.M(xx, yy, ts.u0, a);
      }
      else
      {
	const real t = a + method.npoint(n) * k;
	ts.copy(ts.x, noffset, ts.u, 0, ts.N);
	ode.M(xx, yy, ts.u, t);
      }
      
      // Copy values from yy
      ts.copy(yy, 0, y, noffset, ts.N);
    }
  }

  // Iterate over the stages
  for (uint n = 0; n < method.nsize(); n++)
  {
    const real t = a + method.npoint(n) * k;
    const uint noffset = n * ts.N;

    /*
    // Compute yy = df/du * x for current stage
    for (uint i = 0; i < ts.N; i++)
    {
      real sum = 0.0;
      const Array<uint>& deps = ode.dependencies[i];
      const uint Joffset = Jindices[i];
      for (uint pos = 0; pos < deps.size(); pos++)
      {
	const uint j = deps[pos];
	sum += Jvalues[Joffset + pos] * xxx[noffset + j];
      }
      yy[i] = sum;
    }
    */

    // Copy values to xx and u
    ts.copy(x, noffset, xx, 0, ts.N);
    ts.copy(ts.x, noffset, ts.u, 0, ts.N);

    // Compute z = df/du * x for current stage
    ode.J(xx, yy, ts.u, t);

    // Add z with correct weights to y
    for (uint m = 0; m < method.nsize(); m++)
    {
      const uint moffset = m * ts.N;

      // Get correct weight
      real w = 0.0;
      if ( method.type() == Method::cG )
	w = - k * method.nweight(m, n + 1);
      else
	w = - k * method.nweight(m, n);

      // Add w*yy to y
      for (uint i = 0; i < ts.N; i++)
	y(moffset + i) += w * yy(i);
    }
  }
}
//-----------------------------------------------------------------------------
void MonoAdaptiveJacobian::update(const uBlasVector& u, real t)
{
  const uint N = ode.size();

  real k = ts.endtime() - ts.starttime();

//   cout << "k: " << k << endl;


  Matrix& Atmp = ode.Jmatrix(u, t);
  Matrix& Mtmp = ode.Mmatrix(t);

  tic();
  if(As == 0)
  {
    As = new Matrix(Atmp);
  }
  message("Matrix dup took %g seconds",toc());

#ifdef HAVE_PETSC_H
  tic();
  Mat As_M = As->mat().mat();
  Mat Atmp_M = Atmp.mat().mat();
  Mat Mtmp_M = Mtmp.mat().mat();

  MatCopy(Mtmp_M, As_M, SUBSET_NONZERO_PATTERN);
  message("Matrix copy2 took %g seconds",toc());

  tic();
  if(method.type() == Method::cG)
  {
    if(method.degree() != 1)
      error("Sparse Jacobian only implemented for dG(0) and cG(1)");

    MatAXPY(As_M, -0.5 * k, Atmp_M, SAME_NONZERO_PATTERN);
  }
  else
  {
    if(method.degree() != 0)
      error("Sparse Jacobian only implemented for dG(0) and cG(1)");

    MatAXPY(As_M, -k, Atmp_M, SAME_NONZERO_PATTERN);
  }
  message("Matrix axpy took %g seconds",toc());
#else
  error("Sparse Jacobian only implemented for PETSc");
#endif
}
