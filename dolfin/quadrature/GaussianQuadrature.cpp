// Copyright (C) 2003-2007 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-06-03
// Last changed: 2007-07-18

#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/common/real.h>
#include <dolfin/la/uBLASVector.h>
#include <dolfin/la/uBLASDenseMatrix.h>
#include <dolfin/la/HighPrecision.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/Legendre.h>
#include "GaussianQuadrature.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GaussianQuadrature::GaussianQuadrature(unsigned int n) : Quadrature(n, 2.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GaussianQuadrature::init()
{
  compute_points();
  compute_weights();
}
//-----------------------------------------------------------------------------
void GaussianQuadrature::compute_weights()
{
  // Compute the quadrature weights by solving a linear system of equations
  // for exact integration of polynomials. We compute the integrals over
  // [-1,1] of the Legendre polynomials of degree <= n - 1; These integrals
  // are all zero, except for the integral of P0 which is 2.
  //
  // This requires that the n-point quadrature rule is exact at least for
  // polynomials of degree n-1.

  const uint n = points.size();

  // Special case n = 0
  if (n == 0)
  {
    weights[0] = 2.0;
    return;
  }

  uBLASDenseMatrix A(n, n);
  ublas_dense_matrix& _A = A.mat();
  std::vector<real> A_real(n*n);

  uBLASVector b(n);
  ublas_vector& _b = b.vec();
  std::vector<real> b_real(n);

  // Compute the matrix coefficients
  for (unsigned int i = 0; i < n; i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      A_real[i + n*j] = Legendre::eval(i, points[j]);
      _A(i, j) = to_double(A_real[i + n*j]);
      _b[i] = 0.0;
      b_real[i] = 0.0;
    }
  }
  _b[0] = 2.0;
  b_real[0] = 2.0;

#ifndef HAS_GMP
  // Solve the system of equations
  // FIXME: Do we get high enough precision?
  //LU lu;
  //lu.set("LU report", false);
  //lu.solve(A, x, b);
  uBLASVector x(n);
  A.solve(x, b);

  ublas_vector& _x = x.vec();

  // Save the weights
  for (uint i = 0; i < n; i++)
    weights[i] = _x[i];

#else
  //With extended precision: Use the double precision result as initial guess for the
  //extended precision linear solver.

  uBLASDenseMatrix A_inv(A);
  A_inv.invert();

  // Solve using A_inv as preconditioner
  HighPrecision::real_solve_precond(n, &A_real[0], &weights[0], &b_real[0],
                                    A_inv, real_epsilon());

#endif
}
//-----------------------------------------------------------------------------
bool GaussianQuadrature::check(unsigned int q) const
{
  // Checks that the points and weights are correct. We compute the
  // value of the integral of the Legendre polynomial of degree q.
  // This value should be zero for q > 0 and 2 for q = 0

  real sum = 0.0;
  for (unsigned int i = 0; i < points.size(); i++)
    sum += weights[i]*Legendre::eval(q, points[i]);

  //info("Checking quadrature weights: %.2e.", fabs(sum));

  if (q == 0)
  {
    if (real_abs(sum - 2.0) < 100.0*real_epsilon())
      return true;
  }
  else
  {
    if (real_abs(sum) < 100.0*real_epsilon())
      return true;
  }

  info("Quadrature check failed: r = %.2e.", to_double(real_abs(sum)));

  return false;
}
//-----------------------------------------------------------------------------
