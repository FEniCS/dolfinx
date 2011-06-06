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
#include <dolfin/la/uBLASVector.h>
#include <dolfin/la/uBLASDenseMatrix.h>
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
  std::vector<double> A_double(n*n);

  uBLASVector b(n);
  ublas_vector& _b = b.vec();
  std::vector<double> b_double(n);

  // Compute the matrix coefficients
  for (unsigned int i = 0; i < n; i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      A_double[i + n*j] = Legendre::eval(i, points[j]);
      _A(i, j) = A_double[i + n*j];
      _b[i] = 0.0;
      b_double[i] = 0.0;
    }
  }
  _b[0] = 2.0;
  b_double[0] = 2.0;

  // Solve the system of equations
  uBLASVector x(n);
  A.solve(x, b);

  ublas_vector& _x = x.vec();

  // Save the weights
  for (uint i = 0; i < n; i++)
    weights[i] = _x[i];
}
//-----------------------------------------------------------------------------
bool GaussianQuadrature::check(unsigned int q) const
{
  // Checks that the points and weights are correct. We compute the
  // value of the integral of the Legendre polynomial of degree q.
  // This value should be zero for q > 0 and 2 for q = 0

  double sum = 0.0;
  for (unsigned int i = 0; i < points.size(); i++)
    sum += weights[i]*Legendre::eval(q, points[i]);

  bool _check = false;
  if (q == 0 && std::abs(sum - 2.0) < 100.0*DOLFIN_EPS)
    _check = true;
  else if (std::abs(sum - 2.0) < 100.0*DOLFIN_EPS)
    _check = true;

  info("Quadrature check failed: r = %.2e.", std::abs(sum));

  return _check;
}
//-----------------------------------------------------------------------------
