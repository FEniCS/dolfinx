// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/LU.h>
#include <dolfin/Legendre.h>
#include <dolfin/GaussianQuadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GaussianQuadrature::GaussianQuadrature(unsigned int n) : Quadrature(n)
{
  // Length of interval [-1,1]
  m = 2.0;
}
//-----------------------------------------------------------------------------
void GaussianQuadrature::init()
{
  computePoints();
  computeWeights();
}
//-----------------------------------------------------------------------------
void GaussianQuadrature::computeWeights()
{
  // Compute the quadrature weights by solving a linear system of equations
  // for exact integration of polynomials. We compute the integrals over
  // [-1,1] of the Legendre polynomials of degree <= n - 1; These integrals
  // are all zero, except for the integral of P0 which is 2.
  //
  // This requires that the n-point quadrature rule is exact at least for
  // polynomials of degree n-1.

  // Special case n = 0
  if ( n == 0 ) {
    weights[0] = 2.0;
    return;
  }
 
  Matrix A(n, n);
  Vector x(n), b(n);

  // Compute the matrix coefficients
  for (unsigned int i = 0; i < n; i++) {
    Legendre p(i);
    for (unsigned int j = 0; j < n; j++)
      A(i, j) = p(points[j]);
    b(i) = 0.0;
  }
  b(0) = 2.0;
    
  // Solve the system of equations
  // FIXME: Do we get high enough precision?
  LU lu;
  lu.solve(A, x, b);

  // Save the weights
  for (unsigned int i = 0; i < n; i++)
    weights[i] = x(i);
}
//-----------------------------------------------------------------------------
bool GaussianQuadrature::check(unsigned int q) const
{
  // Checks that the points and weights are correct. We compute the
  // value of the integral of the Legendre polynomial of degree q.
  // This value should be zero for q > 0 and 2 for q = 0
  
  Legendre p(q);
  
  real sum = 0.0;
  for (unsigned int i = 0; i < n; i++)
    sum += weights[i] * p(points[i]);
  
  if ( q == 0 )
    return fabs(sum - 2.0) < DOLFIN_EPS;

  return fabs(sum) < DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
