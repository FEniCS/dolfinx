// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Legendre.h>
#include <dolfin/GaussQuadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GaussQuadrature::GaussQuadrature(int n) : GaussianRules(n)
{
  computePoints();
  computeWeights();

  if ( !check() )
    dolfin_error("Gauss quadrature not ok, check failed.");
  
  dolfin_info("Gauss quadrature points (n = %d) computed, check passed.", n);
}
//-----------------------------------------------------------------------------
void GaussQuadrature::computePoints()
{
  // Compute Gauss quadrature points on [-1,1] as the
  // as the zeroes of the Legendre polynomials using Newton's method.
  
  // Special case n = 1
  if ( n == 1 ) {
    points[0] = 0.0;
    return;
  }

  Legendre p(n);
  real x, dx;
  
  // Compute the points by Newton's method
  for (int i = 0; i <= ((n-1)/2); i++) {
    
    // Initial guess
    x = cos(DOLFIN_PI*(real(i+1)-0.25)/(real(n)+0.5));
    
    // Newton's method
    do {
      dx = - p(x) / p.dx(x);
      x  = x + dx;
    } while ( fabs(dx) > DOLFIN_EPS );
    
    // Save the value using the symmetry of the points
    points[i] = - x;
    points[n-1-i] = x;
    
  }
  
  // Set middle node
  if ( (n % 2) != 0 )
    points[n/2] = 0.0;
}
//-----------------------------------------------------------------------------
bool GaussQuadrature::check()
{
  // Checks that the points and weights are correct.
  //
  // Gauss quadrature with n points should be exact for polynomials of
  // degree p = 2n - 1:
  //
  //   n = 2    exact for p <= 3
  //   n = 3    exact for p <= 5
  //   ...
  //
  // We compute the value of the integral of the Legendre polynomial of degree p.
  // This value should be zero.
  
  Legendre p(2*n - 1);
  int n;
  
  real sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += weights[i] * p(points[i]);
    
  return fabs(sum) < DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
