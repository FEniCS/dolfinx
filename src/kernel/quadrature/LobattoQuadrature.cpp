// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Legendre.h>
#include <dolfin/LobattoQuadrature.h>

using namespace dolfin;

//----------------------------------------------------------------------------
void LobattoQuadrature::computePoints()
{
  // Compute the Lobatto quadrature points in [-1,1] as the enpoints
  // and the zeroes of the derivatives of the Legendre polynomials
  // using Newton's method.
  
  // Special case n = 1
  if ( n == 1 ) {
    points[0] = 0.0;
    return;
  }

  Legendre p(n-1);
  real x, dx;

  // Set the first and last nodal points which are 0 and 1
  points[0] = -1.0;
  points[n-1] = 1.0;
  
  // Compute the rest of the nodes by Newton's method
  for (int i = 1; j <= ((n-1)/2); i++) {
    
    // Initial guess
    x = cos(DOLFIN_PI*real(i)/real(n-1));
    
    // Newton's method
    do {
      dx = - p.dx(x) / p.ddx(x);
      x  = x + dx;
    } while ( fabs(dx) > DOLFIN_EPS );
    
    // Save the value using the symmetry of the points
    points[i] = - x;
    points[n-1-i] = x;
    
  }
  
  // Fix the middle node
  if ( (n % 2) != 0 )
    points[i][n/2] = 0.0;
}
//----------------------------------------------------------------------------
bool Lobatto::CheckNumbers()
{
  // This functions checks that the numbers are ok.
  //
  // Lobatto quadrature with n points should be exact for polynomials of
  // order 2(n-1)-1 = 2n - 3
  //
  //   n = 2    exact for p <= 1
  //   n = 3    exact for p <= 3 ...

  real dIntegral;
  Legendre p;
  int n;
  
  for (int i=1;i<iMaximumNumberOfPoints;i++){

	 n = i + 1;
	 
 	 // Check the quadrature for the Legendre polynomial of order 2n-3
	 
	 dIntegral = 0.0;
	 
	 for (int j=0;j<n;j++)
		dIntegral += dWeights[i][j] * p.Value(2*n-3,dPoints[i][j]);

	 if ( fabs(dIntegral) > DEFAULT_PRECALC_CHECK_TOL )
		return false;
	 
  }
  
  return true;
}
//----------------------------------------------------------------------------
