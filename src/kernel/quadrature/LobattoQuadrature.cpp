// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Quadrature.h>

//----------------------------------------------------------------------------
void Lobatto::ComputePoints()
{
  // This function computes the Lobatto quadrature points in [-1,1]
  // as the enpoints and the zeroes of the derivatives of the Legendre
  // polynomials using Newton's method.

  Legendre p;
  int n;
  double x, dx;

  // Fix the first nodal point
  dPoints[0][0]  = 1.0;
  
  // Compute the nodal points for every order
  for (int i=1;i<iMaximumNumberOfPoints;i++){

	 // The number of nodal points
	 n = i + 1;
	 
	 // Set the first and last nodal points which are 0 and 1
	 dPoints[i][0]   = -1.0;
	 dPoints[i][n-1] = 1.0;
	 
	 // Compute the rest of the Nodes by Newton's method
	 for (int j=1;j<=((n-1)/2);j++){

		// Initial guess
		x = cos(PI*double(j)/double(n-1));

		// Newton's method
		do {
		  dx = - p.D(n-1,x) / p.D2(n-1,x);
		  x  = x + dx;
		} while ( fabs(dx) > DEFAULT_NODE_TOL );
		  
		// Save the value using the symmetry of the points
		dPoints[i][j]     = - x;
		dPoints[i][n-1-j] = x;

	 }

	 // Fix the middle node
	 if ( (n % 2) != 0 )
		dPoints[i][n/2] = 0.0;
	 
  }
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

  double dIntegral;
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
