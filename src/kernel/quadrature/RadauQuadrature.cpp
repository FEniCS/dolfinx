// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/RadauQuadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void Radau::ComputePoints()
{
  // This function computes the Radau quadrature points in [-1,1]
  // as -1 and the zeros of
  //
  //      ( Pn-1(x) + Pn(x) ) / (1+x)
  //
  // where Pn is the n:th Legendre polynomial
  
  Legendre p;
  int n;
  double x, dx;
  double dStep;
  double dSign;
  
  // Fix the first nodal point
  dPoints[0][0]  = -1.0;
  
  // Compute the nodal points for every order
  for (int i=1;i<iMaximumNumberOfPoints;i++){
    
    // The number of nodal points
    n = i + 1;
    
    // Set size of stepping for seeking starting points
    dStep = 2.0 / ( double(n-1) * 10.0 );
    
    // Set the first nodal point which is -1
    dPoints[i][0]   = -1.0;
    
    // Start at -1 + epsilon
    x = -1.0 + dStep;
    
    // Set the sign at -1 + epsilon
    dSign = ( (p.Value(n-1,x) + p.Value(n,x)) > 0 ? 1.0 : -1.0 );
    
    // Compute the rest of the Nodes by Newton's method
    for (int j=1;j<n;j++){
      
      // Step to a sign change
      while ( (p.Value(n-1,x) + p.Value(n,x))*dSign > 0.0 )
	x += dStep;
      
      // Newton's method
      do {
	dx = - ( ( p.Value(n-1,x) + p.Value(n,x) ) /
		 ( p.D(n-1,x)     + p.D(n,x) ) );
	x  = x + dx;
      } while ( fabs(dx) > DEFAULT_NODE_TOL );
      
      // Set the node value
      dPoints[i][j] = x;
      
      // Fix the step so that it is not too large
      if ( dStep > (dPoints[i][j]-dPoints[i][j-1]) )
	dStep = dPoints[i][j] - dPoints[i][j-1];
      
      // Step forward
      dSign  = - dSign;
      x     += dStep;
      
    }
    
  }
  
}
//-----------------------------------------------------------------------------
bool Radau::CheckNumbers()
{
  // This functions checks that the numbers are ok.
  //
  // Radau quadrature with n points should be exact for polynomials of
  // order 2(n-1) = 2n - 2
  //
  //   n = 2    exact for p <= 2
  //   n = 3    exact for p <= 4 ...
  
  double dIntegral;
  Legendre p;
  int n;
  
  for (int i=1;i<iMaximumNumberOfPoints;i++){
    
    n = i + 1;
    
    // Check the quadrature for the Legendre polynomial of order 2n-3
    
    dIntegral = 0.0;
    
    for (int j=0;j<n;j++)
      dIntegral += dWeights[i][j] * p.Value(2*n-2,dPoints[i][j]);
    
    if ( fabs(dIntegral) > DEFAULT_PRECALC_CHECK_TOL )
      return false;
    
  }
  
  return true;
}
//-----------------------------------------------------------------------------
