// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-06-03
// Last changed: 2005-12-09

#include <stdio.h>
#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Legendre.h>
#include <dolfin/GaussQuadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GaussQuadrature::GaussQuadrature(unsigned int n) : GaussianQuadrature(n)
{
  init();

  if ( !check(2*n-1) )
    dolfin_error("Gauss quadrature not ok, check failed.");
  
  //dolfin_info("Gauss quadrature computed for n = %d, check passed.", n);
}
//-----------------------------------------------------------------------------
void GaussQuadrature::disp() const
{
  cout << "Gauss quadrature points and weights on [-1,1] for n = " 
       << n << ":" << endl;

  cout << " i    points                   weights" << endl;
  cout << "-----------------------------------------------------" << endl;
  
  for (unsigned int i = 0; i < n; i++)
    dolfin_info("%2d   % .16e   %.16e", i, points[i].x(), weights[i]);
}
//-----------------------------------------------------------------------------
void GaussQuadrature::computePoints()
{
  // Compute Gauss quadrature points on [-1,1] as the
  // as the zeroes of the Legendre polynomials using Newton's method
  
  // Special case n = 1
  if ( n == 1 ) {
    points[0] = 0.0;
    return;
  }

  Legendre p(n);
  real x, dx;
  
  // Compute the points by Newton's method
  for (unsigned int i = 0; i <= ((n-1)/2); i++) {
    
    // Initial guess
    x = cos(DOLFIN_PI*(real(i+1)-0.25)/(real(n)+0.5));
    
    // Newton's method
    do {
      dx = - p(x) / p.ddx(x);
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
