// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Legendre.h>
#include <dolfin/LobattoQuadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LobattoQuadrature::LobattoQuadrature(unsigned int n) : GaussianQuadrature(n)
{
  if ( n < 2 )
    dolfin_error("Lobatto quadrature requires at least 2 points.");

  init();

  if ( !check(2*n-3) )
    dolfin_error("Lobatto quadrature not ok, check failed.");

  //dolfin_info("Lobatto quadrature computed for n = %d, check passed.", n);
}
//-----------------------------------------------------------------------------
void LobattoQuadrature::show() const
{
  cout << "Lobatto quadrature points and weights on [-1,1] for n = " 
       << n << ":" << endl;

  cout << " i    points                   weights" << endl;
  cout << "-----------------------------------------------------" << endl;

  for (unsigned int i = 0; i < n; i++)
    dolfin_info("%2d   % .16e   %.16e", i, points[i].x, weights[i]);
}
//----------------------------------------------------------------------------
void LobattoQuadrature::computePoints()
{
  // Compute the Lobatto quadrature points in [-1,1] as the enpoints
  // and the zeroes of the derivatives of the Legendre polynomials
  // using Newton's method
  
  // Special case n = 1 (should not be used)
  if ( n == 1 ) {
    points[0] = 0.0;
    return;
  }

  // Special case n = 2
  if ( n == 2 ) {
    points[0] = -1.0;
    points[1] = 1.0;
    return;
  }

  Legendre p(n-1);
  real x, dx;

  // Set the first and last nodal points which are 0 and 1
  points[0] = -1.0;
  points[n-1] = 1.0;
  
  // Compute the rest of the nodes by Newton's method
  for (unsigned int i = 1; i <= ((n-1)/2); i++) {
    
    // Initial guess
    x = cos(DOLFIN_PI*real(i)/real(n-1));
    
    // Newton's method
    do {
      dx = - p.ddx(x) / p.d2dx(x);
      x  = x + dx;
    } while ( fabs(dx) > DOLFIN_EPS );
    
    // Save the value using the symmetry of the points
    points[i] = - x;
    points[n-1-i] = x;
    
  }
  
  // Fix the middle node
  if ( (n % 2) != 0 )
    points[n/2] = 0.0;
}
//----------------------------------------------------------------------------
