// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Legendre.h>
#include <dolfin/RadauQuadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RadauQuadrature::RadauQuadrature(int n) : GaussianQuadrature(n)
{
  init();

  if ( !check(2*n-2) )
    dolfin_error("Radau quadrature not ok, check failed.");

  dolfin_info("Radau quadrature computed for n = %d, check passed.", n);
}
//-----------------------------------------------------------------------------
void RadauQuadrature::computePoints()
{
  // Compute the Radau quadrature points in [-1,1] as -1 and the zeros
  // of ( Pn-1(x) + Pn(x) ) / (1+x) where Pn is the n:th Legendre
  // polynomial. Computation is a little different than for Gauss and
  // Lobatto quadrature, since we don't know of any good initial
  // approximation for the Newton iterations.
  
  // Special case n = 1
  if ( n == 1 ) {
    points[0] = -1.0;
    return;
  }

  Legendre p1(n-1), p2(n);
  real x, dx, step, sign;
  
  // Set size of stepping for seeking starting points
  step = 2.0 / ( real(n-1) * 10.0 );
  
  // Set the first nodal point which is -1
  points[0] = -1.0;
  
  // Start at -1 + step
  x = -1.0 + step;
  
  // Set the sign at -1 + epsilon
  sign = ( (p1(x) + p2(x)) > 0 ? 1.0 : -1.0 );
  
  // Compute the rest of the nodes by Newton's method
  for (int i = 1; i < n; i++) {
    
    // Step to a sign change
    while ( (p1(x) + p2(x))*sign > 0.0 )
      x += step;
    
    // Newton's method
    do {
      dx = - (p1(x) + p2(x)) / (p1.dx(x) + p2.dx(x));
      x  = x + dx;
    } while ( fabs(dx) > DOLFIN_EPS );
    
    // Set the node value
    points[i] = x;
    
    // Fix step so that it's not too large
    if ( step > (points[i] - points[i-1])/10.0 )
      step = (points[i] - points[i-1]) / 10.0;
    
    // Step forward
    sign = - sign;
    x += step;
    
  }
  
}
//-----------------------------------------------------------------------------
// Output
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<<(LogStream& stream, const RadauQuadrature& radau)
{
  stream << "Radau quadrature points and weights on [-1,1] for n = " << radau.size() << ":" << dolfin::endl;
  stream << dolfin::endl;

  char number[32];
  char point[32];
  char weight[32];

  stream << " i    points                   weights" << dolfin::endl;
  stream << "-----------------------------------------------------" << dolfin::endl;
  for (int i = 0; i < radau.size(); i++) {

    sprintf(number, "%2d", i);
    sprintf(point,  "% .16e", radau.point(i).x);
    sprintf(weight, "% .16e", radau.weight(i));
    
    stream << number << "   " << point << "  " << weight;

    if ( i < (radau.size()-1) )
      stream << dolfin::endl;

  }

  return stream;
}
//-----------------------------------------------------------------------------
