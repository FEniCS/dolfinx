// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem7 : public ODE
{
public:
  
  TestProblem7() : ODE(101, 1.0)
  {
    h = 1.0 / (static_cast<real>(N) - 1);
    message("The heat equation on [0,1] with h = %f", h);
  }
  
  void u0(uBLASVector& u)
  {
    u.zero();
  }

  void f(const uBLASVector& u, real t, uBLASVector& y)
  {
    // Boundary values
    y[0]   = 0.0;
    y[N-1] = 0.0;

    // Interior values
    for (unsigned int i = 1; i < N - 1; i++)
    {
      // Heat source
      real source = 0.0;
      if ( i == N/2 )
	source = 100.0;
      
      y[i] = (u[i-1] - 2.0*u[i] + u[i+1]) / (h*h) + source;
    }
  }
  
private:
  
  real h;

};
