// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class TestProblem7 : public ODE
{
public:
  
  TestProblem7() : ODE(101)
  {
    // Mesh size
    h = 1.0 / (static_cast<real>(N) - 1);

    // Final time
    T = 1.0;

    dolfin_info("The heat equation on [0,1] with h = %f", h);
    
    // Compute sparsity
    sparse();
  }
  
  real u0(unsigned int i)
  {
    return 0.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    // Boundary values
    if ( i == 0 || i == (N-1) )
      return 0.0;
    
    // Heat source
    real source = 0.0;
    if ( i == N/2 )
      source = 100.0;

    return (u(i-1) - 2.0*u(i) + u(i+1)) / (h*h) + source;
  }
  
private:
  
  real h;

};
