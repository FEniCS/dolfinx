// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2003, 2004.

#include <dolfin.h>

using namespace dolfin;

class TestProblem8 : public ODE
{
public:
  
  TestProblem8() : ODE(3)
  {
    dolfin_info("System of fast and slow chemical reactions, taken from the book by");
    dolfin_info("Hairer and Wanner, page 3.");

    // Final time
    T = 0.3;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 1.0;
    else
      return 0.0;
  }
  
  real f(const Vector& u, real t, unsigned int i)
  {
    //if ( i == 1 )
    // dolfin_info("t = %.16e u = [%.16e %.16e %.16e]", t, u(0), u(1), u(2));

    if ( i == 0 )
      return -0.04 * u(0) + 1.0e4 * u(1) * u(2);
    
    if ( i == 1 )
      return 0.04 * u(0) - 1.0e4 * u(1) * u(2) - 3.0e7 * u(1) * u(1);
    
    return 3.0e7 * u(1) * u(1);
  }
  
};
