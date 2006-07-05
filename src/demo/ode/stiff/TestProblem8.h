// Copyright (C) 2003-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2003-2006.

#include <dolfin.h>

using namespace dolfin;

class TestProblem8 : public ODE
{
public:
  
  TestProblem8() : ODE(3, 0.3)
  {
    dolfin_info("System of fast and slow chemical reactions, taken from the book by");
    dolfin_info("Hairer and Wanner, page 3.");
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 1.0;
    else
      return 0.0;
  }
  
  void f(const DenseVector& u, real t, DenseVector& y)
  {
    y(0) = -0.04 * u(0) + 1.0e4 * u(1) * u(2);
    y(1) = 0.04 * u(0) - 1.0e4 * u(1) * u(2) - 3.0e7 * u(1) * u(1);
    y(2) = 3.0e7 * u(1) * u(1);
  }
  
};
