// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class TestProblem4 : public ODE
{
public:
  
  TestProblem4() : ODE(8)
  {
    dolfin_info("The HIRES problem.");

    // Final time
    T = 321.8122;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    switch (i) {
    case 0:
      return 1.0;
    case 1:
      return 0.0;
    case 2:
      return 0.0;
    case 3:
      return 0.0;
    case 4:
      return 0.0;
    case 5:
      return 0.0;
    case 6:
      return 0.0;
    default:
      return 0.0057;
    }
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    switch (i) {
    case 0:
      return -1.71*u(0) + 0.43*u(1) + 8.32*u(2) + 0.0007;
    case 1:
      return 1.71*u(0) - 8.75*u(1);
    case 2:
      return -10.03*u(2) + 0.43*u(3) + 0.035*u(4);
    case 3:
      return 8.32*u(1) + 1.71*u(2) - 1.12*u(3);
    case 4:
      return -1.745*u(4) + 0.43*u(5) + 0.43*u(6);
    case 5:
      return -280.0*u(5)*u(7) + 0.69*u(3) + 1.71*u(4) - 0.43*u(5) + 0.69*u(6);
    case 6:
      return 280.0*u(5)*u(7) - 1.81*u(6);
    default:
      return -280.0*u(5)*u(7) + 1.81*u(6);
    }
  }

};
