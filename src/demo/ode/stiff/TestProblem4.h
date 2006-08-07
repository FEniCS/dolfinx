// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2006-07-05

#include <dolfin.h>

using namespace dolfin;

class TestProblem4 : public ODE
{
public:
  
  TestProblem4() : ODE(8, 321.8122)
  {
    dolfin_info("The HIRES problem.");
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

  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    y(0) = -1.71*u(0) + 0.43*u(1) + 8.32*u(2) + 0.0007;
    y(1) = 1.71*u(0) - 8.75*u(1);
    y(2) = -10.03*u(2) + 0.43*u(3) + 0.035*u(4);
    y(3) = 8.32*u(1) + 1.71*u(2) - 1.12*u(3);
    y(4) = -1.745*u(4) + 0.43*u(5) + 0.43*u(6);
    y(5) = -280.0*u(5)*u(7) + 0.69*u(3) + 1.71*u(4) - 0.43*u(5) + 0.69*u(6);
    y(6) = 280.0*u(5)*u(7) - 1.81*u(6);
    y(7) = -280.0*u(5)*u(7) + 1.81*u(6);
  }

};
