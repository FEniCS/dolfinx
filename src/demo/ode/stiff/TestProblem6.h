// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2006-07-05

#include <cmath>
#include <dolfin.h>

using namespace dolfin;

class TestProblem6 : public ODE
{
public:
  
  TestProblem6() : ODE(2, 100.0)
  {
    dolfin_info("Van der Pol's equation.");

    mu = 10.0;
  }

  real u0(unsigned int i)
  {
    switch (i) {
    case 0:
      return 2.0;
    default:
      return 0.0;
    }
  }

  void f(const DenseVector& u, real t, DenseVector& y)
  {
    y(0) = u(1);
    y(1) = mu*(1.0 - u(0)*u(0))*u(1) - u(0);
  }

private:

  real mu;

};
