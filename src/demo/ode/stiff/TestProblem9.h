// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-07-05

#include <dolfin.h>

using namespace dolfin;

class TestProblem9 : public ODE
{
public:
  
  TestProblem9() : ODE(3, 30.0)
  {
    dolfin_info("A mixed stiff/nonstiff test problem.");

    lambda = 1000.0;
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    if ( i == 1 )
      return 1.0;

    return 1.0;
  }
  
  void f(const DenseVector& u, real t, DenseVector& y)
  {
    y(0) = u(1);
    y(1) = -(1.0 - u(2))*u(0);
    y(2) = -lambda * (u(0)*u(0) + u(1)*u(1)) * u(2);
  }

private:

  real lambda;

};
