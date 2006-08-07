// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2006-07-05

#include <dolfin.h>

using namespace dolfin;

class TestProblem2 : public ODE
{
public:
  
  TestProblem2() : ODE(2, 10.0), A(2, 2)
  {
    dolfin_info("The simple test system.");

    A(0, 0) = -100.0;
    A(1, 1) = -1000.0;
  }

  real u0(unsigned int i)
  {
    return 1.0;
  }
  
  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    A.mult(u, y);
  }

private:

  DenseMatrix A;
  
};
