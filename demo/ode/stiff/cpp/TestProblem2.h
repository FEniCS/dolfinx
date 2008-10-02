// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem2 : public ODE
{
public:
  
  TestProblem2() : ODE(2, 10.0), A(2, 2)
  {
    message("The simple test system.");

    A.mat()(0, 0) = -100.0;
    A.mat()(0, 0) = -100.0;
    A.mat()(1, 1) = -1000.0;
  }

  void u0(uBLASVector& u)
  {
    u[0] = 1.0;
    u[1] = 1.0;
  }
  
  void f(const uBLASVector& u, double t, uBLASVector& y)
  {
    A.mult(u, y);
  }

private:

  uBLASDenseMatrix A;
  
};
