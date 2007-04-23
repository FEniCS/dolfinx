// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem3 : public ODE
{
public:
  
  TestProblem3() : ODE(2, 1.0), A(2,2)
  {
    dolfin_info("A non-normal test problem, critically damped oscillation");
    dolfin_info("with eigenvalues l1 = l2 = 100.");

    A(0,0) = 0.0;    
    A(0,1) = 1.0;
    A(1,0) = -1e4;
    A(1,1) = -200.0;
  }

  void u0(uBlasVector& u)
  {
    u(0) = 1.0;
    u(1) = 1.0;
  }

  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    A.mult(u, y);
  }

private:
  
  DenseMatrix A;

};
