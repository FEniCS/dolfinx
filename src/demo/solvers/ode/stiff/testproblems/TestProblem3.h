// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class TestProblem3 : public ODE
{
public:
  
  TestProblem3() : ODE(2), A(2,2)
  {
    dolfin_info("A non-normal test problem, critically damped oscillation");
    dolfin_info("with eigenvalues l1 = l2 = 100.");

    T = 1.0;

    A(0,0) = 0.0;    
    A(0,1) = -1.0;
    A(1,0) = 1e4;
    A(1,1) = 200;
  }

  real u0(unsigned int i)
  {
    return 1.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    return -A.mult(u, i);
  }

private:
  
  Matrix A;

};
