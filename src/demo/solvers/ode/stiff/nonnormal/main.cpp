// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A non-normal test problem: critically damped oscillation
// with eigenvalues l1 = l2 = 100.

#include <dolfin.h>

using namespace dolfin;

class NonNormal : public ODE
{
public:
  
  NonNormal() : ODE(2), A(2,2)
  {
    T = 2.0;

    real lambda = 100.0;
    real epsilon = 1.0;

    A(0,0) = 0.0;    
    A(0,1) = -epsilon;
    A(1,0) = lambda*lambda/epsilon;
    A(1,1) = 2.0*lambda;
    
    A.show();
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

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("tolerance", 0.1);
  dolfin_set("method", "dg");
  dolfin_set("order", 0);
  dolfin_set("initial time step", 0.1);
  //dolfin_set("fixed time step", true);

  NonNormal nonnormal;
  nonnormal.solve();
  
  return 0;
}
