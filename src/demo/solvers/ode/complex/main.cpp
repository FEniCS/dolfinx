// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// This example demonstrates the solution of a complex-valued ODE:
// 
//     z'(t) = j*z(t) on (0,10]
//     z(0)  = 1
//
// where j is the imaginary unit. The exact solution of this system
// is given by
//
//     z(t) = exp(j*t) = (cos(t), sin(t)).

#include <dolfin.h>

using namespace dolfin;

class Exponential : public ComplexODE
{
public:
  
  Exponential() : ComplexODE(1)
  {
    T = 10.0;
  }
  
  complex z0(unsigned int i)
  {
    return 1.0;
  }

  complex f(const complex z[], real t, unsigned int i)
  {
    return j*z[0];
  }

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("solve dual problem", false);
  dolfin_set("use new ode solver", true);
  dolfin_set("method", "cg");
  dolfin_set("order", 1);

  Exponential exponential;
  exponential.solve();

  return 0;
}
