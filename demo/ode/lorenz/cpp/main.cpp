// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-02
// Last changed: 2007-05-02

#include <stdio.h>
#include <dolfin.h>

using namespace dolfin;

class Lorenz : public ODE
{
public:

  Lorenz() : ODE(3, 50)
  {
    // Parameters
    s = 10.0;
    b = 8.0 / 3.0;
    r = 28.0;
  }
  
  void u0(uBlasVector& u)
  {
    u[0] = 1.0;
    u[1] = 0.0;
    u[2] = 0.0;
  }

  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    y[0] = s*(u[1] - u[0]);
    y[1] = r*u[0] - u[1] - u[0]*u[2];
    y[2] = u[0]*u[1] - b*u[2];
  }

  void J(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t)
  {
    y[0] = s*(x[1] - x[0]);
    y[1] = (r - u[2])*x[0] - x[1] - u[0]*x[2];
    y[2] = u[1]*x[0] + u[0]*x[1] - b*x[2];
  }

  void JT(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t) {
    y[0] = -x[0]*s + (r-u[2])*x[1] + u[1]*x[2];
    y[1] = s*x[0] - x[1] + u[0]*x[2];
    y[2] = -u[0]*x[1] - b*x[2];
  }

 
private:

  // Parameters
  real s;
  real b;
  real r;

};

int main()
{
  dolfin_set("ODE number of samples", 500);
  dolfin_set("ODE initial time step", 0.01);
  dolfin_set("ODE fixed time step", true);
  dolfin_set("ODE nonlinear solver", "newton");
  dolfin_set("ODE method", "cg");
  dolfin_set("ODE order", 5);
  dolfin_set("ODE discrete tolerance", 1e-12);
  dolfin_set("ODE save solution", true);
 
  Lorenz lorenz;
  lorenz.solve();
  
  return 0;
}
