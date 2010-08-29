// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet, 2008.
//
// First added:  2003-07-02
// Last changed: 2009-09-08

#include <stdio.h>
#include <dolfin.h>
#include <dolfin/common/real.h>

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

  void u0(RealArray& u)
  {
    u[0] = 1.0;
    u[1] = 0.0;
    u[2] = 0.0;
  }

  void f(const RealArray& u, real t, RealArray& y)
  {
    y[0] = s*(u[1] - u[0]);
    y[1] = r*u[0] - u[1] - u[0]*u[2];
    y[2] = u[0]*u[1] - b*u[2];
  }

  void J(const RealArray& x, RealArray& y, const RealArray& u, real t)
  {
    y[0] = s*(x[1] - x[0]);
    y[1] = (r - u[2])*x[0] - x[1] - u[0]*x[2];
    y[2] = u[1]*x[0] + u[0]*x[1] - b*x[2];
  }

  void JT(const RealArray& x, RealArray& y, const RealArray& u, real t)
  {
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
  Lorenz lorenz;

  lorenz.parameters["number_of_samples"] = 500;
  lorenz.parameters["initial_time_step"] = 0.01;
  lorenz.parameters["fixed_time_step"] = true;
  lorenz.parameters["method"] = "cg";
  lorenz.parameters["order"] = 5;
  lorenz.parameters["discrete_tolerance"] = 1e-10;
  lorenz.parameters["save_solution"] = true;

  ODESolution u;

  lorenz.solve(u);
  lorenz.analyze_stability_computation(u);
  

  return 0;
}
