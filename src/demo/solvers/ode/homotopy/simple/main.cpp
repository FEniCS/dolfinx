// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// This example demonstrates the homotopy for finding all
// solutions of a system of polynomial equations for the
// simple test system F(z) = -z^2 + 1 = 0, taken from
// Alexander P. Morgan, ACM TOMS 1983.

#include <string>
#include <iostream>
#include <dolfin.h>

using namespace dolfin;

class Homotopy : public ComplexODE
{
public:
  
  Homotopy(unsigned int m) : ComplexODE(1), 
			     m(static_cast<real>(m)),
			     p(static_cast<real>(3)),
			     c(0.00143289, 0.982727)
  {
    T = 1.0;
  }
  
  complex z0(unsigned int i)
  {
    real r = std::pow(std::abs(c), 1.0/p);
    real a = std::arg(c) / p;

    complex z = std::polar(r, a + m/p*2.0*DOLFIN_PI);

    return z;
  }
  
  complex f(const complex z[], real t, unsigned int i)
  {
    return z[0]*z[0]*z[0] - c + z[0]*z[0] - 1.0;
  }

  void M(const complex x[], complex y[], const complex z[], real t)
  {
    y[0] = (3.0*(1.0 - t)*z[0]*z[0] - 2.0*t*z[0]) * x[0];
    std::cout << "Product at t = " << t << ": " << y[0] << " = M * " << x[0] << std::endl;
  }

private:

  // Which root to start at, m = 0, 1, 2
  real m;

  // Degree of homotopy
  real p;

  // Parameter for start root, x^3 = c
  complex c;

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("solve dual problem", false);
  dolfin_set("use new ode solver", true);
  dolfin_set("method", "dg");
  dolfin_set("order", 0);
  dolfin_set("implicit", true);

  dolfin_set("initial time step", 0.0001);
  dolfin_set("fixed time step", true);
  dolfin_set("tolerance", 1e-5);

  // Iterate over the different starting points
  char filename[16];
  for (unsigned int i = 0; i < 3; i++)
  {
    sprintf(filename, "primal_%d.m", i);
    dolfin_set("file name", filename);

    Homotopy homotopy(i);
    std::cout << "Solving homotopy for starting point z = " << homotopy.z0(0) << std::endl;
    homotopy.solve();
  }

  return 0;
}
