// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// This example demonstrates the homotopy for finding all
// solutions of a system of polynomial equations for the
// simple test system F(z) = -z^2 + 1 = 0, taken from
// Alexander P. Morgan, ACM TOMS 1983.

#include <dolfin.h>

using namespace dolfin;

class Simple : public Homotopy
{
public:

  Simple() : Homotopy(1) {}

  void F(const complex z[], complex y[])
  {
    y[0] = 1.0 - z[0]*z[0];
  }
  
  void JF(const complex z[], const complex x[], complex y[])
  {
    y[0] = - 2.0*z[0]*x[0];
  }

  unsigned int degree(unsigned int i) const
  {
    return 2;
  }

};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  dolfin_set("method", "cg");
  dolfin_set("order", 2);

  Simple simple;
  simple.solve();

  return 0;
}
