// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-02-03
// Last changed: 2005
//
// This example demonstrates the homotopy for finding
// all solutions of a system of polynomial equations
// for a couple of simple test systems taken from
// Alexander P. Morgan, ACM TOMS 1983.

#include <dolfin.h>

using namespace dolfin;

class Quadratic : public Homotopy
{
public:

  Quadratic() : Homotopy(1) {}

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

class Cubic : public Homotopy
{
public:

  Cubic() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    y[0] = 4.0*z[0]*z[0]*z[0] - 3.0*z[0] - z[1];
    y[1] = z[0]*z[0] - z[1];
  }
  
  void JF(const complex z[], const complex x[], complex y[])
  {
    y[0] = 12.0*z[0]*z[0]*x[0] - 3.0*x[0] - x[1];
    y[1] = 2.0*z[0]*x[0] - x[1];
  }

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 3;
    else
      return 2;
  }

};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("tolerance", 0.05);
  dolfin_set("homotopy monitoring", true);
  dolfin_set("homotopy randomize", false);

  Quadratic quadratic;
  quadratic.solve();

  //Cubic cubic;
  //cubic.solve();
  
  return 0;
}
