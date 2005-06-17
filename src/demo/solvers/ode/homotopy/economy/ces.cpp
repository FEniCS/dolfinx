// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "CES.h"

using namespace dolfin;

void test1()
{
  // Seed random number generator so we get the same system every time
  seed(0);

  unsigned int m = 5;
  unsigned int n = 5;
  
  PolynomialIntegerCES ec(m, n);

  ec.alpha = 5;

  for (unsigned int i = 0; i < m; i++)
    ec.beta[i] = 1;

  ec.disp();
  ec.solve();
}

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("tolerance", 0.01);
  dolfin_set("discrete tolerance", 1e-10);
  dolfin_set("initial time step", 0.0001);
  dolfin_set("linear solver", "direct");
  dolfin_set("adaptive samples", false);
  dolfin_set("homotopy monitoring", false);
  dolfin_set("homotopy divergence tolerance", 10.0);
  dolfin_set("homotopy randomize", false);

  test1();

  return 0;
}
