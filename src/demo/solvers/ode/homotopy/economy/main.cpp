// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "CES.h"
#include "Leontief.h"
#include "Polemarchakis.h"

using namespace dolfin;

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("adaptive samples", true);
  //dolfin_set("homotopy monitoring", true);
  dolfin_set("tolerance", 0.01);
  dolfin_set("initial time step", 0.01);
  dolfin_set("homotopy divergence tolerance", 10.0);
  dolfin_set("homotopy randomize", true);
  dolfin_set("linear solver", "direct");

  //Leontief leontief(2, 2, true);
  //leontief.solve();

  //CES ces(2, 2, 0.5);
  //ces.disp();
  //ces.solve();

  //Polemarchakis economy;
  PolynomialPolemarchakis economy;
  economy.solve();

  return 0;
}
