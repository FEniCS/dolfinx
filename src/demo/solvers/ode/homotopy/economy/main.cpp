// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "CES.h"
#include "Leontief.h"
#include "Polemarchakis.h"

using namespace dolfin;

// CES test problem from Eaves and Schmedders (rational form)
void test1()
{
  CES ec(2, 2, 0.5);

  ec.a[0][0] = 4.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 4.0;
  
  ec.w[0][0] = 10.0; ec.w[0][1] =  1.0;
  ec.w[1][0] =  1.0; ec.w[1][1] = 12.0;
  
  ec.b[0] = 0.2;
  ec.b[1] = 0.2;    

  ec.disp();
  ec.solve();
}

// CES test problem from Eaves and Schmedders (polynomial form)
void test2()
{
  PolynomialCES ec(2, 2, 0.5);

  ec.a[0][0] = 4.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 4.0;
  
  ec.w[0][0] = 10.0; ec.w[0][1] =  1.0;
  ec.w[1][0] =  1.0; ec.w[1][1] = 12.0;
  
  ec.b[0] = 0.2;
  ec.b[1] = 0.2;    

  ec.disp();
  ec.solve();
}

// Leontief test problem (rational form)
void test3()
{
  Leontief ec(2, 2);

  ec.a[0][0] = 2.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 2.0;
  
  ec.w[0][0] = 1.0; ec.w[0][1] = 2.0;
  ec.w[1][0] = 2.0; ec.w[1][1] = 1.0;

  ec.disp();
  ec.solve();
}

// Leontief test problem (polynomial form)
void test4()
{
  PolynomialLeontief ec(2, 2);

  ec.a[0][0] = 2.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 2.0;
  
  ec.w[0][0] = 1.0; ec.w[0][1] = 2.0;
  ec.w[1][0] = 2.0; ec.w[1][1] = 1.0;

  ec.disp();
  ec.solve();
}

// Leontief test problem from Polemarchakis (rational form)
void test5()
{
  Polemarchakis ec;
  ec.solve();
}

// Leontief test problem from Polemarchakis (polynomial form)
void test6()
{
  PolynomialPolemarchakis ec;
  ec.solve();
}

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("tolerance", 0.01);
  dolfin_set("initial time step", 0.01);
  dolfin_set("linear solver", "direct");
  dolfin_set("adaptive samples", true);
  dolfin_set("homotopy monitoring", false);
  dolfin_set("homotopy divergence tolerance", 10.0);
  dolfin_set("homotopy randomize", true);

  test1();

  return 0;
}
