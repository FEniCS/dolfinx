// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "CES.h"
#include "Leontief.h"
#include "Polemarchakis.h"
#include "EavesSchmedders.h"

using namespace dolfin;

/// CES test problem from Eaves and Schmedders (rational form).
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test1()
{
  EavesSchmedders ec;
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (polynomial form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test2()
{
  PolynomialEavesSchmedders ec;
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (rational form).
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test3()
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

/// CES test problem from Eaves and Schmedders (polynomial form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test4()
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

/// Leontief test problem (rational form)
/// Has one unique solution: p = (0.5, 0.5)
/// (Also p = (0, 1) with first equation removed)

void test5()
{
  Leontief ec(2, 2);

  ec.a[0][0] = 2.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 2.0;
  
  ec.w[0][0] = 1.0; ec.w[0][1] = 2.0;
  ec.w[1][0] = 2.0; ec.w[1][1] = 1.0;

  ec.disp();
  ec.solve();
}

/// Leontief test problem (polynomial form)
/// Has one unique solution: p = (0.5, 0.5)
/// (Also p = (0, 1) with first equation removed)

void test6()
{
  PolynomialLeontief ec(2, 2);

  ec.a[0][0] = 2.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 2.0;
  
  ec.w[0][0] = 1.0; ec.w[0][1] = 2.0;
  ec.w[1][0] = 2.0; ec.w[1][1] = 1.0;

  ec.disp();
  ec.solve();
}

/// Leontief test problem from Polemarchakis (rational form)
/// Has two solutions:
///
///   p = (1, 1)
///   p = (-2149/600, 1) = (-3.581667, 1)

void test7()
{
  Polemarchakis ec;
  ec.solve();
}

/// Leontief test problem from Polemarchakis (polynomial form)
/// Has two solutions:
///
///   p = (1, 1)
///   p = (-2149/600, 1) = (-3.581667, 1)

void test8()
{
  PolynomialPolemarchakis ec;
  ec.solve();
}

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("tolerance", 0.01);
  dolfin_set("discrete tolerance", 1e-10);
  dolfin_set("initial time step", 0.01);
  dolfin_set("linear solver", "direct");
  dolfin_set("adaptive samples", false);
  dolfin_set("homotopy monitoring", false);
  dolfin_set("homotopy divergence tolerance", 100.0);
  dolfin_set("homotopy randomize", false);

  //test2();

  return 0;
}
