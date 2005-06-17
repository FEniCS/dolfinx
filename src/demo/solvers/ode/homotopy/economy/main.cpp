// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "CES.h"
#include "Leontief.h"
#include "Polemarchakis.h"
#include "EavesSchmedders.h"

using namespace dolfin;

/// CES test problem from Eaves and Schmedders (rational-rational form).
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test1()
{
  RationalRationalES ec;
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (polynomial-rational form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test2()
{
  PolynomialRationalES ec;
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (rational-integer form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test3()
{
  RationalIntegerES ec;
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (polynomial-integer form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test4()
{
  PolynomialIntegerES ec;
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (rational-rational form).
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test5()
{
  RationalRationalCES ec(2, 2);

  ec.a[0][0] = 4.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 4.0;
  
  ec.w[0][0] = 10.0; ec.w[0][1] =  1.0;
  ec.w[1][0] =  1.0; ec.w[1][1] = 12.0;
  
  ec.b[0] = 0.2;
  ec.b[1] = 0.2;    

  ec.disp();
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (polynomial-rational form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test6()
{
  PolynomialRationalCES ec(2, 2);

  ec.a[0][0] = 4.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 4.0;
  
  ec.w[0][0] = 10.0; ec.w[0][1] =  1.0;
  ec.w[1][0] =  1.0; ec.w[1][1] = 12.0;
  
  ec.b[0] = 0.2;
  ec.b[1] = 0.2;    

  ec.disp();
  ec.solve();
}

/// CES test problem from Eaves and Schmedders (polynomial-integer form)
/// Has 3 solutions (one real-valued):
///
///   p = (0.951883, 0.048117)
///   p = (0.026816 + 0.309611*i, 0.973184 - 0.309611*i)
///   p = (0.026816 - 0.309611*i, 0.973184 + 0.309611*i)

void test7()
{
  PolynomialIntegerCES ec(2, 2);

  ec.a[0][0] = 4.0; ec.a[0][1] = 1.0;
  ec.a[1][0] = 1.0; ec.a[1][1] = 4.0;
  
  ec.w[0][0] = 10.0; ec.w[0][1] =  1.0;
  ec.w[1][0] =  1.0; ec.w[1][1] = 12.0;
  
  ec.beta[0] = 1;
  ec.beta[1] = 1;

  ec.alpha = 5;

  ec.disp();
  ec.solve();
}

/// Leontief test problem (rational form)
/// Has one unique solution: p = (0.5, 0.5)
/// (Also p = (0, 1) with first equation removed)

void test8()
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

void test9()
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

void test10()
{
  Polemarchakis ec;
  ec.solve();
}

/// Leontief test problem from Polemarchakis (polynomial form)
/// Has two solutions:
///
///   p = (1, 1)
///   p = (-2149/600, 1) = (-3.581667, 1)

void test11()
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
  dolfin_set("initial time step", 0.001);
  //dolfin_set("linear solver", "direct");
  dolfin_set("adaptive samples", false);
  dolfin_set("homotopy monitoring", false);
  dolfin_set("homotopy divergence tolerance", 10.0);
  dolfin_set("homotopy randomize", false);

  // Test  1: CES test problem from Eaves and Schmedders (rational-rational form).
  // Test  2: CES test problem from Eaves and Schmedders (polynomial-rational form)
  // Test  3: CES test problem from Eaves and Schmedders (rational-integer form)
  // Test  4: CES test problem from Eaves and Schmedders (polynomial-integer form)
  // Test  5: CES test problem from Eaves and Schmedders (rational-rational form).
  // Test  6: CES test problem from Eaves and Schmedders (polynomial-rational form)
  // Test  7: CES test problem from Eaves and Schmedders (polynomial-integer form)
  // Test  8: Leontief test problem (rational form)
  // Test  9: Leontief test problem (polynomial form)
  // Test 10: Leontief test problem from Polemarchakis (rational form)
  // Test 11: Leontief test problem from Polemarchakis (polynomial form)
  //
  // Test 1 and 5 should give the same result (one solution).
  // Test 2 and 6 should give the same result (two solutions).
  // Test 4 and 7 should give the same result (six solutions, including extras and multiples).
  //
  // Test 4 and 7 find the most solutions.
  
  test7();

  return 0;
}
