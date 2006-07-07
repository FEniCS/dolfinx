// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-07-07
//
// Stiff test problems for the ODE solver.

#include <string>
#include <iostream>
#include <dolfin.h>
#include "TestProblem1.h"
#include "TestProblem2.h"
#include "TestProblem3.h"
#include "TestProblem4.h"
#include "TestProblem5.h"
#include "TestProblem6.h"
#include "TestProblem7.h"
#include "TestProblem8.h"
#include "TestProblem9.h"

using namespace dolfin;

void solveTestProblem1()
{
  TestProblem1 testProblem;
  testProblem.solve();
}

void solveTestProblem2()
{
  TestProblem2 testProblem;
  testProblem.solve();
}

void solveTestProblem3()
{
  TestProblem3 testProblem;
  testProblem.solve();
}

void solveTestProblem4()
{
  TestProblem4 testProblem;
  testProblem.solve();
}

void solveTestProblem5()
{
  TestProblem5 testProblem;
  testProblem.solve();
}

void solveTestProblem6()
{
  TestProblem6 testProblem;
  testProblem.solve();
}

void solveTestProblem7()
{
  TestProblem7 testProblem;
  testProblem.solve();
}

void solveTestProblem8()
{
  TestProblem8 testProblem;
  testProblem.solve();
}

void solveTestProblem9()
{
  TestProblem9 testProblem;
  testProblem.solve();
}

int main(int argc, char* argv[])
{
  // Check arguments
  if ( argc != 2 )
  {
    dolfin_info("Usage: dolfin-ode-stiff-testproblems n");
    dolfin_info("");
    dolfin_info("where n is one of");
    dolfin_info("");
    dolfin_info("  1 - the test equation");
    dolfin_info("  2 - the test system");
    dolfin_info("  3 - a non-normal test problem");
    dolfin_info("  4 - the HIRES problem");
    dolfin_info("  5 - the Chemical Akzo-Nobel problem");
    dolfin_info("  6 - Van der Pol's equation");
    dolfin_info("  7 - the heat equation");
    dolfin_info("  8 - a chemical reaction test problem");
    dolfin_info("  9 - a mixed stiff/nonstiff test problem");

    return 1;
  }

  // Get the number of the test problem
  int n = atoi(argv[1]);

  // Parameters
  set("ODE method", "dg");
  set("ODE order", 1);
  set("ODE maximum time step", 5.0);
  set("ODE tolerance", 0.01);
  set("ODE nonlinear solver", "newton");
  set("ODE adaptive samples", true);
  set("ODE solve dual problem", false);

  // Choose test problem
  switch (n) {
  case 1:
    dolfin_info("Solving test problem number 1.");
    solveTestProblem1();
    break;
  case 2:
    dolfin_info("Solving test problem number 2.");
    solveTestProblem2();
    break;
  case 3:
    dolfin_info("Solving test problem number 3.");
    solveTestProblem3();
    break;
  case 4:
    dolfin_info("Solving test problem number 4.");
    solveTestProblem4();
    break;
  case 5:
    dolfin_info("Solving test problem number 5.");
    solveTestProblem5();
    break;
  case 6:
    dolfin_info("Solving test problem number 6.");
    solveTestProblem6();
    break;
  case 7:
    dolfin_info("Solving test problem number 7.");
    solveTestProblem7();
    break;
  case 8:
    dolfin_info("Solving test problem number 8.");
    solveTestProblem8();
    break;
  case 9:
    dolfin_info("Solving test problem number 9.");
    solveTestProblem9();
    break;
  default:
    dolfin_error("No such test problem.");
  }
  
  return 0;
}
