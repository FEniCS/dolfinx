// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Stiff test problems for the multi-adaptive solver.

#include <stdlib.h>
#include <dolfin.h>
#include "TestProblem8.h"

using namespace dolfin;

void solveTestProblem1()
{
  dolfin_info("Not implemented");
}

void solveTestProblem2()
{
  dolfin_info("Not implemented");
}

void solveTestProblem3()
{
  dolfin_info("Not implemented");
}

void solveTestProblem4()
{
  dolfin_info("Not implemented");
}

void solveTestProblem5()
{
  dolfin_info("Not implemented");
}

void solveTestProblem6()
{
  dolfin_info("Not implemented");
}

void solveTestProblem7()
{
  dolfin_info("Not implemented");
}

void solveTestProblem8()
{
  TestProblem8 testProblem;
  testProblem.solve();
}

void solveTestProblem9()
{
  dolfin_info("Not implemented");
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
    dolfin_info("  3 - a nonnormal test problem");
    dolfin_info("  4 - the HIRES problem");
    dolfin_info("  5 - the Akzo-Nobel problem");
    dolfin_info("  6 - Van der Pol's equation");
    dolfin_info("  7 - the heat equation");
    dolfin_info("  8 - chemical reactions");
    dolfin_info("  9 - a mixed stiff/nonstiff test problem");

    return 1;
  }

  // Get the number of the test problem
  int n = atoi(argv[1]);

  dolfin_set("output", "plain text");

  // DOLFIN settings
  dolfin_set("method", "dg");
  dolfin_set("order", 0);

  // Choose test problem
  switch (n) {
  case 1:
    dolfin_info("Solving test problem number 1");
    solveTestProblem1();
    break;
  case 2:
    dolfin_info("Solving test problem number 2");
    solveTestProblem2();
    break;
  case 3:
    dolfin_info("Solving test problem number 3");
    solveTestProblem3();
    break;
  case 4:
    dolfin_info("Solving test problem number 4");
    solveTestProblem4();
    break;
  case 5:
    dolfin_info("Solving test problem number 5");
    solveTestProblem5();
    break;
  case 6:
    dolfin_info("Solving test problem number 6");
    solveTestProblem6();
    break;
  case 7:
    dolfin_info("Solving test problem number 7");
    solveTestProblem7();
    break;
  case 8:
    dolfin_info("Solving test problem number 8");
    solveTestProblem8();
    break;
  case 9:
    dolfin_info("Solving test problem number 9");
    solveTestProblem9();
    break;
  default:
    dolfin_error("No such test problem.");
  }
  
  return 0;
}
