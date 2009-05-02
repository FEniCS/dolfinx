// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003
// Last changed: 2006-08-21
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

int main(int argc, char* argv[])
{
  // Check arguments
  if ( argc != 2 )
  {
    info("Usage: dolfin-ode-stiff-testproblems n");
    info("");
    info("where n is one of");
    info("");
    info("  1 - the test equation");
    info("  2 - the test system");
    info("  3 - a non-normal test problem");
    info("  4 - the HIRES problem");
    info("  5 - the Chemical Akzo-Nobel problem");
    info("  6 - Van der Pol's equation");
    info("  7 - the heat equation");
    info("  8 - a chemical reaction test problem");
    info("  9 - a mixed stiff/nonstiff test problem");

    return 1;
  }

  // Get the number of the test problem
  int n = atoi(argv[1]);

  // Parameters
  dolfin_set("ODE method", "dg");
  dolfin_set("ODE order", 1);
  dolfin_set("ODE maximum time step", 5.0);
  dolfin_set("ODE tolerance", 0.01);
  dolfin_set("ODE nonlinear solver", "newton");
  dolfin_set("ODE adaptive samples", true);
  dolfin_set("ODE solve dual problem", false);

  // Choose test problem
  switch (n) {
  case 1:
    {
      info("Solving test problem number 1.");
      dolfin_set("ODE solution file name", "solution_1.py");
      TestProblem1 test_problem;
      test_problem.solve();
    }
    break;
  case 2:
    {
      info("Solving test problem number 2.");
      dolfin_set("ODE solution file name", "solution_2.py");
      TestProblem2 test_problem;
      test_problem.solve();
    }
    break;
  case 3:
    {
      info("Solving test problem number 3.");
      dolfin_set("ODE solution file name", "solution_3.py");
      TestProblem3 test_problem;
      test_problem.solve();
    }
    break;
  case 4:
    {
      info("Solving test problem number 4.");
      dolfin_set("ODE solution file name", "solution_4.py");
      TestProblem4 test_problem;
      test_problem.solve();
    }
    break;
  case 5:
    {
      info("Solving test problem number 5.");
      dolfin_set("ODE solution file name", "solution_5.py");
      TestProblem5 test_problem;
      test_problem.solve();
    }
    break;
  case 6:
    {
      info("Solving test problem number 6.");
      dolfin_set("ODE solution file name", "solution_6.py");
      TestProblem6 test_problem;
      test_problem.solve();
    }
    break;
  case 7:
    {
      info("Solving test problem number 7.");
      dolfin_set("ODE solution file name", "solution_7.py");
      TestProblem7 test_problem;
      test_problem.solve();
    }
    break;
  case 8:
    {
      info("Solving test problem number 8.");
      dolfin_set("ODE solution file name", "solution_8.py");
      TestProblem8 test_problem;
      test_problem.solve();
    }
    break;
  case 9:
    {
      info("Solving test problem number 9.");
      dolfin_set("ODE solution file name", "solution_9.py");
      TestProblem9 test_problem;
      test_problem.solve();
    }
    break;
  default:
    error("No such test problem.");
  }

  return 0;
}
