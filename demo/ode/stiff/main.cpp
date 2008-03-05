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
    message("Usage: dolfin-ode-stiff-testproblems n");
    message("");
    message("where n is one of");
    message("");
    message("  1 - the test equation");
    message("  2 - the test system");
    message("  3 - a non-normal test problem");
    message("  4 - the HIRES problem");
    message("  5 - the Chemical Akzo-Nobel problem");
    message("  6 - Van der Pol's equation");
    message("  7 - the heat equation");
    message("  8 - a chemical reaction test problem");
    message("  9 - a mixed stiff/nonstiff test problem");

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
      message("Solving test problem number 1.");
      dolfin_set("ODE solution file name", "solution_1.py");
      TestProblem1 testProblem;
      testProblem.solve();
    }
    break;
  case 2:
    {
      message("Solving test problem number 2.");
      dolfin_set("ODE solution file name", "solution_2.py");
      TestProblem2 testProblem;
      testProblem.solve();
    }
    break;
  case 3:
    {
      message("Solving test problem number 3.");
      dolfin_set("ODE solution file name", "solution_3.py");
      TestProblem3 testProblem;
      testProblem.solve();
    }
    break;
  case 4:
    {
      message("Solving test problem number 4.");
      dolfin_set("ODE solution file name", "solution_4.py");
      TestProblem4 testProblem;
      testProblem.solve();
    }
    break;
  case 5:
    {
      message("Solving test problem number 5.");
      dolfin_set("ODE solution file name", "solution_5.py");
      TestProblem5 testProblem;
      testProblem.solve();
    }
    break;
  case 6:
    {
      message("Solving test problem number 6.");
      dolfin_set("ODE solution file name", "solution_6.py");
      TestProblem6 testProblem;
      testProblem.solve();
    }
    break;
  case 7:
    {
      message("Solving test problem number 7.");
      dolfin_set("ODE solution file name", "solution_7.py");
      TestProblem7 testProblem;
      testProblem.solve();
    }
    break;
  case 8:
    {
      message("Solving test problem number 8.");
      dolfin_set("ODE solution file name", "solution_8.py");
      TestProblem8 testProblem;
      testProblem.solve();
    }
    break;
  case 9:
    {
      message("Solving test problem number 9.");
      dolfin_set("ODE solution file name", "solution_9.py");
      TestProblem9 testProblem;
      testProblem.solve();
    }
    break;
  default:
    error("No such test problem.");
  }
  
  return 0;
}
