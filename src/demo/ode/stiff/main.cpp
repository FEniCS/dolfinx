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
    {
      dolfin_info("Solving test problem number 1.");
      set("ODE solution file name", "solution_1.py");
      TestProblem1 testProblem;
      testProblem.solve();
    }
    break;
  case 2:
    {
      dolfin_info("Solving test problem number 2.");
      set("ODE solution file name", "solution_2.py");
      TestProblem2 testProblem;
      testProblem.solve();
    }
    break;
  case 3:
    {
      dolfin_info("Solving test problem number 3.");
      set("ODE solution file name", "solution_3.py");
      TestProblem3 testProblem;
      testProblem.solve();
    }
    break;
  case 4:
    {
      dolfin_info("Solving test problem number 4.");
      set("ODE solution file name", "solution_4.py");
      TestProblem4 testProblem;
      testProblem.solve();
    }
    break;
  case 5:
    {
      dolfin_info("Solving test problem number 5.");
      set("ODE solution file name", "solution_5.py");
      TestProblem5 testProblem;
      testProblem.solve();
    }
    break;
  case 6:
    {
      dolfin_info("Solving test problem number 6.");
      set("ODE solution file name", "solution_6.py");
      TestProblem6 testProblem;
      testProblem.solve();
    }
    break;
  case 7:
    {
      dolfin_info("Solving test problem number 7.");
      set("ODE solution file name", "solution_7.py");
      TestProblem7 testProblem;
      testProblem.solve();
    }
    break;
  case 8:
    {
      dolfin_info("Solving test problem number 8.");
      set("ODE solution file name", "solution_8.py");
      TestProblem8 testProblem;
      testProblem.solve();
    }
    break;
  case 9:
    {
      dolfin_info("Solving test problem number 9.");
      set("ODE solution file name", "solution_9.py");
      TestProblem9 testProblem;
      testProblem.solve();
    }
    break;
  default:
    dolfin_error("No such test problem.");
  }
  
  return 0;
}
