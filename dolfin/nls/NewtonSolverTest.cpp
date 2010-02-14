// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2009.
// Modified by Martin Alnes, 2008.
//
// First added:  2005-10-23
// Last changed: 2009-09-08

#include "NonlinearProblemTest.h"
#include "NewtonSolverTest.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolverTest::NewtonSolverTest()
{

}
//-----------------------------------------------------------------------------
NewtonSolverTest::~NewtonSolverTest()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonSolverTest::solve(NonlinearProblemTest& nonlinear_problem, 
                              double x)
{
  double aa(0.0), bb(0.0);
  nonlinear_problem.F(aa, bb);
}
//-----------------------------------------------------------------------------

