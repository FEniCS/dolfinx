// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2009.
//
// First added:  2005-10-23
// Last changed: 2009-06-29

#ifndef __NEWTON_SOLVER_TEST_H
#define __NEWTON_SOLVER_TEST_H

namespace dolfin
{

  class NonlinearProblemTest;

  class NewtonSolverTest
  {
  public:

    NewtonSolverTest();

    ~NewtonSolverTest();

    void solve(NonlinearProblemTest& nonlinear_function, double x);

  };

}

#endif

