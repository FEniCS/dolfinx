// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2005-10-24
// Last changed: 2008-08-26

#ifndef __NONLINEAR_PROBLEM_TEST_H
#define __NONLINEAR_PROBLEM_TEST_H

#include <dolfin/log/log.h>

namespace dolfin
{

  class NonlinearProblemTest
  {
  public:

    /// Constructor
    NonlinearProblemTest() {}

    /// Destructor
    virtual ~NonlinearProblemTest() {}

    virtual void F(double a, double b) = 0;

  };
}

#endif
