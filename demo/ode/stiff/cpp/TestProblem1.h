// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem1 : public ODE
{
public:
  
  TestProblem1() : ODE(1, 10.0)
  {
    message("The simple test equation: u' = -1000 u, u(0) = 1.");
  }

  void u0(uBlasVector& u)
  {
    u(0) = 1.0;
  }
  
  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    y(0) = -1000.0 * u(0);
  }
  
};
