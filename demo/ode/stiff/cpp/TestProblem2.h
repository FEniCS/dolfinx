// Copyright (C) 2004-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004
// Last changed: 2008-10-07

#include <dolfin.h>

using namespace dolfin;

class TestProblem2 : public ODE
{
public:
  
  TestProblem2() : ODE(2, 10.0)
  {
    message("The simple test system.");
  }

  void u0(double* u)
  {
    u[0] = 1.0;
    u[1] = 1.0;
  }
  
  void f(const double* u, double t, double* y)
  {
    y[0] = -100.0*u[0];
    y[1] = -1000.0*u[1];
  }
  
};
