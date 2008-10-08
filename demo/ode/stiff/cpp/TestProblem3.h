// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003
// Last changed: 2008-10-07

#include <dolfin.h>

using namespace dolfin;

class TestProblem3 : public ODE
{
public:
  
  TestProblem3() : ODE(2, 1.0)
  {
    message("A non-normal test problem, critically damped oscillation.");
  }

  void u0(double* u)
  {
    u[0] = 1.0;
    u[1] = 1.0;
  }

  void f(const double* u, double t, double* y)
  {
    y[0] = -1000*u[0] + 10000*u[1];
    y[1] = -100*u[1];
  }

};
