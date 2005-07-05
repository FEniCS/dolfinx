// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2005

#include <cmath>
#include <dolfin.h>

using namespace dolfin;

class TestProblem6 : public ODE
{
public:
  
  TestProblem6() : ODE(2)
  {
    dolfin_info("Van der Pol's equation.");

    // Final time
    T = 100.0;

    // Parameters
    mu = 10.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    switch (i) {
    case 0:
      return 2.0;
    default:
      return 0.0;
    }
  }

  real f(const real u[], real t, unsigned int i)
  {
    switch (i) {
    case 0:
      return u[1];
    default:
      return mu*(1.0 - u[0]*u[0])*u[1] - u[0];
    }
  }

private:

  real mu;

};
