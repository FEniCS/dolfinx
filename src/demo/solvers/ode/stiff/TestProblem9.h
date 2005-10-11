// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005

#include <dolfin.h>

using namespace dolfin;

class TestProblem9 : public ODE
{
public:
  
  TestProblem9() : ODE(3, 30.0)
  {
    dolfin_info("A mixed stiff/nonstiff test problem.");

    // Parameters
    lambda = 1000.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    if ( i == 1 )
      return 1.0;

    return 1.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
    if ( i == 0 )
      return u[1];

    if ( i == 1 )
      return -(1.0 - u[2])*u[0];

    return -lambda * (u[0]*u[0] + u[1]*u[1]) * u[2];
  }

private:

  real lambda;

};
