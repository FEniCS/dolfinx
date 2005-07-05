// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2005

#include <dolfin.h>

using namespace dolfin;

class TestProblem2 : public ODE
{
public:
  
  TestProblem2() : ODE(2), A(2,2)
  {
    dolfin_info("The simple test system.");

    // Final time
    T = 10;

    // The matrix A
    A(0,0) = 100.0;
    A(1,1) = 1000.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    return 1.0;
  }
  
  real f(const real u[], real t, unsigned int i)
  {
    return -A.mult(u, i);
  }

private:

  Matrix A;
  
};
