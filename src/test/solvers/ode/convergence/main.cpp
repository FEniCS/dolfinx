// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Benchmark problem for the multi-adaptive ODE-solver, a system of n
// particles connected with springs. All springs, except the first spring,
// are of equal stiffness k = 1.

#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2)
  {
    // Final time
    T = 30.0;

    // Compute sparsity
    dependencies.detect(*this);
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 1.0;

    return 1.0;
  }

    /// Return right-hand side for ODE
  real f(const real u[], real t, unsigned int i)
  {
    if ( i == 0 )
      return -1.0 * u[0];

    return -5.0 * u[1];
  }
  real timestep(unsigned int i)
  {
    if ( i == 0 )
      return 2.0;
    else
      return 2.0;
  }
};

int main()
{
  //dolfin_set("output", "plain text");
  dolfin_set("tolerance", 1e-4);
  dolfin_set("number of samples", 100);
  dolfin_set("solve dual problem", false);
  dolfin_set("fixed time step", true);
  dolfin_set("partitioning threshold", 0.99);
  //dolfin_set("initial time step", 0.1);
  dolfin_set("save solution", true);
  dolfin_set("use new ode solver", true);
  dolfin_set("method", "mdg");
  dolfin_set("order", 0);
  //dolfin_set("solver", "newton");

  Harmonic harmonic;
  harmonic.solve();
  
  return 0;
}
