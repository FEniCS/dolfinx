// Copyright (C) 2002 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Updates by Anders Logg 2003.

#include <dolfin.h>
#include <dolfin/Partition.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

class Reaction : public ODE
{
public:
  
  Reaction() : ODE(3)
  {
    // Final time
    T = 0.5;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 1.0;
    else
      return 0.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    if ( i == 0 )
      return -0.04 * u(0) + 1.0e4 * u(1) * u(2);

    if ( i == 1 )
      return 0.04 * u(0) - 1.0e4 * u(1) * u(2) - 3.0e7 * u(1) * u(1);

    return 3.0e7 * u(1) * u(1);
  }
  
};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("debug time steps", 1);
  dolfin_set("tolerance", 0.1);
  dolfin_set("initial time step", 2e-3);
  dolfin_set("maximum time step", 2e-3);
  dolfin_set("partitioning threshold", 1.0);
  dolfin_set("interval threshold", 0.9);
  dolfin_set("number of samples", 100);
  dolfin_set("element cache size", 32);

  Reaction reaction;
  reaction.solve();
  
  return 0;
}
