// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Spring : public MechanicalSystem
{
public:
  
  Spring() : MechanicalSystem(1, 1)
  {
    // Final time
    T = 30.0;

    // Spring constant
    k = 5.0;

    // Damping constant
    b = 0.0;

    // Compute sparsity
    sparse();
  }

  real x0(unsigned int i)
  {
    return 1.0;
  }

  real v0(unsigned int i)
  {
    return 0.0;
  }

  real Fx(unsigned int i, real t)
  {
    return - k * x(0) - b * vx(0);
  }

private:

  real k;
  real b;
  
};

int main()
{
  dolfin_set("tolerance", 0.01);

  Spring spring;
  spring.solve();
  
  return 0;
}
