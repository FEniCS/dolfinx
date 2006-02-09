// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2005-12-19

#include <dolfin.h>

using namespace dolfin;

class Spring : public ParticleSystem
{
public:
  
  Spring() : ParticleSystem(1, 1)
  {
    // Final time
    T = 30.0;

    // Spring constant
    k = 5.0;
    
    // Damping constant
    b = 1.0;

    // Compute sparsity
    sparse();
  }

  real x0(unsigned int i)
  {
    return 1.0;
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
  set("tolerance", 0.01);
  set("solver", "newton");

  Spring spring;
  spring.solve();
  
  return 0;
}
