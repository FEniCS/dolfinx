// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Simple : public ParticleSystem
{
public:
  
  Simple() : ParticleSystem(2, 1)
  {
    // Final time
    T = 100.0;

    // The large spring constant
    k = 1e18;
    
    // Compute sparsity
    sparse();
  }

  real x0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;
    else
      return 1.0;
  }

  real Fx(unsigned int i, real t)
  {
    if ( i == 0 )
      return -x(0) + 0.5*x(1)*x(1);
    else
      return -k*x(1);
  }

private:

  real k;
  
};

int main()
{
  dolfin_set("tolerance", 0.1);
  dolfin_set("initial time step", 0.0000000001);
  dolfin_set("solve dual problem", false);
  dolfin_set("number of samples", 1000);
  dolfin_set("automatic modeling", true);
  dolfin_set("average length", 0.0000001);
  dolfin_set("average samples", 1000);

  Simple simple;
  simple.solve();
  
  return 0;
}
