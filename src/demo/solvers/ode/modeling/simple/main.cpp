// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-04-04
// Last changed: 2005-12-19

#include <dolfin.h>

using namespace dolfin;

class Simple : public ParticleSystem
{
public:
  
  Simple() : ParticleSystem(2, 1)
  {
    // Final time
    T = 100.0;
    //T = 4e-7;

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
  set("tolerance", 0.1);
  set("initial time step", 0.0000000001);
  set("solve dual problem", false);
  set("number of samples", 1000);
  set("automatic modeling", true);
  set("average length", 0.0000001);
  set("average samples", 1000);

  Simple simple;
  simple.solve();
  
  return 0;
}
