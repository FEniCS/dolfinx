// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Benchmark problem for the multi-adaptive ODE-solver,
// a system of n particles connected with springs.

#include <dolfin.h>

using namespace dolfin;

class Benchmark : public ParticleSystem
{
public:
  
  Benchmark(unsigned int n) : ParticleSystem(n, 1)
  {
    if ( n < 2 )
      dolfin_error("System must have at least 2 particles.");

    // Final time
    T = 100.0;

    // Spring constant
    k = 1.0;
    
    // Damping constant
    b = 0.0;

    // Grid size
    h = 1.0 / static_cast<real>(n - 1);

    // Compute sparsity
    sparse();
  }

  real x0(unsigned int i)
  {
    if ( i == 0 )
      return h/2;

    return h * static_cast<real>(i);
  }

  real v0(unsigned int i)
  {
    return 0.0;
  }

  real Fx(unsigned int i, real t)
  {
    if ( i == 0 )
      return - k*x(i) + k*(x(i+1) - x(i) - h);
    else if ( i == (n-1) )
      return - k*(x(i) - x(i-1) - h) + k*(1.0 - x(i));
    else
      return - k*(x(i) - x(i-1) - h) + k*(x(i+1) - x(i) - h);
  }

private:

  real k;
  real b;
  real h;
  
};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("tolerance", 0.01);
  dolfin_set("solve dual problem", false);

  Benchmark bench(100);
  bench.solve();
  
  return 0;
}
