// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Benchmark problem for the multi-adaptive ODE-solver, a system of n
// particles connected with springs. All springs, except the first spring,
// are of equal stiffness k = 1.

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
    T = 100;
    
    // Grid size
    h = 1.0 / static_cast<real>(n - 1);

    // Compute sparsity
    dependencies.detect(*this);
    //sparse();
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
      return - 100.0*x(i) + (x(i+1) - x(i) - h);
    else if ( i == (n-1) )
      return - (x(i) - x(i-1) - h) + (1.0 - x(i));
    else
      return - (x(i) - x(i-1) - h) + (x(i+1) - x(i) - h);
  }

  real k(unsigned int i)
  {
    if ( i == 0 )
      return 0.01 * 1;
    else
      return 0.1 * 1;
  }

private:

  real h;
  
};

int main()
{
  //dolfin_set("output", "plain text");
  dolfin_set("tolerance", 1e-6);
  dolfin_set("number of samples", 100);
  dolfin_set("solve dual problem", false);
  dolfin_set("fixed time step", true);
  dolfin_set("initial time step", 0.001);
  dolfin_set("save solution", true);
  dolfin_set("method", "mcg");
  dolfin_set("order", 1);
  //dolfin_set("solver", "newton");

  Benchmark bench(1000);
  bench.solve();
  
  return 0;
}
