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
    T = 2.0;
    
    // Grid size
    h = 1.0 / static_cast<real>(n - 1);

    num_fevals = 0;

    // Compute sparsity
    sparse();
  }

  ~Benchmark()
  {
    dolfin_info("Number of fevals: %d", num_fevals);
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
    num_fevals += 1;

    if ( i == 0 )
      return - 100.0*x(i) + (x(i+1) - x(i) - h);
    else if ( i == (n-1) )
      return - (x(i) - x(i-1) - h) + (1.0 - x(i));
    else
      return - (x(i) - x(i-1) - h) + (x(i+1) - x(i) - h);
  }

private:

  real h;
  
  unsigned int num_fevals;
  
};

int main()
{
  //dolfin_set("output", "plain text");
  dolfin_set("discrete tolerance", 1e-4);
  dolfin_set("number of samples", 100);
  dolfin_set("solve dual problem", false);
  dolfin_set("fixed time step", true);
  dolfin_set("initial time step", 0.01);
  dolfin_set("save solution", false);
  dolfin_set("method", "mcg");
  dolfin_set("order", 1);

  Benchmark bench(1000);
  bench.solve();
  
  return 0;
}
