// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.
// 
// Benchmark problem for the multi-adaptive ODE-solver, a system of n
// particles connected with springs. All springs, except the first
// spring, are of equal stiffness k = 1.

#include <iostream>
#include <sstream>
#include <dolfin.h>

using namespace dolfin;

class Benchmark : public ParticleSystem
{
public:
  
  Benchmark(unsigned int n, unsigned int M, real b) :
    ParticleSystem(n, 1), k(10*M), b(b), M(M)
  {
    if ( n < 2 )
      dolfin_error("System must have at least 2 particles.");
    
    cout << "b: " << b << endl;
    cout << "M: " << M << endl;

    // Final time
    T = 1.0;
    
    // Grid size
    h = 0.1;

    // Compute sparsity
    computeSparsity();
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
      return - k*x(i) + (x(i+1) - x(i) - h);
    
    if ( i == 1 )
      return - (x(i) - x(i-1) - h) + (x(i+1) - x(i) - h) +
	b * (vx(i+1) - vx(i));
    
    if ( i == (n-1) )
      return - (x(i) - x(i-1) - h) - b * (vx(i) - vx(i-1));
    
    return - (x(i) - x(i-1) - h) + (x(i+1)  - x(i) - h) -
      b*(vx(i) - vx(i-1)) + b*(vx(i+1) - vx(i));
  }

  real timestep(unsigned int i)
  {
    if ( i == 0 )
      return 0.1 / static_cast<real>(M);
    if ( i == n )
      return 0.1 / static_cast<real>(M);
    return 0.1;
  }

protected:

  void computeSparsity()
  {
    Matrix S(2*n, 2*n);

    for (unsigned int i = 0; i < n; i++)
    {
      S(i, n + i) = 1;
      S(n + i, i) = 1;
      S(n + i, n + i) = 1;

      if (i != 0)
      {
	S(n + i, i - 1) = 1;
	S(n + i, n + i - 1) = 1;
      }

      if (i != n - 1)
      {
	S(n + i, i + 1) = 1;
	S(n + i, n + i + 1) = 1;
      }
    }

    sparsity.set(S);
  }

  real k;
  real b;
  real M;
  real h;
  
};

int main(int argC, char* argV[])
{
  int n  = 100;
  int M  = 100;
  real b = 100.0;

  std::string method = "mcg";

  if (argC > 1)
  {
    std::string arg = argV[1];
    std::istringstream argstream(arg);
    
    argstream >> n;
  }

  if (argC > 2)
  {
    std::string arg = argV[2];
    std::istringstream argstream(arg);
    
    argstream >> M;
  }

  if (argC > 3)
  {
    std::string arg = argV[3];
    std::istringstream argstream(arg);
    
    argstream >> b;
  }

  if (argC > 4)
  {
    std::string arg = argV[4];
    if(arg == "mcg")
      method = "mcg";
    else if(arg == "mdg")
      method = "mdg";
    else if(arg == "cg")
      method = "cg";
    else if(arg == "dg")
      method = "dg";
    else
      dolfin_error("Unknown method.");
  }

  dolfin_set("output", "plain text");
  dolfin_set("tolerance", 0.1);
  dolfin_set("solve dual problem", false);
  //dolfin_set("save solution", false);
  dolfin_set("partitioning threshold", 0.99);
  dolfin_set("fixed time step", true);  
  dolfin_set("maximum iterations", 20000);
  //dolfin_set("initial time step", 0.01);
  //dolfin_set("fixed time step", true);

  cout << "Creating problem with " << n << " masses" << endl;
  
  if (method == "mcg")
  {
    dolfin_set("method", "cg");
    dolfin_set("order", 1);

    Benchmark bench(n, M, b);
    bench.solve();
  }
  else if (method == "mdg")
  {
    dolfin_set("method", "dg");
    dolfin_set("order", 0);
    
    Benchmark bench(n, M, b);
    bench.solve();
  }
  else if (method == "cg")
  {
    dolfin_set("method", "cg");
    dolfin_set("order", 1);
    dolfin_set("partitioning threshold", 1e-7);
    
    Benchmark bench(n, M, b);
    bench.solve();
  }
  else if (method == "dg")
  {
    dolfin_set("method", "dg");
    dolfin_set("order", 0);
    dolfin_set("partitioning threshold", 1e-7);

    Benchmark bench(n, M, b);
    bench.solve();
  }

  return 0;
}
