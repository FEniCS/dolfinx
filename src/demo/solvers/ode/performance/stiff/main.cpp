// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Benchmark problem for the multi-adaptive ODE-solver, a system of n
// particles connected with springs. All springs, except the first spring,
// are of equal stiffness k = 1.

#include <iostream>
#include <sstream>

#include <dolfin.h>

using namespace dolfin;

class Benchmarkcg : public ParticleSystem
{
public:
  
  void computeSparsity()
  {
    Matrix S(2 * n, 2 * n);
    int deps = 0;

    for(unsigned int i = 0; i < n; i++)
    {
      S(i, n + i) = 1;
      S(n + i, i) = 1;
      S(n + i, n + i) = 1;
      deps += 2;

      if(i != 0)
      {
	S(n + i, i - 1) = 1;
	S(n + i, n + i - 1) = 1;
	deps++;
      }

      if(i != n - 1)
      {
	S(n + i, i + 1) = 1;
	S(n + i, n + i + 1) = 1;
	deps++;
      }
    }

    sparsity.set(S);

    cout << "deps: " << deps << endl;
  }

  Benchmarkcg(unsigned int n, unsigned int m, real b) :
    ParticleSystem(n, 1), b(b), m(m)
  {
    if ( n < 2 )
      dolfin_error("System must have at least 2 particles.");

    cout << "b: " << b << endl;
    cout << "m: " << m << endl;

    // Final time
    T = 1.0;

    // Spring constant
    k = 1.0;

    // Damping constant
    //b = 100.0;
    
    // Grid size
    //h = 1.0 / static_cast<real>(n - 1);
    h = 0.1;

    // Compute sparsity
    //sparse();
    computeSparsity();

    //sparsity.show();
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
    //std::cerr << "eIndex: " << i << std::endl;

    if ( i == 0 )
      return - 100.0*x(i) + k*(x(i+1) - x(i) - h);
    if ( i == 1 )
      return - k*(x(i) - x(i-1) - h) + k*(x(i+1) - x(i) - h) +
	b * (vx(i+1) - vx(i));
    else if ( i == (n-1) )
      return - k*(x(i) - x(i-1) - h) -
	b * (vx(i) - vx(i - 1));
    else
      return - k*(x(i) - x(i-1) - h) + k*(x(i+1) - x(i) - h) -
	b * (vx(i) - vx(i - 1)) + b * (vx(i+1) - vx(i));
  }

  real timestep(unsigned int i)
  {
    if(i == 0)
      return 0.1 / m;
    if(i == n)
      return 0.1 / m;
    return 0.1;
  }

protected:

  real k;
  real b;
  real h;
  real m;
  
};


class Benchmarkdg : public Benchmarkcg
{
public:

  Benchmarkdg(unsigned int n, unsigned int m, real b) : Benchmarkcg(n, m, b)
  {
  }

  real x0(unsigned int i)
  {
    if ( i == 0 )
      return h/2;

    return h * static_cast<real>(i);
  }
};

int main(int argC, char* argV[])
{
  int n = 100;
  int m = 100;
  real b = 100.0;

  std::string method = "mcg";

  if(argC > 1)
  {
    std::string arg = argV[1];
    std::istringstream argstream(arg);
    
    argstream >> n;
  }

  if(argC > 2)
  {
    std::string arg = argV[2];
    std::istringstream argstream(arg);
    
    argstream >> m;
  }

  if(argC > 3)
  {
    std::string arg = argV[3];
    std::istringstream argstream(arg);
    
    argstream >> b;
  }

  if(argC > 4)
  {
    std::string arg = argV[4];
    if(arg == "mcg")
      method = "mcg";
    if(arg == "mdg")
      method = "mdg";
    if(arg == "cg")
      method = "cg";
    if(arg == "dg")
      method = "dg";
  }

  dolfin_set("output", "plain text");
  dolfin_set("tolerance", 0.1);
  dolfin_set("number of samples", 111);
  dolfin_set("solve dual problem", false);
  dolfin_set("save solution", false);
  //dolfin_set("save solution", true);
  dolfin_set("partitioning threshold", 0.99);
  //dolfin::dolfin_set("partitioning threshold", 1e-7);
  dolfin::dolfin_set("fixed time step", true);

  //dolfin_set("initial time step", 0.01);
  //dolfin_set("fixed time step", true);
  dolfin_set("maximum iterations", 20000);



  cout << "Creating problem with " << n << " masses" << endl;


  if(method == "mcg")
  {
    Benchmarkcg bench(n, m, b);
    bench.solve();
  }

  if(method == "mdg")
  {
    dolfin_set("method", "dg");
    dolfin_set("order", 0);

    Benchmarkdg bench(n, m, b);
    bench.solve();
  }

  if(method == "cg")
  {
    dolfin::dolfin_set("partitioning threshold", 1e-7);

    Benchmarkcg bench(n, m, b);
    bench.solve();
  }

  if(method == "dg")
  {
    dolfin::dolfin_set("partitioning threshold", 1e-7);

    dolfin_set("method", "dg");
    dolfin_set("order", 0);

    Benchmarkdg bench(n, m, b);
    bench.solve();
  }

  return 0;
}
