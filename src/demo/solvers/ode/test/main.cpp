// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include <dolfin/Partition.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

class Minimal : public ODE
{
public:
  
  Minimal() : ODE(3)
  {
    // Parameters
    lambda1 = 1.0;
    lambda2 = 1.0;

    // Final time
    T = 20.0;

    // Initial value
    //u0(0) = 1.0;
    //u0(1) = 1.0;

    u0(0) = 0.0;
    u0(1) = 1.0;
    u0(2) = 1.0;

    // Compute sparsity
    sparse();
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    //dolfin_debug("foo");
    //dolfin::cout << "u: " << dolfin::endl;
    //u.show();

    if (i == 0)
    {
      //return -lambda1 * u(0);
      return u(1);
    }
    else if (i == 1)
    {
      //return -lambda2 * u(1);
      return -u(0);
    }
    else if (i == 2)
    {
      return -lambda2 * u(2);
      //return -u(0);
    }
    return 0;
  }

private:

  real lambda1, lambda2;

};

class SpringSystem : public ODE
{
public:
  
  SpringSystem(unsigned int N) : ODE(N)
  {
    // Final time
    T = 5.0;

    // Initial value
    u0 = 0.0;

    // Compute sparsity
    sparse();
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    real k = (real) (i+1);   
    return -k*u(i);
  }

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("debug time slab", 1);
  dolfin_set("tolerance", 0.1);
  dolfin_set("initial time step", 0.1);
  dolfin_set("maximum time step", 1.0);
  dolfin_set("partitioning threshold", 1.0);
  dolfin_set("interval threshold", 0.9);
  dolfin_set("number of samples", 100);

  Minimal minimal;
  minimal.solve();
  
  //SpringSystem springSystem(10);
  //springSystem.solve();

  return 0;
}
