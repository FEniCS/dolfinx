// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include <dolfin/Partition.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

class Simple : public ODE
{
public:
  
  Simple() : ODE(3)
  {
    // Parameters
    lambda = 1000.0;

    // Final time
    T = 30.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    if ( i == 1 )
      return 1.0;

    return 1.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    if ( i == 0 )
      return u(1);

    if ( i == 1 )
      return -u(0);

    return -lambda * u(2);
  }

  Element::Type method(unsigned int i)
  {
    if ( i == 2 )
      return Element::dg;
    
    return Element::cg;
  }

  unsigned int order(unsigned int i)
  {
    if ( i == 2 )
      return 0;
    
    return 1;
  }

  /*
  real timestep(unsigned int i)
  {
    if ( i == 2 )
      return 1e-3;

    return 0.1;
  }
  */
  
private:

  real lambda;

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("debug time steps", 1);
  dolfin_set("tolerance", 0.01);
  dolfin_set("initial time step", 0.1);
  //dolfin_set("maximum time step", 1.0);
  //dolfin_set("fixed time step", true);
  dolfin_set("partitioning threshold", 1.0);
  dolfin_set("interval threshold", 0.9);
  dolfin_set("number of samples", 100);
  dolfin_set("element cache size", 32);
  dolfin_set("maximum iterations", 100);

  Simple simple;
  simple.solve();
  
  return 0;
}
