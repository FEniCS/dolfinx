// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

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

private:

  real lambda;

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("debug time steps", 1);
  dolfin_set("tolerance", 0.01);

  Simple simple;
  simple.solve();
  
  return 0;
}
