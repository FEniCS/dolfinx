// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Heat : public ODE
{
public:
  
  Heat(unsigned int N) : ODE(N)
  {
    dolfin_assert(N >= 2);

    T = 2.0;
    h = 1.0 / (static_cast<real>(N) - 1);
    
    sparse();
  }

  real u0(unsigned int i)
  {
    return 0.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    // Boundary values
    if ( i == 0 || i == (N-1) )
      return 0.0;
    
    // Heat source
    real source = 0.0;
    if ( i == N/3 )
      source = 10.0;

    return (u(i-1) - 2.0*u(i) + u(i+1)) / (h*h) + source;
  }

private:
  
  real h;

};

int main()
{
  dolfin_set("tolerance", 1e-3);
  dolfin_set("method", "dg");
  dolfin_set("order", 0);
  dolfin_set("solve dual problem", false);

  Heat heat(10);
  heat.solve();
  
  return 0;
}
