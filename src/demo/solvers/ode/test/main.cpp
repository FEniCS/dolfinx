// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Single : public ODE
{
public:
  
  Single() : ODE(1)
  {
    T = 30.0;
  }
  
  real u0(unsigned int i)
  {
    return 0.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    return cos(t);
  }

};

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2)
  {
    // Final time
    T = 30.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    return 1.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    if ( i == 0 )
      return u(1);

    return -u(0);
  }
  
};

class SpringSystem : public ODE
{
public:
  
  SpringSystem(unsigned int N) : ODE(2*N)
  {
    // Final time
    T = 5.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    return 1.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    if ( i < N / 2 )
      return u(i+N/2);
    
    real k = (real) (i+1);
    return -k*u(i-N/2);
  }

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("tolerance", 0.01);
  dolfin_set("debug time steps", 1);

  //Single single;
  //single.solve();

  Harmonic harmonic;
  harmonic.solve();
  
  //SpringSystem springSystem(10);
  //springSystem.solve();

  return 0;
}
