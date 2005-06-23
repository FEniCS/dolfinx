// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <stdio.h>
#include <dolfin.h>

using namespace dolfin;

class Simple : public ODE
{
public:
  
  Simple() : ODE(1)
  {
    T = 1.0;
  }
  
  real u0(unsigned int i)
  {
    return 0.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
    return 1.0;
  }

};

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2)
  {
    // Final time
    T = 4.0*DOLFIN_PI;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    return 1.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
    if ( i == 0 )
      return u[1];

    return -u[0];
  }

  bool update(const real u[], real t, bool end)
  {
    if ( !end )
      return true;

    real e0 = u[0] - 0.0;
    real e1 = u[1] - 1.0;
    real e = std::max(fabs(e0), fabs(e1));
    dolfin_info("Error: %.3e", e);

    return true;
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

class TestSystem : public ODE
{
public:
  
  TestSystem() : ODE(3)
  {
    // Final time
    T = 64.0;
  }

  real u0(unsigned int i)
  {
    switch ( i ) {
    case 0:
      return 0.0;
    case 1:
      return 1.0;
    default:
      return 2.0;
    }
  }

  real f(const real u[], real t, unsigned int i)
  {
    switch ( i ) {
    case 0:
      return u[1] + 0.1*u[0] + 0.3*u[2];
    case 1:
      return -u[0] - 0.2*u[1] + 0.4*u[2];
    default:
      return cos(t) + 0.3*u[0] + 0.7*u[1] * 0.2*u[2];
    }
  }

};

int main()
{
  //Simple ode;
  //ode.solve();

  Harmonic ode;
  ode.solve();
  
  //SpringSystem ode(10);
  //ode.solve();

  //TestSystem ode;
  //ode.solve();

  return 0;
}
