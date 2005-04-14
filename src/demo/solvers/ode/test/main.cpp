// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Simple : public ODE
{
public:
  
  Simple() : ODE(1)
  {
    T = 0.05;
  }
  
  real u0(unsigned int i)
  {
    return 0.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
    return 0.0;
  }

};

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2)
  {
    // Final time
    T = 3.0;

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
    T = 0.64;

    // Compute sparsity
    //dependencies.detect(*this);
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

  //real dfdu(real u[], real t, unsigned int i, unsigned int j)
  // {
  //  real a[3][3] = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
  //  return a[i][j];
  // }

  real timestep(unsigned int i)
  {
    switch ( i ) {
    case 0:
      return 0.16;
    case 1:
      return 0.04;
    default:
      return 0.01;
    }
  }
  
};

int main()
{
  dolfin_set("solve dual problem", false);
  dolfin_set("initial time step", 0.01);
  dolfin_set("fixed time step", true);
  dolfin_set("solver", "fixed point");
  dolfin_set("discrete tolerance", 1e-10);
  dolfin_set("number of samples", 200);
  //dolfin_set("solver", "newton");
  //dolfin_set("implicit", true);
  dolfin_set("method", "mcg");
  dolfin_set("order", 1);

  Simple ode;
  ode.solve();

  //Harmonic ode;
  //ode.solve();
  
  //SpringSystem ode(10);
  //ode.solve();

  //TestSystem ode;
  //ode.solve();

  return 0;
}
