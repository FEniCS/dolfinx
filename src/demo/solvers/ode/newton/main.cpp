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
      return 0.2 * u(1);

    return 10.0 * -u(0);
  }

  real timestep(unsigned int i)
  {
    if(i == 0)
    {
      return 0.04;
    }
    else
    {
      return 0.01;
    }
  }
  
};

class SpringSystem : public ODE
{
public:
  
  SpringSystem(unsigned int N) : ODE(2*N)
  {
    // Final time
    T = 10.0;

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

  /*  
  real timestep(unsigned int i)
  {
    if(i == 3)
    {
      return 0.001;
    }
    else if(i == 2)
    {
      return 0.008;
    }
    else if(i == 1)
    {
      return 0.004;
    }
    else
    {
      return 0.008;
    }
  }
  */

};

int main()
{
  dolfin_set("output", "plain text");

  dolfin_set("tolerance", 0.01);
  dolfin_set("stiffness", "stiff");

  dolfin_set("debug iterations", true);

  dolfin_set("maximum time step", 5.0);
  //dolfin_set("fixed time step", true);
  dolfin_set("fixed time step", false);
  //dolfin_set("initial time step", 1e-06);

  //dolfin_set("partitioning threshold", 1 - 1e-07);

  dolfin_set("partitioning threshold", 0.5);

  dolfin::dolfin_set("method", "dg");
  dolfin::dolfin_set("order", 0);
  dolfin::dolfin_set("solve dual problem", false);


  //Single single;
  //single.solve();

  //Harmonic harmonic;
  //harmonic.solve();
  
  SpringSystem springSystem(2);
  springSystem.solve();

  return 0;
}
