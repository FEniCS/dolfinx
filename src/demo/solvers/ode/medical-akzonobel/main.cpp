// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class MedicalAkzoNobel : public ODE {
public:

  MedicalAkzoNobel(unsigned int n) : ODE(2*n)
  {
    T     = 20.0;
    h     = 1.0 / static_cast<real>(n);
    hinv  = 1.0 / h;
    h2inv = hinv * hinv;
    cinv  = 1.0 / 4.0;
    
    setSparsity();
  }

  real u0(unsigned int i)
  {
    if ( i % 2 == 0 )
      return 0.0;
    else
      return 1.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {    
    if ( i % 2 == 0 )
    {
      // Compute the parameters
      real tmp0 = static_cast<real>(i/2+1)*h - 1.0;
      real tmp1 = cinv*tmp0*tmp0;
      real a = 2.0*cinv*tmp0*tmp1;
      real b = tmp1*tmp1;

      // Set boundary conditions
      real ul = 0.0;
      real ur = 0.0;
      if ( i == 0 )
      {
	ul = (t <= 5.0 ? 2.0 : 0.0);
	ur = u(i+2);
      }
      else if ( i == (N - 2) )
      {
	ul = u(i-2);
	ur = u(i);
      }
      else
      {
	ul = u(i-2);
	ur = u(i+2);
      }

      return 0.5*a*hinv*(ur - ul) + b*h2inv*(ul - 2*u(i) + ur) - 100.0*u(i)*u(i+1);
    }
    else
      return - 100.0*u(i)*u(i-1);
  }
  
private:

  void setSparsity()
  {
    sparsity.clear();

    for (unsigned int i = 0; i < N; i++)
    {
      if ( i % 2 == 0 )
      {
	if ( i == 0 )
	{
	  sparsity.setsize(i, 3);
	  sparsity.set(i, i);
	  sparsity.set(i, i+1);
	  sparsity.set(i, i+2);
	}
	else if ( i == (N - 2) )
	{
	  sparsity.setsize(i, 3);
	  sparsity.set(i, i);
	  sparsity.set(i, i+1);
	  sparsity.set(i, i-2);
	}
	else
	{
	  sparsity.setsize(i, 4);
	  sparsity.set(i, i);
	  sparsity.set(i, i+1);
	  sparsity.set(i, i-2);
	  sparsity.set(i, i+2);
	}
      }
      else
      {
	sparsity.setsize(i, 2);
	sparsity.set(i, i);
	sparsity.set(i, i-1);
      }
    }
  }

  real h, hinv, h2inv, cinv;

};

int main()
{
  dolfin_set("method", "dg");
  dolfin_set("order", 0);
  dolfin_set("tolerance", 1e-3);
  dolfin_set("initial time step", 0.001);
  dolfin_set("maximum time step", 0.01);
  //dolfin_set("fixed time step", true);
  //dolfin_set("initial time step", 0.01);
  dolfin_set("number of samples", 10);
  dolfin_set("solve dual problem", false);

  MedicalAkzoNobel ode(20);
  ode.solve();  
  
  return 0;
}
