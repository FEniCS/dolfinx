// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin.h>

using namespace dolfin;

class TestProblem5 : public ODE
{
public:
  
  TestProblem5() : ODE(6)
  {
    dolfin_info("The Chemical Akzo-Nobel problem.");

    // Final time
    T = 180;

    // Parameters
    k1  = 18.7;
    k2  = 0.58;
    k3  = 0.09;
    k4  = 0.42;
    K   = 34.4;
    klA = 3.3;
    Ks  = 115.83;
    p   = 0.9;
    H   = 737.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    switch (i) {
    case 0:
      return 0.444;
    case 1:
      return 0.00123;
    case 2:
      return 0.0;
    case 3:
      return 0.007;
    case 4:
      return 0.0;
    default:
      return 0.36;
    }
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    switch (i) {
    case 0:
      return -2.0*r1(u) + r2(u) - r3(u) - r4(u);
    case 1:
      return -0.5*r1(u) - r4(u) - 0.5*r5(u) + F(u);
    case 2:
      return r1(u) - r2(u) + r3(u);
    case 3:
      return -r2(u) + r3(u) - 2.0*r4(u);
    case 4:
      return r2(u) - r3(u) + r5(u);
    default:
      return Ks*u(0)*u(3) - u(5);
    }
  }

private:

  real r1(const Vector& u)
  {
    return k1*pow(u(0),4.0)*sqrt(u(1));
  }
  
  real r2(const Vector& u)
  {
    return k2*u(2)*u(3);
  }

  real r3(const Vector& u)
  {
    return (k2/K)*u(0)*u(4);
  }

  real r4(const Vector& u)
  {
    return k3*u(0)*pow(u(3),2.0);
  }

  real r5(const Vector& u)
  {
    return k4*pow(u(5),2.0)*sqrt(u(1));
  }

  real F(const Vector& u)
  {
    return klA * (p/H - u(1));
  }

  real k1, k2, k3, k4, K, klA, Ks, p, H;

};
