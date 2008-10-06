// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-02-03
// Last changed: 2006-08-21
//
// This example demonstrates the solution of a complex-valued ODE:
// 
//     z'(t) = j*z(t) on (0,10]
//     z(0)  = 1
//
// where j is the imaginary unit. The exact solution of this system
// is given by
//
//     z(t) = exp(j*t) = (cos(t), sin(t)).

#include <dolfin.h>

using namespace dolfin;

class Exponential : public ComplexODE
{
public:
  
  Exponential() : ComplexODE(1, 10.0)
  {
  }
  
  void z0(complex z[])
  {
    z[0] = 1.0;
  }

  void f(const complex z[], double t, complex y[])
  {
    y[0] = j*z[0];
  }

};

int main()
{
  Exponential exponential;
  exponential.solve();

  return 0;
}
