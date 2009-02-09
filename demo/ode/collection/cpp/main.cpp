// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-09
// Last changed: 2009-02-09
//
// This demo illustrates how to use the ODECollection class to solve a
// collection of ODEs all governed by the same equation but with
// different states.

#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODECollection
{
public:
  
  Harmonic(unsigned n, real T) : ODECollection(n, 2, T) {}

  void u0(real* u)
  {
    u[0] = 0.0;
    u[1] = 1.0;
  }

  void f(const real* u, real t, real* y)
  {
    y[0] = u[1];
    y[1] = - u[0];
  }

};

int main()
{
  unsigned int n = 10;
  real T = 50.0;
  
  Harmonic ode_collection(n, T);
  ODESolution u(ode_collection);

  ode_collection.solve(u, 0.0,  10.0);
  ode_collection.solve(u, 10.0, 20.0);
  ode_collection.solve(u, 20.0, 30.0);
  ode_collection.solve(u, 30.0, 40.0);
  ode_collection.solve(u, 40.0, 50.0);

  return 0;
}
