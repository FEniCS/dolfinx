// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-02-09
// Last changed: 2009-09-08
//
// This demo illustrates how to use the ODECollection class to solve a
// collection of ODEs all governed by the same equation but with
// different states.

#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODE
{
public:

  Harmonic(real T) : ODE(2, T) {}

  void u0(Array<real>& u)
  {
    cout << "Calling u0() to get initial data" << endl;

    u[0] = 0.0;
    u[1] = 1.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = u[1];
    y[1] = - u[0];
  }

};

int main()
{
  // Final time
  const real T = 50.0;

  // Number of ODE systems
  unsigned int n = 3;

  // Create ODE and collection of ODE systems
  Harmonic ode(T);
  ode.parameters["adaptive_samples"] = true;
  ODECollection collection(ode, n);

  // Set initial states for all ODE systems
  Array<real> u0(2);
  for (unsigned int i = 0; i < n; i++)
  {
    u0[0] = 2.0 * static_cast<real>(i) / static_cast<real>(n);
    u0[1] = 1.0;
    collection.set_state(i, u0);
  }

  // Time step ODE collection over a sequence of intervals
  const real k = 10.0;
  real t = 0.0;
  while (t < T)
  {
    collection.solve(t, t + k);
    t += k;
  }

  return 0;
}
