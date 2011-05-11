// Copyright (C) 2006-2008 Anders Logg
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
// First added:  2006-05-24
// Last changed: 2008-10-07
//
// This demo solves a simple model for cardiac excitation,
// proposed in a 1995 paper by Aliev and Panfilov.

#include <dolfin.h>

using namespace dolfin;

class AlievPanfilov : public ODE
{
public:

  AlievPanfilov() : ODE(2, 300.0)
  {
    // Set parameters
    a    = 0.15;
    eps0 = 0.002;
    k    = 8.0;
    mu1  = 0.07;
    mu2  = 0.3;
  }

  void u0(Array<real>& u)
  {
    u[0] = 0.2;
    u[1] = 0.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    const real eps = eps0 + mu1*u[1] / (u[0] + mu2);

    y[0] = -k*u[0]*(u[0] - a)*(u[0] - 1.0) - u[0]*u[1];
    y[1] = eps*(-u[1] - k*u[0]*(u[0] - a - 1.0));
  }

private:

  real a;
  real eps0;
  real k;
  real mu1;
  real mu2;

};

int main()
{
  AlievPanfilov ode;
  ode.solve();

  return 0;
}
