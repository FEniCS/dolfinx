// Copyright (C) 2005-2006 Anders Logg
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

  Exponential() : ComplexODE(1, 10.0) {}

  void z0(complex z[])
  {
    z[0] = 1.0;
  }

  void f(const complex z[], real t, complex y[])
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
