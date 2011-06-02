// Copyright (C) 2003-2006 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem9 : public ODE
{
public:

  TestProblem9() : ODE(3, 30.0)
  {
    info("A mixed stiff/nonstiff test problem.");

    lambda = 1000.0;
  }

  void u0(Array<real>& u)
  {
    u[0] = 0.0;
    u[1] = 1.0;
    u[2] = 1.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = u[1];
    y[1] = -(1.0 - u[2])*u[0];
    y[2] = -lambda * (u[0]*u[0] + u[1]*u[1]) * u[2];
  }

private:

  real lambda;

};
