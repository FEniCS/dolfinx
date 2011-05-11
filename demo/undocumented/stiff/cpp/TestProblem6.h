// Copyright (C) 2004-2006 Anders Logg
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
// First added:  2004
// Last changed: 2006-08-21

#include <cmath>
#include <dolfin.h>

using namespace dolfin;

class TestProblem6 : public ODE
{
public:

  TestProblem6() : ODE(2, 100.0)
  {
    info("Van der Pol's equation.");

    mu = 10.0;
  }

  void u0(Array<real>& u)
  {
    u[0] = 2.0;
    u[1] = 0.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = u[1];
    y[1] = mu*(1.0 - u[0]*u[0])*u[1] - u[0];
  }

private:

  real mu;

};
