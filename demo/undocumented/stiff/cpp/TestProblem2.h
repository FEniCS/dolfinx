// Copyright (C) 2004-2008 Anders Logg
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
// Last changed: 2008-10-07

#include <dolfin.h>

using namespace dolfin;

class TestProblem2 : public ODE
{
public:

  TestProblem2() : ODE(2, 10.0)
  {
    info("The simple test system.");
  }

  void u0(Array<real>& u)
  {
    u[0] = 1.0;
    u[1] = 1.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = -100.0*u[0];
    y[1] = -1000.0*u[1];
  }

};
