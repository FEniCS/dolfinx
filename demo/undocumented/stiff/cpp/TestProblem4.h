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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2004
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem4 : public ODE
{
public:

  TestProblem4() : ODE(8, 321.8122)
  {
    info("The HIRES problem.");
  }

  void u0(Array<real>& u)
  {
    u[0] = 1.0;
    u[1] = 0.0;
    u[2] = 0.0;
    u[3] = 0.0;
    u[4] = 0.0;
    u[5] = 0.0;
    u[6] = 0.0;
    u[7] = 0.0057;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = -1.71*u[0] + 0.43*u[1] + 8.32*u[2] + 0.0007;
    y[1] = 1.71*u[0] - 8.75*u[1];
    y[2] = -10.03*u[2] + 0.43*u[3] + 0.035*u[4];
    y[3] = 8.32*u[1] + 1.71*u[2] - 1.12*u[3];
    y[4] = -1.745*u[4] + 0.43*u[5] + 0.43*u[6];
    y[5] = -280.0*u[5]*u[7] + 0.69*u[3] + 1.71*u[4] - 0.43*u[5] + 0.69*u[6];
    y[6] = 280.0*u[5]*u[7] - 1.81*u[6];
    y[7] = -280.0*u[5]*u[7] + 1.81*u[6];
  }

};
