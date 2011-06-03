// Copyright (C) 2003-2005 Johan Jansson
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
// Modified by Anders Logg 2003-2006.
//
// First added:  2003
// Last changed: 2006-08-21

#include <dolfin.h>

using namespace dolfin;

class TestProblem8 : public ODE
{
public:

  TestProblem8() : ODE(3, 0.3)
  {
    info("System of fast and slow chemical reactions, taken from the book by");
    info("Hairer and Wanner, page 3.");
  }

  void u0(Array<real>& u)
  {
    u[0] = 1.0;
    u[1] = 0.0;
    u[2] = 0.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = -0.04 * u[0] + 1.0e4 * u[1] * u[2];
    y[1] = 0.04 * u[0] - 1.0e4 * u[1] * u[2] - 3.0e7 * u[1] * u[1];
    y[2] = 3.0e7 * u[1] * u[1];
  }

};
