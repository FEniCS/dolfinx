// Copyright (C) 2003-2008 Anders Logg
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
// Last changed: 2008-10-07

#include <dolfin.h>

using namespace dolfin;

class TestProblem7 : public ODE
{
public:

  TestProblem7() : ODE(101, 1.0)
  {
    h = 1.0 / (static_cast<real>(N) - 1);
    info("The heat equation on [0,1] with h = %f", to_double(h));
  }

  void u0(Array<real>& u)
  {
    for (unsigned int i = 0; i < N; i++)
      u[i] = 0.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    // Boundary values
    y[0]   = 0.0;
    y[N-1] = 0.0;

    // Interior values
    for (unsigned int i = 1; i < N - 1; i++)
    {
      // Heat source
      real source = 0.0;
      if ( i == N/2 )
	source = 100.0;

      y[i] = (u[i-1] - 2.0*u[i] + u[i+1]) / (h*h) + source;
    }
  }

private:

  real h;

};
