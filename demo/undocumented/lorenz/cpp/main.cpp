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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet, 2008.
//
// First added:  2003-07-02
// Last changed: 2009-09-08

#include <stdio.h>
#include <dolfin.h>
#include <dolfin/common/real.h>

using namespace dolfin;

class Lorenz : public ODE
{
public:

  Lorenz() : ODE(3, 50)
  {
    // Parameters
    s = 10.0;
    b = 8.0 / 3.0;
    r = 28.0;
  }

  void u0(Array<real>& u)
  {
    u[0] = 1.0;
    u[1] = 0.0;
    u[2] = 0.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = s*(u[1] - u[0]);
    y[1] = r*u[0] - u[1] - u[0]*u[2];
    y[2] = u[0]*u[1] - b*u[2];
  }

  void J(const Array<real>& x, Array<real>& y, const Array<real>& u, real t)
  {
    y[0] = s*(x[1] - x[0]);
    y[1] = (r - u[2])*x[0] - x[1] - u[0]*x[2];
    y[2] = u[1]*x[0] + u[0]*x[1] - b*x[2];
  }

  void JT(const Array<real>& x, Array<real>& y, const Array<real>& u, real t)
  {
    y[0] = -x[0]*s + (r-u[2])*x[1] + u[1]*x[2];
    y[1] = s*x[0] - x[1] + u[0]*x[2];
    y[2] = -u[0]*x[1] - b*x[2];
  }

private:

  // Parameters
  real s;
  real b;
  real r;

};

int main()
{
  Lorenz lorenz;

  lorenz.parameters["number_of_samples"] = 500;
  lorenz.parameters["initial_time_step"] = 0.01;
  lorenz.parameters["fixed_time_step"] = true;
  lorenz.parameters["method"] = "cg";
  lorenz.parameters["order"] = 5;
  lorenz.parameters["discrete_tolerance"] = 1e-10;
  lorenz.parameters["save_solution"] = true;

  ODESolution u;

  lorenz.solve(u);
  lorenz.analyze_stability_computation(u);
  

  return 0;
}
