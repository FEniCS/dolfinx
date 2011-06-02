// Copyright (C) 2002-2008 Anders Logg
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
// First added:  2002
// Last changed: 2010-09-02
//
// This demo solves the harmonic oscillator on
// the time interval (0, 4*pi) and computes the
// error for a set of methods and orders.

#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODE
{
public:

  Harmonic() : ODE(2, 4.0 * real_pi()), e(0.0) {}

  void u0(Array<real>& u)
  {
    u[0] = 0.0;
    u[1] = 1.0;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = u[1];
    y[1] = - u[0];
  }

  bool update(const Array<real>& u, real t, bool end)
  {
    if ( !end )
      return true;

    real e0 = u[0] - 0.0;
    real e1 = u[1] - 1.0;
    e = real_max(real_abs(e0), real_abs(e1));

    return true;
  }

  real error()
  {
    return e;
  }

private:

  real e;

};

int main()
{
  for (int q = 1; q <= 8; q++)
  {
    logging(false);

    Harmonic ode;
    ode.parameters["fixed_time_step"] = true;
    ode.parameters["discrete_tolerance"] = real_epsilon();
    ode.parameters["method"] = "cg";
    ode.parameters["order"] = q;
    ode.solve();

    //ODESolution u;
    //ode.solve(u);
    //ode.solve_dual(u);

    logging(true);
    info("cG(%d): e = %.3e", q, to_double(ode.error()));
  }

  cout << endl;

  for (int q = 0; q <= 8; q++)
  {
    logging(false);

    Harmonic ode;
    ode.parameters["fixed_time_step"] = true;
    ode.parameters["discrete_tolerance"] = real_epsilon();
    ode.parameters["method"] = "dg";
    ode.parameters["order"] = q;
    ode.solve();

    //ODESolution u;
    //ode.solve(u);
    //ode.solve_dual(u);

    logging(true);
    info("dG(%d): e = %.3e", q, to_double(ode.error()));
  }

  return 0;
}
