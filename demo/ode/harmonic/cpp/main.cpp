// Copyright (C) 2002-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2002
// Last changed: 2009-09-08
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

  void u0(real* u)
  {
    u[0] = 0.0;
    u[1] = 1.0;
  }

  void f(const real* u, real t, real* y)
  {
    y[0] = u[1];
    y[1] = - u[0];
  }

  bool update(const real* u, real t, bool end)
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
    ode.parameters["discrete_tolerance"] = 1e-14;
    ode.parameters["method"] = "cg";
    ode.parameters["order"] = q;
    ode.solve();

    logging(true);
    info("cG(%d): e = %.3e", q, to_double(ode.error()));
  }

  dolfin::cout << dolfin::endl;

  for (int q = 0; q <= 8; q++)
  {
    logging(false);

    Harmonic ode;
    ode.parameters["fixed_time_step"] = true;
    ode.parameters["discrete_tolerance"] = 1e-14;
    ode.parameters["method"] = "dg";
    ode.parameters["order"] = q;

    ode.solve();

    logging(true);
    info("dG(%d): e = %.3e", q, to_double(ode.error()));
  }

  return 0;
}
