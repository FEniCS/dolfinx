// Copyright (C) 2002 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2002
// Last changed: 2006-08-22
//
// This demo solves the harmonic oscillator on
// the time interval (0, 4*pi) and computes the
// error for a set of methods and orders.

#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2, 4.0 * DOLFIN_PI), e(0.0) {}

  void u0(uBLASVector& u)
  {
    u[0] = 0.0;
    u[1] = 1.0;
  }

  void f(const uBLASVector& u, real t, uBLASVector& y)
  {
    y[0] = u[1];
    y[1] = - u[0];
  }

  bool update(const uBLASVector& u, real t, bool end)
  {
    if ( !end )
      return true;

    real e0 = u[0] - 0.0;
    real e1 = u[1] - 1.0;
    e = std::max(std::abs(e0), std::abs(e1));

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
  dolfin_set("ODE fixed time step", true);
  dolfin_set("ODE discrete tolerance", 1e-14);

  for (int q = 1; q <= 5; q++)
  {
    dolfin_set("output destination", "silent");
    dolfin_set("ODE method", "cg");
    dolfin_set("ODE order", q);

    Harmonic ode;
    ode.solve();
    dolfin_set("output destination", "terminal");

    message("cG(%d): e = %.3e", q, ode.error());
  }

  cout << endl;

  for (int q = 0; q <= 5; q++)
  {
    dolfin_set("output destination", "silent");
    dolfin_set("ODE method", "dg");
    dolfin_set("ODE order", q);

    Harmonic ode;
    ode.solve();
    dolfin_set("output destination", "terminal");

    message("dG(%d): e = %.3e", q, ode.error());
  }

  return 0;
}
