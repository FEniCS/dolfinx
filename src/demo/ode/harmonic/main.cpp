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

  void u0(uBlasVector& u)
  {
    u(0) = 0.0;
    u(1) = 1.0;
  }

  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    y(0) = u(1);
    y(1) = - u(0);
  }

  bool update(const uBlasVector& u, real t, bool end)
  {
    if ( !end )
      return true;

    real e0 = u(0) - 0.0;
    real e1 = u(1) - 1.0;
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
  set("ODE fixed time step", true);
  set("ODE discrete tolerance", 1e-14);

  for (int q = 1; q <= 5; q++)
  {
    dolfin_log(false);
    set("ODE method", "cg");
    set("ODE order", q);

    Harmonic ode;
    ode.solve();
    dolfin_log(true);

    message("cG(%d): e = %.3e", q, ode.error());
  }

  cout << endl;

  for (int q = 0; q <= 5; q++)
  {
    dolfin_log(false);
    set("ODE method", "dg");
    set("ODE order", q);

    Harmonic ode;
    ode.solve();
    dolfin_log(true);

    message("dG(%d): e = %.3e", q, ode.error());
  }

  return 0;
}
