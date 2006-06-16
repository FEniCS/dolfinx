// Copyright (C) 2002 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-05-29

#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2, 4.0 * DOLFIN_PI), e(0.0)
  {
    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;
    return 1.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
    if ( i == 0 )
      return u[1];
    return -u[0];
  }

  void f(const real u[], real t, real y[])
  {
    y[0] = u[1];
    y[1] = -u[0];
  }

  bool update(const real u[], real t, bool end)
  {
    if ( !end )
      return true;

    real e0 = u[0] - 0.0;
    real e1 = u[1] - 1.0;
    e = std::max(fabs(e0), fabs(e1));

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

    dolfin_info("cG(%d): e = %g", q, ode.error());
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

    dolfin_info("dG(%d): e = %g", q, ode.error());
  }

  return 0;
}
