// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Lorenz : public ODE {
public:

  Lorenz() : ODE(3)
  {
    // Parameters
    s = 10.0;
    b = 8.0 / 3.0;
    r = 28.0;

    // Fixed points
    p0.x = 6.0*sqrt(2.0); p0.y = p0.x; p0.z = 27.0;
    p1.x = -p0.x; p1.y = -p0.y; p1.z = p0.z;

    // Distance between fixed points
    d = p0.dist(p1);
    
    // Current fixed point
    pos = -1;

    // Final time
    T = 50.0;
  }

  real u0(unsigned int i)
  {
    switch (i) {
    case 0:
      return 1.0;
    case 1:
      return 0.0;
    default:
      return 0.0;
    }
  }

  real f(real u[], real t, unsigned int i)
  {
    switch (i) {
    case 0:
      return s*(u[1] - u[0]);
    case 1:
      return r*u[0] - u[1] - u[0]*u[2];
    default:
      return u[0]*u[1] - b*u[2];
    }
  }

  void update(real u[], real t)
  {
    // Compute distance two the two fixed points
    Point p(u[0], u[1], u[2]);
    real d0 = p.dist(p0);
    real d1 = p.dist(p1);
    
    dolfin_info("debug %.16e %.16e %.16e", t, d0, d1);

    // Check where we are
    if ( d0 < d && d1 > d && pos != 0 )
    {
      cout << "Changing to fixed point 0 at t = " << t << endl;
      pos = 0;
    }
    else if ( d1 < d && d0 > d && pos != 1 )
    {
      cout << "Changing to fixed point 1 at t = " << t << endl;
      pos = 1;
    }
    else
      pos = -1;
  }

private:

  // Parameters
  real s;
  real b;
  real r;

  // The two fixed points (not counting x = (0, 0, 0))
  Point p0;
  Point p1;

  // Distance between the two fixed points
  real d;
  
  // Current fixed point
  int pos;

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("number of samples", 500);
  dolfin_set("solve dual problem", false);
  dolfin_set("initial time step", 0.01);
  dolfin_set("fixed time step", true);
  dolfin_set("use new ode solver", true);
  dolfin_set("method", "cg");
  dolfin_set("order", 10);
  dolfin_set("tolerance", 1e-12);
  //dolfin_set("save solution", false);
  
  Lorenz lorenz;
  lorenz.solve();
  
  return 0;
}
