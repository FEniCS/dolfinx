// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
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
    v = p1 - p0;
    
    // Current fixed point
    pos = -1;

    // Number of laps around the two fixed points
    n0 = 0;
    n1 = 0;

    // Open file
    fp = fopen("lorenz.data", "w");

    // Final time
    //T = 1e5;
    T = 50.0;
  }
  
  ~Lorenz()
  {
    fclose(fp);
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

  void f(const real u[], real t, real y[])
  {
    y[0] = s*(u[1] - u[0]);
    y[1] = r*u[0] - u[1] - u[0]*u[2];
    y[2] = u[0]*u[1] - b*u[2];
  }

  void J(const real x[], real y[], const real u[], real t)
  {
    y[0] = s*(x[1] - x[0]);
    y[1] = (r - u[2])*x[0] - x[1] - u[0]*x[2];
    y[2] = u[1]*x[0] + u[0]*x[1] - b*x[2];
  }

  bool update(const real u[], real t, bool end)
  {
    // Check in which region the point is
    Point p(u[0], u[1], u[2]);

    if ( (p - p0) * v < 0 )
    {
      if ( pos != 0 )
      {
	n0++;
	pos = 0;
	real alpha = 0.0;
	if ( n0 > 0 & n1 > 0 )
	  alpha = static_cast<real>(n0) / (static_cast<real>(n1));
	fprintf(fp, "%.12e 0 %d %d %.16e\n", t, n0, n1, alpha);
      }
    }
    else if ( (p - p1) * v > 0 )
    {
      if ( pos != 1 )
      {
	n1++;
	pos = 1;
	real alpha = 0.0;
	if ( n0 > 0 & n1 > 0 )
	  alpha = static_cast<real>(n0) / (static_cast<real>(n1) + DOLFIN_EPS);
	fprintf(fp, "%.12e 1 %d %d %.16e\n", t, n0, n1, alpha);
      }
    }
    else
      pos = -1;

    return true;
  }

private:

  // Parameters
  real s;
  real b;
  real r;

  // The two fixed points (not counting x = (0, 0, 0))
  Point p0;
  Point p1;

  // Vector p1 - p0
  Point v;
  
  // Current fixed point
  int pos;

  // Number of laps around the two fixed points
  int n0;
  int n1;

  // File pointer
  FILE* fp;

};

int main()
{
  dolfin_output("plain text");

  dolfin_set("number of samples", 500);
  dolfin_set("solve dual problem", false);
  dolfin_set("initial time step", 0.01);
  dolfin_set("fixed time step", true);
  dolfin_set("solver", "newton");
  dolfin_set("method", "cg");
  dolfin_set("order", 5);
  dolfin_set("discrete tolerance", 1e-12);
  dolfin_set("save solution", true);
  
  Lorenz lorenz;
  lorenz.solve();
  
  return 0;
}
