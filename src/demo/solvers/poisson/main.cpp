// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

//dolfin_bc myBC(real x, real y, real z, int node, int component);

// Source term
real f(real x, real y, real z, real t)
{
  real pi = DOLFIN_PI;
  return 14.0 * pi*pi * sin(pi*x) * sin(2.0*pi*x) * sin(3.0*pi*x);
}

using namespace dolfin;

void main()
{
  Grid grid("grid.xml.gz");
  Problem poisson("poisson", grid);

  poisson.set("source", f);
  //poisson.set("boundary conditions", myBC);
 
  poisson.solve();
}

/*
dolfin_bc my_bc(real x, real y, real z, int node, int component)
{
  dolfin_bc bc;

  if ( x == 0.0 ){
	 bc.type = dirichlet;
	 bc.val  = 0.0;
  }
  if ( x == 1.0 ){
	 bc.type = dirichlet;
	 bc.val  = 0.0;
  }

  return bc;
}
*/

