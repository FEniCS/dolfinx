// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

dolfin_bc myBC(real x, real y, real z, int node, int component);
real f(real x, real y, real z, real t);

using namespace dolfin;

int main()
{
  Grid grid("grid.xml.gz");  
  Problem poisson("poisson", grid);
  
  poisson.set("source", f);
  poisson.set("boundary conditions", myBC);
  poisson.set("space dimensions", 3);

  poisson.solve();
}

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

real f(real x, real y, real z, real t)
{
  real dx = x - 0.5;
  real dy = y - 0.5;
  real r  = sqrt( dx*dx + dy*dy );
  
  if ( r < 0.3 )
	 return 100.0;
  else
	 return 0.0;
}
