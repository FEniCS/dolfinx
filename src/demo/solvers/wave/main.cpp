// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t)
{
  /*
  if(x > 0.9 && y > 0.9 && t < 1.0)
    return 100.0;
  else
    return 0.0;
  */

  return 0.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  //bc.set(BoundaryCondition::DIRICHLET, 0.0);
}

int main()
{
  dolfin_set("output", "plain text");

  //Mesh mesh("trimesh-32.xml.gz");
  Mesh mesh("trimesh-1.xml.gz");

  /*
  Problem wave("wave", mesh);

  wave.set("source", f);
  wave.set("boundary condition", mybc);
  wave.set("final time", 10.0);
  wave.set("time step", 0.01);

  wave.solve();
  */

  return 0;
}
