// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t, int i)
{
  if(i == 0)
  {
    if (x > 0.9 && y > 0.9 && t < 1.0)
      return 100.0;
    else
      return 0.0;
  }
  else
  {
    if (x > 0.6 && y > 0.6 && t < 0.2)
      return 100.0;
    else
      return 0.0;
  }
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
  bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
}

int main(int argc, char **argv)
{
  Mesh mesh("trimesh-32.xml.gz");
  Problem wavevector("wave-vector", mesh);

  wavevector.set("source", f);
  wavevector.set("boundary condition", mybc);
  wavevector.set("final time", 3.0);
  wavevector.set("time step", 0.01);

  wavevector.solve();
  
  return 0;
}
