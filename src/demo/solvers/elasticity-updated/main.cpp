// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t, int i)
{
  return 0.0;
}

// Initial velocity
real v0(real x, real y, real z, real t, int i)
{
  if(i == 0)
    return 1.0;

  return 0.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  ///*
  if ( bc.coord().x == 0.0 )
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  ///*/
}

int main(int argc, char **argv)
{
  //Mesh mesh("dolfin.xml.gz");
  //Mesh mesh("dolfin.xml.gz");
  //Mesh mesh("trimesh-16.xml.gz");
  //Mesh mesh("trimesh-2.xml.gz");
  //Mesh mesh("trimesh-4.xml.gz");
  //Mesh mesh("trimesh-64.xml.gz");
  //Mesh mesh("trimesh-32.xml.gz");

  dolfin_set("output", "plain text");

  //Mesh mesh("minimal.xml.gz");
  //Mesh mesh("minimal.xml.gz");
  //Mesh mesh("tetmesh-1.xml.gz");
  Mesh mesh("tetmesh-4.xml.gz");
  //Mesh mesh("tetmesh-8.xml.gz");

  Problem elasticity("elasticity-updated", mesh);
  //Problem elasticity("elasticity", mesh);

  elasticity.set("source", f);
  elasticity.set("initial velocity", v0);
  elasticity.set("boundary condition", mybc);
  elasticity.set("final time", 2.0);
  elasticity.set("time step", 0.01);

  elasticity.solve();
  
  return 0;
}
