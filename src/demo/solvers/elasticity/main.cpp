// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t, int i)
{
  if(i == 0)
  {
    if(t > 0.0 && t < 0.5 && x > 0.99)
    {
      return -10.0;
    }
    else if(t > 0.0 && t < 0.5 && x < 0.01)
    {
      return 10.0;
    }
  }


  return 0.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
  
  /*
  if ( bc.coord().x == 0.0 )
  {
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  */
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);

  /*
  // u = 0 on the inflow boundary
  if ( bc.coord().x == 0.0 )
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else if ( bc.coord().x == 1.0 )
    bc.set(BoundaryCondition::DIRICHLET, 0.0);
  else if ( bc.coord().y == 0.0)
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else if ( bc.coord().y == 1.0 )
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else
    bc.set(BoundaryCondition::DIRICHLET, 1.0);
  */
}

int main(int argc, char **argv)
{
  dolfin_set("output", "plain text");

  //Mesh mesh("dolfin.xml.gz");
  //Mesh mesh("dolfin.xml.gz");
  //Mesh mesh("trimesh-16.xml.gz");
  //Mesh mesh("trimesh-2.xml.gz");
  //Mesh mesh("trimesh-4.xml.gz");
  //Mesh mesh("trimesh-64.xml.gz");
  //Mesh mesh("trimesh-32.xml.gz");

  //Mesh mesh("tetmesh-4.xml.gz");
  //Mesh mesh("tetmesh-8.xml.gz");

  Mesh mesh("unitcube0.xml.gz");

  Problem elasticity("elasticity", mesh);

  elasticity.set("source", f);
  elasticity.set("boundary condition", mybc);
  elasticity.set("final time", 2.0);
  elasticity.set("time step", 0.1);

  elasticity.solve();
  
  return 0;
}
